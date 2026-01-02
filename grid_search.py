import argparse
import copy
import hashlib
import importlib
import importlib.util
import itertools
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

from data_generators.data_generator import DataGenerator
from main import setup_logging
from trainers.trainer import Trainer
from utils import set_seed

# Optional in-file default search space; override values to run without --search-space.
DEFAULT_SEARCH_SPACE: Dict[str, List[Any]] = {
    "alpha": [0.1],
    "beta": [0.2],
}


def parse_search_space(raw: str | None) -> Dict[str, List[Any]]:
    """
    Load a search space from:
    - a JSON string,
    - a JSON file,
    - or a Python file that defines either `SEARCH_SPACE` (dict) or `get_search_space()` returning a dict.
    The expected format is a mapping of argument names to lists of candidate values.
    """

    if raw is None:
        search_space = DEFAULT_SEARCH_SPACE
    elif os.path.isfile(raw) and raw.endswith(".py"):
        spec = importlib.util.spec_from_file_location("user_search_space", raw)
        if spec is None or spec.loader is None:
            raise argparse.ArgumentTypeError(f"Cannot load Python search space from {raw}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "SEARCH_SPACE"):
            search_space = module.SEARCH_SPACE
        elif hasattr(module, "get_search_space"):
            search_space = module.get_search_space()
        else:
            raise argparse.ArgumentTypeError(
                f"Python file {raw} must define SEARCH_SPACE or get_search_space()."
            )
    else:
        candidate = raw
        if os.path.isfile(raw):
            with open(raw, "r", encoding="utf-8") as f:
                candidate = f.read()

        try:
            search_space = json.loads(candidate)
        except json.JSONDecodeError as err:
            raise argparse.ArgumentTypeError(f"Invalid JSON for search space: {err}") from err

    if not isinstance(search_space, dict):
        raise argparse.ArgumentTypeError("Search space must be a mapping from argument names to value lists.")

    if not search_space:
        raise argparse.ArgumentTypeError("Search space is empty; provide values or set DEFAULT_SEARCH_SPACE in grid_search.py.")

    for name, values in search_space.items():
        if not isinstance(values, list) or not values:
            raise argparse.ArgumentTypeError(f"Search space for '{name}' must be a non-empty list.")
    return search_space


def iter_combinations(search_space: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(search_space.keys())
    for combo in itertools.product(*(search_space[key] for key in keys)):
        yield dict(zip(keys, combo))


def to_serializable_score(score: Any):
    if score is None:
        return None
    if isinstance(score, np.ndarray):
        return score.tolist()
    if isinstance(score, (list, tuple)):
        return list(score)
    return score


def _sanitize(value: Any) -> str:
    if isinstance(value, float):
        # avoid dots in filenames; keep enough precision for clarity
        value = f"{value}".replace(".", "p")
    elif isinstance(value, (list, tuple)):
        value = "-".join(_sanitize(v) for v in value)
    else:
        value = str(value)

    cleaned = re.sub(r"[^A-Za-z0-9\-]+", "-", value).strip("-")
    return cleaned or "val"


def format_override_slug(overrides: Dict[str, Any]) -> str:
    """Create a compact, filename-safe summary of overrides (excluding model_idx)."""

    parts = []
    for name in sorted(overrides.keys()):
        if name == "model_idx":
            continue
        parts.append(f"{name}{_sanitize(overrides[name])}")

    if not parts:
        return "default"

    slug = "_".join(parts)
    if len(slug) > 60:
        digest = hashlib.md5(slug.encode("utf-8")).hexdigest()[:8]
        slug = f"{slug[:50]}_{digest}"
    return slug


def load_config_module(module_path: str):
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as err:
        raise argparse.ArgumentTypeError(f"Cannot import config module '{module_path}': {err}") from err

    if not hasattr(module, "get_config"):
        raise argparse.ArgumentTypeError(f"Config module '{module_path}' must expose a get_config function.")
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a grid search over hyperparameters for Mixed-diffusion models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-module",
        default="configs.cadirec_config",
        help="Python module path for the config parser to use (must provide get_config).",
    )
    parser.add_argument(
        "--search-space",
        required=False,
        help="JSON string or JSON file describing the grid, e.g. '{\"alpha\": [0.1, 0.2], \"beta\": [0.1, 0.2]}'.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optionally cap the total number of combinations to execute.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and list planned runs without actually training models.",
    )
    return parser


def run_single_experiment(
    base_args, overrides: Dict[str, Any], run_index: int
) -> Dict[str, Any]:
    run_args = copy.deepcopy(base_args)
    for name, value in overrides.items():
        setattr(run_args, name, value)

    if "model_idx" not in overrides:
        slug = format_override_slug(overrides)
        run_args.model_idx = f"{base_args.model_idx}-gs{run_index}-{slug}"

    run_name = f"{run_args.model_name}-{run_args.dataset}-{run_args.model_idx}"
    os.makedirs(run_args.output_dir, exist_ok=True)
    log_path = setup_logging(run_args, run_name)
    run_args.checkpoint_path = os.path.join(run_args.output_dir, run_name + ".pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("=== Grid search run %d (%s) on device %s ===", run_index, overrides, device)
    logging.info("Checkpoint: %s", run_args.checkpoint_path)
    logging.info("Logs: %s", log_path)

    set_seed(run_args.seed)
    data_generator = DataGenerator(run_args)
    trainer = Trainer(run_args, device, data_generator)
    outcome = trainer.train() or {}

    def pick_score(key: str):
        attr_score = getattr(trainer, key, None)
        if attr_score is not None:
            return to_serializable_score(attr_score)
        return to_serializable_score(outcome.get(key))

    return {
        "index": run_index,
        "overrides": overrides,
        "log_path": log_path,
        "checkpoint_path": run_args.checkpoint_path,
        "best_valid_score": pick_score("best_valid_score"),
        "best_test_score": pick_score("best_test_score"),
    }


def main():
    parser = build_parser()
    args, remaining = parser.parse_known_args()

    try:
        search_space = parse_search_space(args.search_space)
        config_module = load_config_module(args.config_module)
    except argparse.ArgumentTypeError as err:
        parser.error(str(err))

    base_args = config_module.get_config(remaining)

    results = []
    for idx, overrides in enumerate(iter_combinations(search_space)):
        if args.max_runs is not None and idx >= args.max_runs:
            break

        if args.dry_run:
            results.append(
                {
                    "index": idx,
                    "overrides": overrides,
                    "note": "dry run (training skipped)",
                }
            )
            continue

        results.append(run_single_experiment(base_args, overrides, idx))

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
