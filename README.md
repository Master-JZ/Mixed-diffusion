## Introduction
The code is implemented based on DDPM (https://github.com/lucidrains/denoising-diffusion-pytorch) and DiffuSeq (https://github.com/Shark-NLP/DiffuSeq). Thanks for their work!

## Environment Dependencies
- Python 3.8
- PyTorch 2.0.0

You can refer to `requirements.txt` for the experimental environment we set to use.

## Run Mixed Noise
Simply use:

`CUDA_VISIBLE_DEVICES=0 python main.py`

For the ml-1m dataset, the general hyperparameters can be found in configs/cadirec_config_ml.py. For other datasets, please refer to configs/cadirec_config.py for the hyperparameters.

For hyperparameter tuning, please adjust the following parameters (alpha, beta, rho (mlm_probability_train and mlm_probability)) within the range of 0.1 to 0.2 for different datasets.

## Grid Search Helper
You can launch a grid search across any configurable arguments with `grid_search.py`. The script forwards all unrecognized CLI options to the underlying config parser (for example, `configs.cadirec_config` by default), so you can still pick a dataset or other defaults while enumerating hyperparameters.

Example (dry run):
```bash
python grid_search.py \
  --dry-run \
  --search-space '{"alpha": [0.1, 0.2], "beta": [0.1, 0.2]}' \
  --config-module configs.cadirec_config \
  --max-runs 2
```

Remove `--dry-run` to actually train each combination; logs and checkpoints will be written to per-run files under `saved_models/`.

- Log/checkpoint names now include a slug of the overridden hyperparameters (e.g., `model_idx` becomes `0-gs0-alpha0p1_beta0p2`), so you can distinguish runs. You can still override `--model_idx` manually if you prefer a custom name.
- When launching multiple grid-search jobs on the same GPU, set a distinct `--model_idx` prefix or `--output_dir` for each job to avoid file-name collisions; the script does not serialize access to the device, so ensure memory is sufficient.

### Defining the search space in a Python file
Instead of passing JSON on the CLI, you can point `--search-space` to a `.py` file that defines either:
- a `SEARCH_SPACE` dictionary, or
- a `get_search_space()` function returning that dictionary.

The dictionary should map argument names to lists of candidate values. Example file (see `configs/search_space_example.py`):
```python
SEARCH_SPACE = {
  "alpha": [0.1, 0.2],
  "beta": [0.1, 0.2],
}
```

Example usage:
```bash
python grid_search.py --search-space configs/search_space_example.py --dataset Beauty --dry-run
```
