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

✅ python grid_search.py --dry-run --search-space '{"alpha":[0.1],"beta":[0.2]}' --max-runs 1
