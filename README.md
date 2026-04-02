# IS-ML-classifier-code

Codebase for the IS-ML project on stochastic beam propagation, convolutional
classification, and diffusion-based generative augmentation.

## Repository structure

```text
IS-ML-classifier-code/
├── classification_utils/
├── data/           # local datasets and generated samples, excluded from git
├── figures/        # exported figures for papers or reports
├── generation_utils/
├── run_classification.py          # main classification entry point
├── simulation_utils/
├── train_conditional_diffusion.py  # main generation entry point
├── run_simulation.py               # main simulation entry point
└── tests/          # lightweight regression tests
```

The repository is intentionally kept flat. The reusable modules are exposed
directly at the repository root through `generation_utils/` and
`simulation_utils/`, and the main executable scripts are placed alongside them.
At present the repository includes the core simulation module, the main
classification pipeline, the first public generation pipeline, and lightweight
regression tests.

## Current simulation module

- `beam_creation.py`
- `beam_propagation.py`
- `simulation.py`

These components have been consolidated into `simulation_utils/`, with
lightweight tests in `tests/test_simulation.py`.

## Current generation module

- `generation_utils/datasets.py`
- `generation_utils/losses.py`
- `generation_utils/diffusion.py`
- `generation_utils/pipeline.py`
- `train_conditional_diffusion.py`

The generation pipeline currently covers the conditional diffusion training
step and the export of class-conditional synthetic samples used for data
augmentation.

## Current classification module

- `classification_utils/models.py`
- `classification_utils/datasets.py`
- `classification_utils/training.py`
- `classification_utils/experiment.py`
- `run_classification.py`

The classification pipeline covers the main intensity-versus-ACF preprocessing,
the controlled crop-and-shift protocol, optional generated-data augmentation,
and the SimpleCNN / ResNet18 training loop used in the paper.

## Usage

### 1. Run one default simulation

The script [run_simulation.py](/home/wangaok23/PDEExperiment/IS-ML_code/IS-ML-classifier-code/run_simulation.py) runs one beam-propagation simulation for the default paper setting:
- code `1101`
- `L_x = 64`
- `N_x = 2048`
- `z = 5`
- `\Delta z = 1/32`
- `\sigma = 5\times 10^{-5}`
- `l_0 = 1.5`

Run:

```bash
python run_simulation.py
```

Default outputs are written to:
- `data/processed/default_simulation_1101/initial_intensity.npy`
- `data/processed/default_simulation_1101/final_intensity.npy`
- `data/processed/default_simulation_1101/summary.json`

If you want a different propagation setting, edit the `CONFIG` block at the top of the script.

### 2. Run the default classification experiment

The script [run_classification.py](/home/wangaok23/PDEExperiment/IS-ML_code/IS-ML-classifier-code/run_classification.py) runs the main baseline classification setting used in the paper:
- dataset `dataset_z-5.00_sigma-5e-05_l0-1.5_Nx-2048_medium`
- centered test protocol
- crop size `64\times 64`
- zero padding
- no generated-data augmentation
- `SimpleCNN` and `ResNet18`

Run:

```bash
python run_classification.py
```

Default outputs are written under `data/processed/`, in a directory named by the experiment configuration, and include:
- `summary.csv`
- `history_SimpleCNN.json`
- `history_ResNet18.json`
- model checkpoints in `checkpoints/`

If you want to switch to ACF inputs, generated-data augmentation, or a different crop-and-shift setting, edit the `CONFIG` block at the top of the script.

### 3. Run conditional diffusion training and sample generation

The script [train_conditional_diffusion.py](/home/wangaok23/PDEExperiment/IS-ML_code/IS-ML-classifier-code/train_conditional_diffusion.py) runs the default conditional diffusion pipeline used for generative augmentation:
- dataset `dataset_z-5.00_sigma-5e-05_l0-1.5_Nx-2048_medium`
- `256\times 256` inputs
- `\mathbf{v}` prediction
- sample-space loss
- frequency-loss weight `\lambda = 1`
- `50` generated samples per class

Run:

```bash
python train_conditional_diffusion.py
```

Default outputs are written under `results/pipeline_output_.../`, including:
- the best pretrained checkpoint
- training summaries
- class-conditional generated samples in `class-*/stage5_pretrained_data/`

This script requires PyTorch, `diffusers`, `accelerate`, and `ema-pytorch`. To change the generative setting, edit the `CONFIG` block at the top of the script.
