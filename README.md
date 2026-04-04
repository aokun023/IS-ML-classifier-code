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

## Environment and installation

The repository was organized for Python-based numerical experiments. A clean
virtual environment or conda environment is recommended.

Recommended Python version:
- Python `3.10` or `3.11`

An environment file is provided at the repository root:

```bash
conda env create -f environment.yml
conda activate isml
```

Core dependencies by module:
- simulation: `numpy`, `scipy`
- classification: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`, `tqdm`
- generation: `numpy`, `pandas`, `torch`, `diffusers`, `accelerate`, `ema-pytorch`, `scikit-learn`, `tqdm`

One possible setup is:

```bash
conda create -n isml python=3.10
conda activate isml
pip install numpy scipy pandas matplotlib scikit-learn tqdm
pip install torch
pip install diffusers accelerate ema-pytorch
```

If you only want to run the simulation code, `torch`, `diffusers`,
`accelerate`, and `ema-pytorch` are not needed.

After installation, you can run the lightweight regression tests from the
repository root:

```bash
python -m unittest tests/test_simulation.py tests/test_classification.py
```

The classification tests are skipped automatically when `torch` is not
available.

## Data layout

The repository uses the following directory convention under `data/`:

- `data/raw/` stores input datasets, for example
  `dataset_z-5.00_sigma-5e-05/`
- `data/processed/` stores outputs created by the public entry scripts,
  including simulation summaries, classification results, checkpoints, and
  confusion matrices

Under the default configuration:
- `run_simulation.py` writes one generated dataset to
  `data/raw/dataset_z-5.00_sigma-5e-05/`
- `run_classification.py` reads from `data/raw/` and writes experiment outputs
  under `data/processed/classification_.../`
- `train_conditional_diffusion.py` reads from `data/raw/` and writes generative
  outputs under `results/generation_.../`

Large datasets, generated samples, and checkpoints are excluded from git by
default.

## Current simulation module

- `beam_creation.py`
- `beam_propagation.py`
- `simulation_class.py`

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

The script `run_simulation.py` generates one default propagated-intensity dataset for the paper setting:
- 15 nonzero binary classes built from the default mode dictionary
- `150` samples per class
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
- `data/raw/dataset_z-5.00_sigma-5e-05/inputs_2048/*.npy`
- `data/raw/dataset_z-5.00_sigma-5e-05/inputs_256/*.npy`
- `data/raw/dataset_z-5.00_sigma-5e-05/metadata_2048.csv`
- `data/raw/dataset_z-5.00_sigma-5e-05/metadata_256.csv`
- `data/raw/dataset_z-5.00_sigma-5e-05/summary.json`

This dataset can then be used directly by both `train_conditional_diffusion.py`
and `run_classification.py`. If you want a different propagation setting or a
different dataset size, edit the `CONFIG` block at the top of the script.

### 2. Run the default classification experiment

The script `run_classification.py` runs the main baseline classification setting used in the paper:
- default dataset root given by `CONFIG["real_data_root"]`
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

If you want to switch to a different dataset, ACF inputs, generated-data
augmentation, or a different crop-and-shift setting, edit the `CONFIG` block at
the top of the script. The saved classification directory keeps a short naming
scheme and records only the dataset parameters `z` and `sigma` from the dataset
name, together with the main preprocessing and shift settings.
Both `real_data_root` and `generated_data_root` are interpreted relative to the
repository root by default, for example `data/raw/<dataset_name>` or
`results/<generation_run_name>`.

### 3. Run conditional diffusion training and sample generation

The script `train_conditional_diffusion.py` runs the default conditional diffusion pipeline used for generative augmentation:
- default dataset root given by `CONFIG["dataset_root"]`
- `256\times 256` inputs
- `\mathbf{v}` prediction
- sample-space loss
- frequency-loss weight `\lambda = 1`
- `50` generated samples per class

Run:

```bash
python train_conditional_diffusion.py
```

Default outputs are written under `results/generation_.../`, including:
- the best pretrained checkpoint
- training summaries
- class-conditional generated samples in `class-*/stage5_pretrained_data/`

This script requires PyTorch, `diffusers`, `accelerate`, and `ema-pytorch`. To change the generative setting, edit the `CONFIG` block at the top of the script.
