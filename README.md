# IS-ML-classifier-code

Codebase for the IS-ML project on stochastic beam propagation, convolutional
classification, and diffusion-based generative augmentation.

## Repository structure

```text
IS-ML-classifier-code/
├── configs/        # explicit experiment configuration files
├── data/           # local data layout (kept out of git except small metadata)
├── docs/           # short technical notes and reproduction instructions
├── figures/        # exported figures intended for papers or reports
├── notebooks/      # exploratory notebooks kept separate from source code
├── results/        # derived outputs: summaries, tables, logs, plots
├── scripts/        # runnable entry-point scripts
│   ├── analysis/
│   ├── classification/
│   ├── figures/
│   ├── generation/
│   └── simulation/
├── src/            # importable project code
│   └── isml/
│       ├── analysis/
│       ├── classification/
│       ├── generation/
│       ├── simulation/
│       └── utils/
└── tests/          # lightweight regression tests
```

## Recommended migration order

1. Move reusable numerical kernels into `src/isml/simulation/`.
2. Move CNN and diffusion utilities into `src/isml/classification/` and
   `src/isml/generation/`.
3. Keep paper-specific figure builders in `scripts/figures/`.
4. Keep one-off exploratory notebooks in `notebooks/`, not in `src/`.
5. Store large datasets, checkpoints, and generated samples under `data/` or
   `results/`, and exclude them from git.

## First files to migrate

- `beam_creation.py`
- `beam_propagation.py`
- `SimulationClass.py`
- `models.py`
- `TrainconditionalModelDM.py`
- `ACF_Shift_Analysis.py`

These can later be refactored into importable modules while preserving the
current scripts as thin entry points under `scripts/`.
