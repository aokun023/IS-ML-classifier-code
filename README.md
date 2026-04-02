# IS-ML-classifier-code

Codebase for the IS-ML project on stochastic beam propagation, convolutional
classification, and diffusion-based generative augmentation.

## Repository structure

```text
IS-ML-classifier-code/
├── data/           # local datasets and generated samples, excluded from git
├── figures/        # exported figures for papers or reports
├── src/
│   └── simulation/
└── tests/          # lightweight regression tests
```

The current repository is intentionally kept flat. Only the simulation module
and its regression tests are included at this stage. Classification,
generation, and paper-specific scripts can be added later once their public
interfaces are cleaned and stabilized.

## Current simulation module

- `beam_creation.py`
- `beam_propagation.py`
- `simulation.py`

These components have been consolidated into `src/simulation/`, with
lightweight tests in `tests/test_simulation.py`.
