# RUBIS examples

Run the examples from the repository root after installing RUBIS:

```bash
python -m pip install -e .
```

## Basic deformation

Deform a continuous, uniformly rotating polytrope:

```bash
python examples/basic_deformation.py
```

## Composite deformation

Deform a model containing a density discontinuity and inspect the material and vacuum outputs:

```bash
python examples/composite_deformation.py
```

## Post-processing

Compute diagnostics and radiative flux, then write the converged model in the historical RUBIS format:

```bash
python examples/post_processing.py
```