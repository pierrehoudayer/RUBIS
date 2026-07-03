<p align="center">
  <img src="misc/rubis-logo.png" alt="RUBIS logo" width="420">
</p>

# RUBIS

RUBIS (*Rotation code Using Barotropy conservation over Isopotential
Surfaces*) computes the centrifugal deformation of barotropic stellar
and planetary models under conservative cylindrical rotation.

Starting from a spherically symmetric density profile, RUBIS preserves
the relation between density and pressure on isopotential surfaces and
solves Poisson's equation iteratively. Continuous models use a radial
solver, while models containing density discontinuities are handled in
spheroidal coordinates.

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/pierrehoudayer/RUBIS.git
cd RUBIS
python -m pip install -e .
```

Plotting utilities are optional:

```bash
python -m pip install -e ".[plot]"
```

For development:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

```python
from rubis.api import deform
from rubis.config import (
    DeformationConfig,
    PolytropeConfig,
    RotationConfig,
    SolverOptions,
)
from rubis.rotation_profiles import solid

config = DeformationConfig(
    model=PolytropeConfig(
        index=3.0,
        n_points=501,
    ),
    rotation=RotationConfig(
        profile=solid,
        target=0.9,
    ),
    solver=SolverOptions(
        max_degree=51,
        angular_resolution=51,
        verbose=True,
    ),
)

model, vacuum, info = deform(config)
```

`model` contains the converged material structure, `vacuum` contains the
exterior solution when the spheroidal solver is used, and `info`
summarises convergence.

Additional examples cover composite models, diagnostics, radiative flux
and plotting in [`examples/`](examples/).

## Included model

[`Models/Jupiter.txt`](Models/Jupiter.txt) is a discontinuous Jupiter
model used by the full-scale acceptance tests.

## Reference

The numerical method is presented in:

P. S. Houdayer and D. R. Reese,
“RUBIS: A simple tool for calculating the centrifugal deformation of
stars and planets”, *Astronomy & Astrophysics* **675**, A181 (2023).

DOI: https://doi.org/10.1051/0004-6361/202346403

## License

RUBIS is distributed under the GNU General Public License v3.0.
See [`LICENSE.md`](LICENSE.md).
