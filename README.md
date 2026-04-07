# jaxsgp4

[![CI](https://github.com/cmpriestley/jaxsgp4/actions/workflows/ci.yml/badge.svg)](https://github.com/cmpriestley/jaxsgp4/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2603.27830-b31b1b)](https://arxiv.org/abs/2603.27830)

A pure JAX implementation of the SGP4 (Simplified General Perturbations 4) satellite orbit propagation algorithm.

![Starlink satellite trajectories](docs/starlink_trajectories_two_tone.png)

## Overview

`jaxsgp4` provides a differentiable and JIT-compilable SGP4 propagator built entirely in JAX. The SGP4 algorithm is the standard method for propagating Two-Line Element (TLE) sets to predict satellite positions and velocities. For full details, see the accompanying paper: [*jaxsgp4: GPU-accelerated mega-constellation propagation with batch parallelism*](https://arxiv.org/abs/2603.27830).

Because the implementation uses only JAX primitives, it is fully compatible with JAX's transformation system:

- **`jax.jit`** — compile the propagator for fast repeated evaluation
- **`jax.vmap`** — vectorize over satellites, time steps, or both
- **`jax.grad`** / **`jax.jacobian`** — compute exact derivatives for optimisation and orbit determination

## Installation

```bash
pip install jaxsgp4
```

Or install from source:

```bash
git clone https://github.com/cmpriestley/jaxsgp4.git
cd jaxsgp4
pip install -e ".[dev]"
```

## Quick Start

```python
from jaxsgp4 import tle2sat, sgp4

# Parse a TLE
tle_line1 = "1 44714U 19074B   26013.33334491  .00010762  00000+0  67042-3 0  9990"
tle_line2 = "2 44714  53.0657  75.1067 0002699  79.3766  82.4805 15.10066292  5798"

sat = tle2sat(tle_line1, tle_line2)

# Propagate 60 minutes from epoch
rv, error_code = sgp4(sat, 60.0)

r = rv[:3]   # Position in km (TEME frame)
v = rv[3:]   # Velocity in km/s (TEME frame)
```

## API Reference

### Data Model

**`Satellite`** — A `NamedTuple` holding orbital elements parsed from a TLE:

| Field | Description |
|-------|-------------|
| `n0` | Mean motion (revolutions/day) |
| `e0` | Eccentricity |
| `i0` | Inclination (degrees) |
| `w0` | Argument of perigee (degrees) |
| `Omega0` | Right ascension of ascending node (degrees) |
| `M0` | Mean anomaly (degrees) |
| `Bstar` | Drag coefficient (Earth radii⁻¹) |
| `epochdays` | Epoch as day of year (fractional) |
| `epochyr` | Epoch year (4-digit) |

### Functions

| Function | Description |
|----------|-------------|
| `tle2sat(tle_1, tle_2)` | Parse a single TLE into a `Satellite` object |
| `tle2sat_array(tle_1_array, tle_2_array)` | Parse multiple TLEs into a vectorized `Satellite` |
| `sgp4(sat, tsince)` | Propagate `tsince` minutes from epoch. Returns `(rv, error_code)` where `rv` is a length-6 array `[x, y, z, vx, vy, vz]` in km and km/s |
| `sgp4_jdfr(sat, jd, fr)` | Propagate to a Julian Date (`jd` + `fr`). Returns `(rv, error_code)` |

### Error Codes

| Code | Meaning |
|------|---------|
| 0 | No error |
| 1 | Mean eccentricity out of range |
| 2 | Mean motion ≤ 0 |
| 4 | Semi-latus rectum < 0 |
| 6 | Satellite radius below Earth's surface |

## Examples

See the [examples guide](docs/examples.md) for detailed usage, including:

- Vectorizing over multiple satellites and time steps with `jax.vmap`
- JIT compilation for performance
- Computing gradients with `jax.grad` and `jax.jacobian`
- Parsing bulk TLE catalogues

## Performance

`jaxsgp4` exploits GPU batch parallelism to dramatically outperform traditional C++ SGP4 implementations. On an NVIDIA A100, propagating the entire Starlink constellation (9,341 satellites × 1,000 time steps) completes in **3.8 ms** — a **1,500× speedup** over C++.

| Hardware | Peak speedup | Break-even batch size |
|----------|-------------:|----------------------:|
| NVIDIA T4 | ~250× | ~300 |
| NVIDIA A100 | ~1,500× | ~500 |

See the [paper](https://arxiv.org/abs/2603.27830) for full benchmark methodology and results.

## Precision

By default JAX uses 32-bit floating point. FP32 introduces ~1 m positional error at epoch, remaining under 1 km over two weeks of propagation — well within SGP4's inherent physical model error (~1 km/day). FP64 matches the C++ reference to ~1 μm.

> **Tip:** when using FP32, supply propagation time via `sgp4(sat, tsince)` (minutes since epoch) rather than Julian dates to avoid systematic epoch-representation errors.

## Limitations

- **Near-Earth orbits only** — orbital period must be < 225 minutes (SDP4 deep-space extensions are under development)
- **WGS-72 constants only** — matches the standard SGP4 specification

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT

## Citation

If you use `jaxsgp4` in your research, please cite:

```bibtex
@article{priestley2026jaxsgp4,
  title={jaxsgp4: GPU-accelerated mega-constellation propagation with batch parallelism},
  author={Priestley, Charlotte and Handley, Will},
  year={2026},
  eprint={2603.27830},
  archivePrefix={arXiv},
  primaryClass={cs.DC}
}
```

## References

This implementation was written directly from the equations laid out in:

- Hoots, F. R., Schumacher Jr., P. W., & Glover, R. A. (2004). *History of Analytical Orbit Modeling in the U.S. Space Surveillance System.* Journal of Guidance, Control, and Dynamics, 27(2), 174–185. [doi:10.2514/1.9161](https://doi.org/10.2514/1.9161)

originally taken from:

- Hoots, F. R., & Roehrich, R. L. (1980). *Spacetrack Report No. 3: Models for Propagation of NORAD Element Sets.* Aerospace Defense Command, United States Air Force. [(PDF)](https://celestrak.org/NORAD/documentation/spacetrk.pdf)

with reference to the [sgp4 Python library](https://github.com/brandon-rhodes/python-sgp4) by Brandon Rhodes for implementation specifics.
