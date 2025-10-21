# Pyomo Devel

The `pyomo.devel` directory contains **experimental, research-oriented, or
rapidly evolving** extensions to Pyomo. These packages are **not guaranteed to
be stable** and may change or be removed between releases **without
deprecation warnings**.

## Purpose

`pyomo.devel` is intended for:

- Experimental algorithms and modeling approaches.
- Research prototypes under active development.
- Functionality that is **not yet ready for long-term API guarantees**.
- Work that may later graduate into `pyomo.addons` after sufficient
  maturity, including testing and documentation.

Examples include:
- `doe`
- `piecewise`
- `solver`
- `sensitivity_toolbox`

## Expectations

Packages placed in `pyomo.devel` are **encouraged to be exploratory** but must
still meet basic project standards to ensure they do not break the broader
Pyomo ecosystem.

### Stability
- APIs in this directory are **not stable** and **may change or be removed**
  without notice.
- Users should not rely on `pyomo.devel` components for production workflows.

### Testing
- All `devel` packages must include **basic tests** verifying import and
  execution of key components.
- Tests should run under Pyomo's CI but may be skipped in qualifying scenarios.

### Documentation
- At minimum, each package should contain:
  - A **README.md** describing the purpose and experimental status.

### Dependencies
- Packages in `devel` **may not introduce required dependencies** for Pyomo core.
- Optional dependencies must be:
  - Declared in `setup.py` under the appropriate category.
  - Publicly available on PyPI and/or Anaconda and reasonably maintained.
