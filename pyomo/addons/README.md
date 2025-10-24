# Pyomo Addons

The `pyomo.addons` directory contains **mostly stable extensions** to Pyomo
that are **not part of the core library*. These packages extend Pyomo's
modeling, solver, and analysis capabilities, and have demonstrated sufficient
maturity, documentation, and testing.

## Purpose

`pyomo.addons` houses packages that:

- Have **mostly stable APIs** suitable for downstream use.
- Are **sufficiently documented** and **tested**.
- Represent functionality that is **useful to the wider Pyomo community** but
  not required for core operation.

Examples include:
- `gdpopt`
- `incidence_analysis`
- `latex_printer`
- `trustregion`
- `viewer`

## Requirements

A package may be accepted into `pyomo.addons` once it meets all of the
following criteria.

### Stability and Maintenance
- The code must be **mostly stable** with minimal expected API changes.
- There must be at least one **designated maintainer** responsible for upkeep,
  compatibility, and user support.
- The package must not rely on deprecated or experimental Pyomo internals.

### Testing
- Test coverage should be **at least 80%**, or comparable to similar supported modules.
- Tests must run successfully within the Pyomo CI environment.
- Expensive, solver-specific, or optional tests should be properly marked
  (e.g., with `@pytest.mark.expensive`, `@pytest.mark.solver`).

### Documentation
- Each addon must include:
  - (PREFERABLE) A **reference page** in the Pyomo online documentation.
  - (MINIMUM) A **README.md** file in the addon directory summarizing:
    - Functionality
    - Example usage or link to docs

### Dependencies
- Addons **may not introduce required dependencies** for Pyomo core.
- Optional dependencies must be:
  - Declared in `setup.py` under the appropriate category.
  - Publicly available on PyPI and/or Anaconda and reasonably maintained.

### Backward Compatibility
- Addons are expected to maintain **backward-compatible APIs** between minor
  Pyomo releases.
- Breaking changes must follow the documented **deprecation process** and
  appear in release notes.
