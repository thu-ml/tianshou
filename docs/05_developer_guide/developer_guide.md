# Developer Guide

The section addresses developers of Tianshou, providing information for 
both casual contributors and maintainers alike.


## Python Virtual Environment

Tianshou is built and managed by [poetry](https://python-poetry.org/). 

The development environment uses Python 3.11.

To install all relevant requirements (as well as Tianshou itself in editable mode)
you can simply call

    poetry install --with dev

```{important}
Depending on your setup, you may need to create and activate an empty virtual environment
using the right Python version beforehand. For instance, to do this with conda, use:

    conda create -n tianshou python=3.11
    conda activate tianshou
```


## Code Style and Auto-Formatting

When editing code in Tianshou, strive for **local consistency**, i.e.
adhere to the style already present in the codebase.

Tianshou uses an auto-formatting for consistency.
To apply it, call

    poe format

To check whether your formatting is compliant without applying the
auto-formatter, call

    poe lint


## Type Checking

We use [mypy](https://github.com/python/mypy/) to perform static type analysis. 
To check typing, run

    poe type-check


## Tests

### Running the Test Suite Locally

Tianshou uses pytest. Tests are located in `./test`.

To run the full set of tests locally, run

    poe test

### Determinism Tests

We implemented **determinism tests** for Tianshou's algorithms, which allow us to determine
whether algorithms still compute exactly the same results even after large refactorings.
These tests are applied by

  1. creating a behavior snapshot in the old code branch before the changes and then
  2. running the respective determinism test in the new branch to ensure that the behavior is the same.

Unfortunately, full determinism is difficult to achieve across different platforms and even different
machines using the same platform an Python environment.
Therefore, these tests are not carried out in the CI pipeline.
Instead, it is up to the developer to run them locally and check the results whenever a change
is made to the codebase that could affect algorithm behavior.

Technically, the two steps are handled by setting static flags in class `AlgorithmDeterminismTest` and then
running either the full test suite or a specific determinism test (`test_*_determinism`, e.g. `test_ddpg_determinism`)
in the two branches to be compared.

  1. On the old branch: (Temporarily) set `ENABLED=True` and `FORCE_SNAPSHOT_UPDATE=True` and run the test(s).
  2. On the new branch: (Temporarily) set `ENABLED=True` and `FORCE_SNAPSHOT_UPDATE=False` and run the test(s).
  3. Inspect the test results; find a summary in `determinism_tests.log`

### Tests in CI (GitHub Actions)

CI tests will extensively test Tianshou's functionality in multiple environments.

In particular, we test
  * on Ubuntu (full functionality tested)
    * **py_pinned**: using the pinned development environment (Python 3.11, known versions of all dependencies)
    * **py_latest**: using a more recent Python version with the newest set of compatible dependencies (automatically resolved)
  * on Windows and macOS (core functionality tested)


#### Principle of Maximum Compatibility

The idea behind testing with dynamically resolved dependencies is that we want to maximize the applicability 
of Tianshou: For important dependencies that could conflict with environments used by our users, **we do not restrict the version of a dependency unless there is a known incompatibility.**

If incompatibilities should arise (e.g. by the "py_latest" test failing), we either 
 * resolve them by making the code compatible with both old and new versions OR
 * add an upper bound to our dependency declarations (excluding the incompatible versions) and release a new 
   version of Tianshou to make these exclusions explicit.


## High-Level API

The high-level API provides a declarative, user-friendly interface for setting up reinforcement learning experiments. From a library developer's perspective, it is important that this API be clearly structured and maintainable. This section explains the architectural principles and how to extend the API to support new algorithms.

### Core Abstractions

The high-level API is built around a clear separation of concerns:

**Parameter Classes** are dataclasses (inheriting from `Params`) that represent algorithm-specific configuration. 
They capture hyperparameters in a high-level, user-friendly form. 
Because the high-level interface must abstract away from low-level details, parameters may need transformation before being passed to policy classes. 
This is handled via `ParamTransformer` instances, which successively transform the parameter dictionary representation. 
To maintain clarity and reduce coupling, parameter transformers are co-located with the parameters they affect. 
The system uses inheritance and mixins extensively to reduce duplication while maintaining flexibility.

**Factories** embody the principle of declarative configuration. 
Because object creation may depend on other objects that don't yet exist at configuration time (e.g., neural networks depend on environment properties), 
the API transitions from objects to factories. 
Key factory types include:
- `EnvFactory` for creating training, test, and watch environments
- `AgentFactory` as the central factory that creates policies, trainers, and collectors
- Various specialized factories for optimizers, actors, critics, noise, distributions, learning rate schedulers, and policy wrappers

**Algorithm Factories** (subclasses of `AlgorithmFactory`) are the core components responsible for orchestrating the creation of all algorithm-specific objects. 
They handle the creation of neural network architectures, apply parameter transformations, instantiate policies, and create trainers with appropriate collectors. 
To support a new algorithm, this is the primary extension point.

**Experiment Builders** (subclasses of `ExperimentBuilder`) provide the user-facing interface following the builder pattern. 
They contain sensible defaults while allowing customization through fluent `with_*` methods. 
Builder mixins provide composable functionality for common patterns (e.g., actor/critic configuration), avoiding code duplication across algorithm-specific implementations.

### Supporting a New Algorithm

Extending the high-level API to support a new algorithm involves creating three main components:

**Parameter Class**: Define a dataclass in `tianshou/highlevel/params/algorithm_params.py` that inherits from appropriate base classes and mixins. 
The choice of base class depends on the algorithm's architecture (actor-critic, single network, etc.) and learning paradigm (on-policy, off-policy). 
Override `_get_param_transformers()` to specify how high-level parameters should be transformed for the low-level policy API.
Common transformers handle optimizer creation, noise instantiation, and environment-dependent parameter resolution.

**Algorithm Factory**: Implement a subclass of `AlgorithmFactory` in `tianshou/highlevel/algorithm.py`. 
In most cases, inherit from existing base factories like `ActorCriticOnPolicyAlgorithmFactory`, `ActorCriticOffPolicyAlgorithmFactory`, 
or `DiscreteCriticOnlyOffPolicyAlgorithmFactory`, which handle common creation patterns. 
The primary requirement is implementing `_get_algorithm_class()` to return the appropriate algorithm class. 
For algorithms with non-standard requirements, override `_create_algorithm()`, `_create_kwargs()`, etc. to customize the instantiation logic.

**Experiment Builder**: Add a builder class in `tianshou/highlevel/experiment.py` that inherits from `OnPolicyExperimentBuilder` or `OffPolicyExperimentBuilder` 
along with appropriate mixins. The mixins provide standard functionality for configuring actors and critics 
(single critic, dual critics, critic ensembles, parameter sharing patterns, etc.). 
The main responsibility is implementing `_create_algorithm_factory()` to instantiate the algorithm factory with appropriate parameters and network factories. 
Optionally provide `with_*` methods for algorithm-specific configuration.

Export the new classes in `tianshou/highlevel/__init__.py` to make them available to users.

### Design Principles

The architecture follows several key principles:

**Separation of Concerns**: Configuration is cleanly separated from implementation. 
The transformation system bridges these layers while maintaining independence.

**Declarative Configuration**: Factories enable a declarative style where experiments are defined by what should be created rather than imperative steps. 
This makes experiments easily serializable and reproducible.

**Composition and Inheritance**: Mixins and inheritance reduce code duplication. 
Common functionality is factored into reusable components while maintaining flexibility for algorithm-specific requirements.

**Progressive Disclosure**: The API provides sensible defaults for simple use cases while allowing deep customization when needed. 
Users can progress from simple configurations to advanced setups without fighting the abstractions.

**Co-location**: Related code is kept together. Parameter transformers are defined near the parameters they transform, 
maintaining clarity about dependencies and making the codebase easier to navigate.

**Type Safety**: Extensive use of generics and type hints ensures that type checkers can catch configuration errors at development time rather than runtime.


## Documentation

Documentation is in the `docs/` directory, using Markdown (`.md`), ReStructuredText (`.rst`) and notebook files. 
`index.rst` is the main page. 

API References are automatically generated by [Sphinx](http://www.sphinx-doc.org/en/stable/) according to the outlines under `docs/api/` and should be modified when any code changes.

To compile documentation into webpage, run

    poe doc-build

The generated webpages can subsequently be found in `docs/_build` and can be viewed with any browser.

### Verifications

We have several automated verification methods for documentation:

1. pydocstyle (as part of ruff): tests all docstring under `tianshou/`;

2. doc8 (as part of ruff): tests ReStructuredText format;

3. sphinx spelling and test: test if there is any error/warning when generating front-end html documentation.


## Creating a Release

To release a new version on PyPI,

 * set the version to be released in `tianshou/__init__.py` and in `pyproject.toml`, creating a commit
 * tag the commit with the version (using the format `v1.2.3`)
 * push the commit (`git push`) and the tag (`git push --tags`)
 * create a new release on GitHub based on the tag; this will trigger the release job for PyPI.

In the past, we provided releases to conda-forge as well, but this is currently not maintained.
