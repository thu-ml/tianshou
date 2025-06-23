# Changelog

## Release 1.2.0

### Changes/Improvements

- `trainer`:
    - Custom scoring now supported for selecting the best model. #1202
- `highlevel`:
    - `DiscreteSACExperimentBuilder`: Expose method `with_actor_factory_default` #1248 #1250
    - `ActorFactoryDefault`: Fix parameters for hidden sizes and activation not being 
      passed on in the discrete case (affects `with_actor_factory_default` method of experiment builders)
    - `ExperimentConfig`: Do not inherit from other classes, as this breaks automatic handling by
      `jsonargparse` when the class is used to define interfaces (as in high-level API examples)
    - `AutoAlphaFactoryDefault`: Differentiate discrete and continuous action spaces
      and allow coefficient to be modified, adding an informative docstring
      (previous implementation was reasonable only for continuous action spaces)
        - Adjust usage in `atari_sac_hl` example accordingly.
    - `NPGAgentFactory`, `TRPOAgentFactory`: Fix optimizer instantiation including the actor parameters
      (which was misleadingly suggested in the docstring in the respective policy classes; docstrings were fixed),
      as the actor parameters are intended to be handled via natural gradients internally
- `data`:
    - `ReplayBuffer`: Fix collection of empty episodes being disallowed 
    - Collection was slow due to `isinstance` checks on Protocols and due to Buffer integrity validation. This was solved
      by no longer performing `isinstance` on Protocols and by making the integrity validation disabled by default.
- Tests:
    - We have introduced extensive **determinism tests** which allow to validate whether
      training processes deterministically compute the same results across different development branches.
      This is an important step towards ensuring reproducibility and consistency, which will be 
      instrumental in supporting Tianshou developers in their work, especially in the context of
      algorithm development and evaluation. 
  
### Breaking Changes

- `trainer`:
    - `BaseTrainer.run` and `__iter__`: Resetting was never optional prior to running the trainer,
      yet the recently introduced parameter `reset_prior_to_run` of `run` suggested that it _was_ optional.
      Yet the parameter was ultimately not respected, because `__iter__` would always call `reset(reset_collectors=True, reset_buffer=False)`
      regardless. The parameter was removed; instead, the parameters of `run` now mirror the parameters of `reset`,
      and the implicit `reset` call in `__iter__` was removed.     
      This aligns with upcoming changes in Tianshou v2.0.0.  
        * NOTE: If you have been using a trainer without calling `run` but by directly iterating over it, you
          will need to call `reset` on the trainer explicitly before iterating over the trainer.
        * Using a trainer as an iterator is considered deprecated and support for this will be removed in Tianshou v2.0.0.
- `data`:
    - `InfoStats` has a new non-optional field `best_score` which is used
      for selecting the best model. #1202
- `highlevel`:
    - Change the way in which seeding is handled: The mechanism introduced in v1.1.0 
      was completely revised:
        - The `train_seed` and `test_seed` attributes were removed from `SamplingConfig`.
          Instead, the seeds are derived from the seed defined in `ExperimentConfig`.
        - Seed attributes of `EnvFactory` classes were removed. 
          Instead, seeds are passed to methods of `EnvFactory`.

## Release 1.1.0

**NOTE**: This release introduced (potentially severe) performance regressions in data collection, please switch to a newer release for better performance.

### Highlights

#### Evaluation Package

This release introduces a new package `evaluation` that integrates best
practices for running experiments (seeding test and train environmets) and for
evaluating them using the [rliable](https://github.com/google-research/rliable)
library. This should be especially useful for algorithm developers for comparing
performances and creating meaningful visualizations. **This functionality is
currently in alpha state** and will be further improved in the next releases.
You will need to install tianshou with the extra `eval` to use it.

The creation of multiple experiments with varying random seeds has been greatly
facilitated. Moreover, the `ExpLauncher` interface has been introduced and
implemented with several backends to support the execution of multiple
experiments in parallel.

An example for this using the high-level interfaces can be found
[here](examples/mujoco/mujoco_ppo_hl_multi.py), examples that use low-level
interfaces will follow soon.

#### Improvements in Batch

Apart from that, several important
extensions have been added to internal data structures, most notably to `Batch`.
Batches now implement `__eq__` and can be meaningfully compared. Applying
operations in a nested fashion has been significantly simplified, and checking
for NaNs and dropping them is now possible.

One more notable change is that torch `Distribution` objects are now sliced when
slicing a batch. Previously, when a Batch with say 10 actions and a dist
corresponding to them was sliced to `[:3]`, the `dist` in the result would still
correspond to all 10 actions. Now, the dist is also "sliced" to be the
distribution of the first 3 actions.

A detailed list of changes can be found below.

### Changes/Improvements

- `evaluation`: New package for repeating the same experiment with multiple
  seeds and aggregating the results. #1074 #1141 #1183
- `data`:
    - `Batch`:
        - Add methods `to_dict` and `to_list_of_dicts`. #1063 #1098
        - Add methods `to_numpy_` and `to_torch_`. #1098, #1117
        - Add `__eq__` (semantic equality check). #1098
        - `keys()` deprecated in favor of `get_keys()` (needed to make iteration
          consistent with naming) #1105.
        - Major: new methods for applying functions to values, to check for NaNs
          and drop them, and to set values. #1181
        - Slicing a batch with a torch distribution now also slices the
          distribution. #1181
    - `data.collector`:
        - `Collector`:
            - Introduced `BaseCollector` as a base class for all collectors.
              #1123
            - Add method `close` #1063
            - Method `reset` is now more granular (new flags controlling
              behavior). #1063
        - `CollectStats`: Add convenience
          constructor `with_autogenerated_stats`. #1063
- `trainer`:
    - Trainers can now control whether collectors should be reset prior to
      training. #1063
- `policy`:
    - introduced attribute `in_training_step` that is controlled by the trainer.
      #1123
    - policy automatically set to `eval` mode when collecting and to `train`
      mode when updating. #1123
    - Extended interface of `compute_action` to also support array-like inputs
      #1169
- `highlevel`:
    - `SamplingConfig`:
        - Add support for `batch_size=None`. #1077
        - Add `training_seed` for explicit seeding of training and test
          environments, the `test_seed` is inferred from `training_seed`. #1074
    - `experiment`:
        - `Experiment` now has a `name` attribute, which can be set
          using `ExperimentBuilder.with_name` and
          which determines the default run name and therefore the persistence
          subdirectory.
          It can still be overridden in `Experiment.run()`, the new parameter
          name being `run_name` rather than
          `experiment_name` (although the latter will still be interpreted
          correctly). #1074 #1131
        - Add class `ExperimentCollection` for the convenient execution of
          multiple experiment runs #1131
        - The `World` object, containing all low-level objects needed for
          experimentation,
          can now be extracted from an `Experiment` instance. This enables
          customizing
          the experiment prior to its execution, bridging the low and high-level
          interfaces. #1187
        - `ExperimentBuilder`:
            - Add method `build_seeded_collection` for the sound creation of
              multiple
              experiments with varying random seeds #1131
            - Add method `copy` to facilitate the creation of multiple
              experiments from a single builder #1131
    - `env`:
        - Added new `VectorEnvType` called `SUBPROC_SHARED_MEM_AUTO` and used in
          for Atari and Mujoco venv creation. #1141
- `utils`:
    - `logger`:
        - Loggers can now restore the logged data into python by using the
          new `restore_logged_data` method. #1074
        - Wandb logger extended #1183
    - `net.continuous.Critic`:
        - Add flag `apply_preprocess_net_to_obs_only` to allow the
          preprocessing network to be applied to the observations only (without
          the actions concatenated), which is essential for the case where we
          want
          to reuse the actor's preprocessing network #1128
    - `torch_utils` (new module)
        - Added context managers `torch_train_mode`
          and `policy_within_training_step` #1123
    - `print`
        - `DataclassPPrintMixin` now supports outputting a string, not just
          printing the pretty repr. #1141

### Fixes

- `highlevel`:
    - `CriticFactoryReuseActor`: Enable the Critic
      flag `apply_preprocess_net_to_obs_only` for continuous critics,
      fixing the case where we want to reuse an actor's preprocessing network
      for the critic (affects usages
      of the experiment builder method `with_critic_factory_use_actor` with
      continuous environments) #1128
    - Policy parameter `action_scaling` value `"default"` was not correctly
      transformed to a Boolean value for
      algorithms SAC, DDPG, TD3 and REDQ. The value `"default"` being truthy
      caused action scaling to be enabled
      even for discrete action spaces. #1191
- `atari_network.DQN`:
    - Fix constructor input validation #1128
    - Fix `output_dim` not being set if `features_only`=True
      and `output_dim_added_layer` is not None #1128
- `PPOPolicy`:
    - Fix `max_batchsize` not being used in `logp_old` computation
      inside `process_fn` #1168
- Fix `Batch.__eq__` to allow comparing Batches with scalar array values #1185

### Internal Improvements

- `Collector`s rely less on state, the few stateful things are stored explicitly
  instead of through a `.data` attribute. #1063
- Introduced a first iteration of a naming convention for vars in `Collector`s.
  #1063
- Generally improved readability of Collector code and associated tests (still
  quite some way to go). #1063
- Improved typing for `exploration_noise` and within Collector. #1063
- Better variable names related to model outputs (logits, dist input etc.).
  #1032
- Improved typing for actors and critics, using Tianshou classes
  like `Actor`, `ActorProb`, etc.,
  instead of just `nn.Module`. #1032
- Added interfaces for most `Actor` and `Critic` classes to enforce the presence
  of `forward` methods. #1032
- Simplified `PGPolicy` forward by unifying the `dist_fn` interface (see
  associated breaking change). #1032
- Use `.mode` of distribution instead of relying on knowledge of the
  distribution type. #1032
- Exception no longer raised on `len` of empty `Batch`. #1084
- tests and examples are covered by `mypy`. #1077
- `NetBase` is more used, stricter typing by making it generic. #1077
- Use explicit multiprocessing context for creating `Pipe` in `subproc.py`.
  #1102

### Breaking Changes

- `data`:
    - `Collector`:
        - Removed `.data` attribute. #1063
        - Collectors no longer reset the environment on initialization.
          Instead, the user might have to call `reset` expicitly or
          pass `reset_before_collect=True` . #1063
        - Removed `no_grad` argument from `collect` method (was unused in
          tianshou). #1123
    - `Batch`:
        - Fixed `iter(Batch(...)` which now behaves the same way
          as `Batch(...).__iter__()`.
          Can be considered a bugfix. #1063
        - The methods `to_numpy` and `to_torch` in are not in-place anymore
          (use `to_numpy_` or `to_torch_` instead). #1098, #1117
        - The method `Batch.is_empty` has been removed. Instead, the user can
          simply check for emptiness of Batch by using `len` on dicts. #1144
        - Stricter `cat_`, only concatenation of batches with the same structure
          is allowed. #1181
        - `to_torch` and `to_numpy` are no longer static methods.
          So `Batch.to_numpy(batch)` should be replaced by `batch.to_numpy()`.
          #1200
- `utils`:
    - `logger`:
        - `BaseLogger.prepare_dict_for_logging` is now abstract. #1074
        - Removed deprecated and unused `BasicLogger` (only affects users who
          subclassed it). #1074
    - `utils.net`:
        - `Recurrent` now receives and returns
          a `RecurrentStateBatch` instead of a dict. #1077
    - Modules with code that was copied from sensAI have been replaced by
      imports from new dependency sensAI-utils:
        - `tianshou.utils.logging` is replaced with `sensai.util.logging`
        - `tianshou.utils.string` is replaced with `sensai.util.string`
        - `tianshou.utils.pickle` is replaced with `sensai.util.pickle`
- `env`:
    - All VectorEnvs now return a numpy array of info-dicts on reset instead of
      a list. #1063
- `policy`:
    - Changed interface of `dist_fn` in `PGPolicy` and all subclasses to take a
      single argument in both
      continuous and discrete cases. #1032
- `AtariEnvFactory` constructor (in examples, so not really breaking) now
  requires explicit train and test seeds. #1074
- `EnvFactoryRegistered` now requires an explicit `test_seed` in the
  constructor. #1074
- `highlevel`:
    - `params`: The parameter `dist_fn` has been removed from the parameter
      objects (`PGParams`, `A2CParams`, `PPOParams`, `NPGParams`, `TRPOParams`).
      The correct distribution is now determined automatically based on the
      actor factory being used, avoiding the possibility of
      misspecification. Persisted configurations/policies continue to work as
      expected, but code must not specify the `dist_fn` parameter.
      #1194 #1195
    - `env`:
        - `EnvFactoryRegistered`: parameter `seed` has been replaced by the pair
          of parameters `train_seed` and `test_seed`
          Persisted instances will continue to work correctly.
          Subclasses such as `AtariEnvFactory` are also affected requires
          explicit train and test seeds. #1074
        - `VectorEnvType`: `SUBPROC_SHARED_MEM` has been replaced
          by `SUBPROC_SHARED_MEM_DEFAULT`. It is recommended to
          use `SUBPROC_SHARED_MEM_AUTO` instead. However, persisted configs will
          continue working. #1141

### Tests

- Fixed env seeding it `test_sac_with_il.py` so that the test doesn't fail
  randomly. #1081
- Improved CI triggers and added telemetry (if requested by user) #1177
- Improved environment used in tests.
- Improved tests bach equality to check with scalar values #1185

### Dependencies

- [DeepDiff](https://github.com/seperman/deepdiff) added to help with diffs of
  batches in tests. #1098
- Bumped black, idna, pillow
- New extra "eval"
- Bumped numba to >=60.0.0, permitting installation on python 3.12 # 1177
- New dependency sensai-utils

Started after v1.0.0
