Benchmarks
==========

Tianshou's algorithm implementations lead to state-of-the-art results on standard benchmarks.

An efficient parallel implementation for
evaluating algorithms on mujoco or atari is in `benchmark/run_benchmark.py`. It can easily be adapted for custom
 benchmarks as well. The reported results are thus completely reproducible.

The evaluation code uses Tianshou's integration with the `rliable <https://github.com/google-research/rliable>`_ framework,
which supports best practices for trustworthy RL evaluation.
Each experiment is conducted under 5 random seeds, we report the interquartile mean (IQM) and 95% confidence intervals over these seeds.

.. raw:: html

    <center>
        <select id="env-mujoco" onchange="showMujocoResults(this)"></select>
        <br>
        <div id="vis-mujoco"></div>
        <br>
    </center>
