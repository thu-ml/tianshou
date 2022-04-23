Benchmark
=========


Mujoco Benchmark
----------------

Tianshou's Mujoco benchmark contains state-of-the-art results.

Every experiment is conducted under 10 random seeds for 1-10M steps. Please refer to https://github.com/thu-ml/tianshou/tree/master/examples/mujoco for source code and detailed results.

.. raw:: html

    <center>
        <select id="env-mujoco" onchange="showEnv(this)"></select>
        <br>
        <div id="vis-mujoco"></div>
        <br>
    </center>

The table below compares the performance of Tianshou against published results on OpenAI Gym MuJoCo benchmarks. We use max average return in 1M timesteps as the reward metric. ~ means the result is approximated from the plots because quantitative results are not provided. - means results are not provided. The best-performing baseline on each task is highlighted in boldface. Referenced baselines include `TD3 paper <https://arxiv.org/pdf/1802.09477.pdf>`_, `SAC paper <https://arxiv.org/pdf/1812.05905.pdf>`_, `PPO paper <https://arxiv.org/pdf/1707.06347.pdf>`_, `ACKTR paper <https://arxiv.org/abs/1708.05144>`_, `OpenAI Baselines <https://github.com/openai/baselines>`_ and `Spinning Up <https://spinningup.openai.com/en/latest/spinningup/bench.html>`_.

.. image:: /_static/images/mujoco_comparison.svg

Runtime averaged on 8 benchmarked MuJoCo tasks is listed below. All results are obtained using a single Nvidia TITAN X GPU and
up to 48 CPU cores (at most one CPU core for each thread). 

.. image:: /_static/images/mujoco_time.svg


Atari Benchmark
---------------

Please refer to https://github.com/thu-ml/tianshou/tree/master/examples/atari
