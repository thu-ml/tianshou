Logging Experiments
===================

Tianshou comes with multiple experiment tracking and logging solutions to manage and reproduce your experiments.
The dashboard loggers currently available are:

* :class:`~tianshou.utils.TensorboardLogger`
* :class:`~tianshou.utils.LazyLogger`


TensorboardLogger
-----------------

Tensorboard tracks your experiment metrics in a local dashboard. Here is how you can use TensorboardLogger in your experiment:

::

    from torch.utils.tensorboard import SummaryWriter
    from tianshou.utils import TensorboardLogger

    log_path = os.path.join(args.logdir, args.task, "dqn")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)
    result = trainer(..., logger=logger)

LazyLogger
----------

This is a place-holder logger that does nothing.




Weights and Biases Integration
------------------------------

:class:`~tianshou.utils.wandb_init` can be used to visualize your experiments in a hosted `W&B dashboard <https://wandb.ai/home>`_. It can be installed via ``pip install wandb``. You can also save your checkpoints in the cloud and restore your runs from those checkpoints. Here is how you can enable experiment tracking:

::

    from tianshou.utils import wandb_init

    wandb_run = wandb_init(args, run_name=None, resume_id=None)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer, wandb_run=wandb_run)
    result = trainer(..., logger=logger)

For logging checkpoints on any device, you need to define a ``save_checkpoint_fn`` which saves the experiment checkpoint and returns the path of the saved checkpoint:

::

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = ...
        # save model
        return ckpt_path

Then, use this function with ``TensorboardLogger(writer, wandb_run=wandb_run)`` to automatically version your experiment checkpoints after every ``save_interval`` step.

For resuming runs from checkpoint artifacts on any device, pass the W&B ``run_id`` of the run that you want to continue in ``wandb_init(..., resume_id=run_id)``. It will then download the latest version of the checkpoint and resume your runs from the checkpoint.

The example script is under `atari_dqn.py <https://github.com/thu-ml/tianshou/blob/master/examples/atari/atari_dqn.py>`_.
