Logging Experiments
===================

Tianshou comes with multiple experiment tracking and logging solutions to manage and reproduce your experiments.
The dashboard loggers currently available are:
* TensorboardLogger
* WandbLogger

TensorboardLogger
-----------------
Tensorboard tracks you experiment metrics in a local dashboard. Here's how you can use TensorboardLogger in your experiment.
::
        log_path = os.path.join(args.logdir, args.task, 'psrl')
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        logger = TensorboardLogger(writer)
        
        train(..., logger = logger)
        

WandbLogger
-----------
WandbLogger can be used to visualize your experiments in a hosted W&B dashboard. You can also save your 
checkpoints in the cloud and restrore your runs from those checkpoints. Here's how you can enable WandbLogger.

* Install `wandb` using `pip install wandb`
* Example usage
::

    logger = WandBLogger(...)
    result = trainer(...,logger=logger)

    :param int train_interval: the log interval in log_train_data(). Default to 1000.
    :param int test_interval: the log interval in log_test_data(). Default to 1.
    :param int update_interval: the log interval in log_update_data().Default to 1000.
    :param save_interval: the log interval for checkpoints
    :param str project: W&B project name. Default to "tianshou".
    :param str name: W&B run name. Default to None. If None, random name is assigned.
    :param str entity: W&B team/organization name. Default to None.
    :param str run_id: run id of W&B run to be resumed. Default to None.
    :param argparse.Namespace config: experiment configurations. Default to None.

* Logging checkpoints and resuming runs on any device
    You need to define a ``save_checkpoint_fn`` which saves the experiment checkpoint and returns the path of the saved checkpoint. Here's an example.
:: 
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = os.path.join(log_path, 'checkpoint.pth')
        torch.save({'model': policy.state_dict()}, ckpt_path)
        return ckpt_path
Then, use this checkpointing function with ``WandbLogger`` to automatically version your experiment checkpoints after every ``save_interval`` steps

* Resuming runs from checkpoint artifacts on any device.
    
    Just pass the W&B ``run_id`` of the run that you want to resume in ``WandbLogger``. This will then download the lastest version of the checkpoint and resume your runs from the checkpoint.

    

