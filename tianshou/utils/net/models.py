import lightning.pytorch as pl
import logging
import torch
from copy import copy
from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import TensorDataset, random_split
from typing import Any, Callable, Optional, Tuple

from tianshou.data.utils.batching import BatchDataLoader
from tianshou.utils.types import TOptimFactory

log = logging.getLogger(__name__)


def get_default_pl_callbacks(es_monitor: Optional[str] = "val_loss") -> list[Callback]:
    callbacks = []
    if es_monitor is not None:
        callbacks.append(
            EarlyStopping(monitor=es_monitor, patience=3, verbose=False, mode="min")
        )
    return callbacks


def get_default_pl_trainer(
    max_epochs: int,
    min_epochs=1,
    barebones=False,
    es_monitor: Optional[str] = "val_loss",
) -> pl.Trainer:
    if barebones:
        if es_monitor is not None:
            log.warning(
                "Early stopping not supported in barebones mode, "
                "setting es_monitor to None"
            )
            es_monitor = None

    callbacks = get_default_pl_callbacks(es_monitor=es_monitor)
    return pl.Trainer(
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        barebones=barebones,
        callbacks=callbacks,
        devices=1,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        num_sanity_val_steps=0,
    )


class PLTrainable(pl.LightningModule):
    def __init__(
        self,
        module: nn.Module,
        pl_trainer: pl.Trainer,
        loss_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = nn.functional.mse_loss,
        optim_factory: TOptimFactory = torch.optim.Adam,
        lr: float = 3e-4,
        lr_scheduler_factory: Optional[Callable[[Optimizer], LRScheduler]] = None,
        train_ratio: float = 1,
        shuffle_on_fit: bool = False,
    ):
        super().__init__()
        self.module = module
        self.loss_fn = loss_fn
        self.lr = lr
        self.optim_class = optim_factory
        self.scheduler_factory = lr_scheduler_factory
        self.train_ratio = train_ratio
        self.shuffle_on_fit = shuffle_on_fit
        self._train_losses = []

        es_monitor = "val_loss" if train_ratio < 1 else None
        # IMPORTANT: .trainer is added to the model by lightning during fit
        # This might have undesired consequences, so it's probably better to not
        # use this reserved attribute, although here the set trainer
        # might actually point to the user-provided trainer in all cases
        self.pl_trainer = pl_trainer or get_default_pl_trainer(es_monitor=es_monitor)

    def configure_optimizers(self) -> Any:
        optim = self.optim_class(self.parameters(), lr=self.lr)
        result = {
            "optimizer": optim,
        }
        if self.scheduler_factory is not None:
            result["lr_scheduler"] = self.scheduler_factory(optim)
        return result

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], args, **kwargs):
        X, Y = batch
        loss = self._get_loss(X, Y)
        self.log("train_loss", loss)
        self._train_losses.append(loss.item())
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], *args, **kwargs
    ):
        X, Y = batch
        loss = self._get_loss(X, Y)
        self.log("val_loss", loss)
        return loss

    # TODO: doesn't cover cases of more complicated forwards (e.g. with actions)
    #  Should be extended if ever needed
    def _get_loss(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        v = self(X)
        return self.loss_fn(v, Y)

    def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.module(X, *args, **kwargs)

    def _reset_trainer(self):
        self.pl_trainer.fit_loop.epoch_progress.reset()

    def fit(self, X: torch.Tensor, Y: torch.Tensor, batch_size=1024):
        train_dataloader, val_dataloader = self._get_dataloaders(X, Y, batch_size)

        self._reset_trainer()
        self.pl_trainer.fit(
            self,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        losses = copy(self._train_losses)
        self._train_losses = []
        return losses

    def _get_dataloaders(self, X: torch.Tensor, Y: torch.Tensor, batch_size: int):
        X = torch.as_tensor(X, dtype=torch.float32)
        Y = torch.as_tensor(Y, dtype=torch.float32)

        dataset = TensorDataset(X, Y)

        if self.train_ratio == 1:
            train_dataloader = BatchDataLoader(
                dataset, batch_size=batch_size, shuffle=self.shuffle_on_fit
            )
            return train_dataloader, None

        train_size = int(self.train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        # Don't use torch's DataLoader since it's unbearable slow
        # See https://github.com/pytorch/pytorch/issues/106704
        train_dataloader = BatchDataLoader(
            train_dataset, batch_size=batch_size, shuffle=self.shuffle_on_fit
        )
        val_dataloader = BatchDataLoader(val_dataset, batch_size=batch_size)
        return train_dataloader, val_dataloader
