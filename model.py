import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning import LightningModule

from TransUNet.model import create_model, load_model


class SabreModel(LightningModule):
    def __init__(self, hparams) -> None:
        super(SabreModel, self).__init__()

        self.save_hyperparameters(hparams)
        
        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, args=self.hparams)
        else:
            self.model = create_model(self.hparams)
            
        self._reset_losses_dict()
        
    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.lr_min,
        )
        
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]
    
    def forward(self, batch):
        return self.model(batch['z'])
    
    def training_step(self, batch, batch_idx):
        return self._process_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._process_step(batch, "val")
    
    def _process_step(self, batch, stage):
        with torch.set_grad_enabled(stage == "train"):
            pred = self(batch)

        loss = 0.0

        labels = batch["labels"]
        
        self.losses[stage].append(loss.detach())
        
        return loss
    
    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()
        
    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            result_dict = {
                "epoch": float(self.current_epoch),
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": torch.stack(self.losses["train"]).mean(),
                "val_loss": torch.stack(self.losses["val"]).mean(),
            }
            
            self.log_dict(result_dict, sync_dist=True)
            
        self._reset_losses_dict()  
        
    def _reset_losses_dict(self):
        self.losses = {
            "train": [], "val": [], 
        }