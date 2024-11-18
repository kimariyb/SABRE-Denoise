import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning import LightningModule

from TransUNet.main_model import create_model


class SabreModel(LightningModule):
    def __init__(self, hparams) -> None:
        super(SabreModel, self).__init__()

        self.save_hyperparameters(hparams)
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
        return self.model(batch['raws'])
    
    def training_step(self, batch, batch_idx):
        return self._process_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._process_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        with torch.set_grad_enabled(False):
            pred = self(batch)
            
        label = batch['labels']
        
        self._plot_spectra(pred, label)
        
    def _process_step(self, batch, stage):
        with torch.set_grad_enabled(stage == "train"):
            pred = self(batch)
            
        label = batch['labels']

        loss = 0.0
        loss = self._calc_loss(pred, label)
        
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
        
    def _calc_loss(pred, label):
        """
        Calculate the loss between the predicted and the label.
        
        Parameters
        ----------
        pred : torch.Tensor
            The predicted complex-valued image.
        label : torch.Tensor
            The label complex-valued image.
        
        Returns
        -------
        loss : torch.Tensor
            The loss between the predicted and the label.
        """
        pred_complex = pred[:, 0] + 1j * pred[:, 1]
        label_complex = label[:, 0] + 1j * label[:, 1]

        loss = nn.functional.mse_loss(torch.abs(pred_complex), torch.abs(label_complex))
        
        return loss 
    
    
    def _plot_spectra(self, pred, label):
        # 将 tensor 转换为 numpy 数组
        pred = pred.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        
        # 生成 x 轴坐标和 y 轴坐标
        x = range(pred.shape[2])
        pred_y = pred[0, 0, :].reshape(-1)
        label_y = label[0, 0, :].reshape(-1)
        
        # 绘制上下两个谱图
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(x, pred_y, color='r')
        plt.title("Predicted")
        plt.subplot(2, 1, 2)
        plt.plot(x, label_y, color='b')
        plt.title("Label")
        plt.show()

        