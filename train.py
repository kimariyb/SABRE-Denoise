import os
import re

import torch
import pytorch_lightning as pl

from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary

from data import SabreDataModule, SabreTestModule
from model import SabreModel

from TransUNet.dataset import NMRData

from argparse import Namespace
from datetime import datetime


class Hyperparameters(Namespace):
    def __init__(self):
        # Initialize default values
        super().__init__(
            load_model=None,
            embedding_dim=2048,
            ffn_embedding_dim=8192,
            num_heads=16,
            num_layers=9,
            patch_size=16,
            dropout=0.5,
            attn_dropout=0.5,

            loss_type="mae",
            num_epochs=50,
            lr_warmup_steps=10000,
            lr=2.e-04,
            lr_patience=5,
            lr_min=1.e-07,
            lr_factor=0.8,
            weight_decay=1.e-03,
            early_stopping_patience=15,
        
            reload=1,
            batch_size=32,
            inference_batch_size=32,
            dataset_root='./data',
            test_root='./test',
            train_size=None,
            val_size=None,
            num_workers=16,
          
            num_nodes=1,
            precision=32,
            log_dir="./log",
            seed=42,
            accelerator="gpu",
            save_interval=1,
            task="train"
        )


def auto_start(args):
    dir_name = (
        f"bs_{args.batch_size}"
        + f"_L{args.num_layers}_D{args.embedding_dim}_F{args.ffn_embedding_dim}"
        + f"_H{args.num_heads}"
        + f"_P{args.patch_size}"
        + f"_lr_{args.lr}"
        + f"_drop_{args.dropout}"
        + f"_loss_{args.loss_type}"
        + f"_seed_{args.seed}"
    )

    if args.load_model is None:    
        args.log_dir = os.path.join(args.log_dir, dir_name)
        if os.path.exists(args.log_dir):
            if os.path.exists(
                os.path.join(args.log_dir, "checkpoints", "last.ckpt")
            ):
                args.load_model = os.path.join(
                    args.log_dir, "checkpoints", "last.ckpt"
                )
                print(
                    f"***** model {args.log_dir} exists, resuming from the last checkpoint *****"
                )
                
            csv_path = os.path.join(args.log_dir, "metrics", "metrics.csv")
            
            while os.path.exists(csv_path):
                csv_path = csv_path + ".bak"
                
            if os.path.exists(
                os.path.join(args.log_dir, "metrics", "metrics.csv")
            ):
                os.rename(
                    os.path.join(args.log_dir, "metrics", "metrics.csv"),
                    csv_path,
                )

    return args


def main():
    args = Hyperparameters()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set seed
    pl.seed_everything(args.seed, workers=True)
    args = auto_start(args)
    
    # Initialize model
    model = SabreModel(args).to(device)
    
    # Initialize logger
    csv_logger = CSVLogger(args.log_dir, name="metrics", version="")
    
    if args.task == "train":
        # Initialize data module
        data = SabreDataModule(args)
        data.prepare_dataset()

        # Initialize callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.log_dir, "checkpoints"),
            monitor="val_loss",
            save_top_k=10,
            save_last=True,
            every_n_epochs=args.save_interval,
            filename="{epoch}-{val_loss:.4f}",
        )

        early_stopping = EarlyStopping(
            "val_loss", patience=args.early_stopping_patience
        )
        
        # Initialize logger
        tb_logger = TensorBoardLogger(
            args.log_dir,
            name="tensorbord",
            version="",
            default_hp_metric=False,
        )

        strategy = SingleDeviceStrategy(device)

        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            num_nodes=args.num_nodes,
            accelerator=args.accelerator,
            deterministic=False,
            default_root_dir=args.log_dir,
            callbacks=[early_stopping, checkpoint_callback, ModelSummary(max_depth=1)],
            logger=[tb_logger, csv_logger],
            reload_dataloaders_every_n_epochs=args.reload,
            precision=args.precision,
            strategy=strategy,
            enable_progress_bar=True,
            inference_mode=False,
        )

        start_time = datetime.now()

        trainer.fit(model, datamodule=data, ckpt_path=args.load_model)
        
        print(f"Training completed in {datetime.now() - start_time}")
    
    elif args.task == "test":
        test_trainer = pl.Trainer(
            enable_model_summary=True,
            logger=[csv_logger],
            max_epochs=-1,
            num_nodes=1,
            devices=1,
            default_root_dir=args.log_dir,
            enable_progress_bar=True,
            callbacks=[ModelSummary()],
            accelerator=args.accelerator,
            inference_mode=False,
        )
        
        # 创建测试用例
        test_data = SabreTestModule(root=args.test_root)
        
        # 读取测试模型
        ckpt = torch.load(args.load_model, map_location="cpu")
        state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
        model.model.load_state_dict(state_dict)

        # 开始测试
        test_trainer.test(model=model, datamodule=test_data)

if __name__ == "__main__":
    main()
