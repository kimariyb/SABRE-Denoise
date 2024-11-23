import random
import torch

from torch.utils.data import Subset, DataLoader
from pytorch_lightning import LightningDataModule

from SabreNet.dataset import SABREDataset, SABRETestDataset


class SabreDataCollator:
    def __call__(self, features):
        batch = dict()
        
        batch["labels"] = torch.stack(
            [feat.label for feat in features]
        )
        
        batch["raws"] = torch.stack(
            [feat.raw for feat in features]
        )
        
        return batch  


class SabreTestModule(LightningDataModule): 
    def __init__(self, root):
        super(SabreTestModule, self).__init__()
        self.dataset = SABRETestDataset(root=root)
        self.collator = SabreDataCollator()
        
    def test_dataloader(self):
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collator,
        )
        
        return dataloader


class SabreDataModule(LightningDataModule):
    def __init__(self, hparams):
        super(SabreDataModule, self).__init__()
        self.hparams.update(hparams.__dict__) if hasattr(
            hparams, "__dict__"
        ) else self.hparams.update(hparams)
        self._saved_dataloaders = dict()
        self.dataset = None
        
    def prepare_dataset(self):  
        # Load dataset
        self.dataset = SABREDataset(root=self.hparams["dataset_root"])
        # 分割 indices
        num_samples = len(self.dataset)
        train_size = int(0.9 * num_samples)
        
        self.idx_train = random.sample(range(num_samples), train_size)
        self.idx_val = [i for i in range(num_samples) if i not in self.idx_train]
        
        print(
            f"train {len(self.idx_train)}, val {len(self.idx_val)}"
        )
        
        self.train_dataset = Subset(self.dataset, self.idx_train)
        self.val_dataset = Subset(self.dataset, self.idx_val)
        
    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, "val")
    
    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        store_dataloader = store_dataloader and not self.hparams["reload"]
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage == "val":
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False
            
        collate_fn = SabreDataCollator()

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn
        )
        
        if store_dataloader:
            self._saved_dataloaders[stage] = dataloader
            
        return dataloader