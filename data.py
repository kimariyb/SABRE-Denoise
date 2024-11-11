import random

from torch.utils.data import Subset, DataLoader
from pytorch_lightning import LightningDataModule

from TransUNet.dataset import SABREDataset


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
        train_size = int(0.85 * num_samples)
        
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

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            drop_last=False,
        )
        
        if store_dataloader:
            self._saved_dataloaders[stage] = dataloader
            
        return dataloader