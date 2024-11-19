from TransUNet.dataset import SABREDataset

dataset = SABREDataset(root=r'./data/', nums=1)

dataset[1].plot()