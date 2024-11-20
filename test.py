from TransUNet.dataset import SABREDataset

dataset = SABREDataset(root=r'./data/')

dataset[6].plot()