from ResNet.dataset import SABREDataset


if __name__ == '__main__':
    dataset = SABREDataset("./data")
    
    dataset.plot_data(20)