from model.dataset import SABREDataset, SABRETestDataset


if __name__ == '__main__':
    dataset = SABREDataset("./data")
    test_dataset = SABRETestDataset("./test")
    
    test_dataset.plot_data(7)