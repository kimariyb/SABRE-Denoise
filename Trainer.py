import dill

if __name__ == '__main__':
    with open('./data/saber_data.pkl', 'rb') as f:
        dataset = dill.load(f)
    
    print(dataset[1])