import torch
from torch.utils.data import Dataset
import os
import numpy as np

class LoadDataset_from_numpy(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, x_data_file, y_data_file):
        super(LoadDataset_from_numpy, self).__init__()

        # load files
        X_train = np.load(x_data_file)
        y_train = np.load(y_data_file)

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()
        
        self.x_data = self.x_data.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    
#def seed_worker(_worker_id):
#    worker_seed = torch.initial_seed() % 2**32
#    np.random.seed(worker_seed)
#    random.seed(worker_seed)
    

def data_generator_np(training_files, validation_files, test_files, batch_size, seed_worker):
    train_dataset = LoadDataset_from_numpy(training_files)
    val_dataset = LoadDataset_from_numpy(validation_files)
    test_dataset = LoadDataset_from_numpy(test_files)

    # to calculate the ratio for the CAL
    
    ## train에만 있는 데이터만 가지고 label 비율 계산해야 한다고 생각함
    #all_ys = np.concatenate((train_dataset.y_data, test_dataset.y_data))
    all_ys = train_dataset.y_data.tolist()
    num_classes = len(np.unique(all_ys))
    
    ## label 확인하고 진행필요
    counts = [all_ys.count(i) for i in range(num_classes)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0,
                                               worker_init_fn=seed_worker)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0,
                                               worker_init_fn=seed_worker)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0,
                                              worker_init_fn=seed_worker)

    return train_loader, val_loader, test_loader, counts