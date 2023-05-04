import argparse
import collections
import numpy as np
import random
import os

import model.loss as module_loss
import model.metric as module_metric
import model.model_lw as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils.util import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# fix random seeds for reproducibility
seed=42
random.seed(seed)
np.random.seed(seed) 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)
os.environ["PYTHONHASHSEED"] = str(seed)

def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



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
        
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

        self.x_data = self.x_data.to("cuda:0").float()
        #self.y_data = self.y_data.to("cuda:0").float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    

def data_generator_np(training_files, train_label, validation_files, val_label, test_files, test_label, batch_size):
    train_dataset = LoadDataset_from_numpy(training_files, train_label)
    val_dataset = LoadDataset_from_numpy(validation_files, val_label)
    test_dataset = LoadDataset_from_numpy(test_files, test_label)

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
                                               shuffle=False,
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


def main(config, training_files, train_label, validation_files, val_label, test_files, test_label):
    batch_size = config["data_loader"]["args"]["batch_size"]
    #batch_size = 16

    logger = config.get_logger('train')

    # build model architecture, initialize weights, then print to console
    model = config.init_obj('arch', module_arch)
    model.apply(weights_init_normal)
    logger.info(model)

    #PATH = 'C:/Users/KimTS/Desktop/GSR_attention_based/AttnSleep-main/saved/mcnn_1th_sp/20_11_2022_19_50_31/model_best.pth'
    #checkpoint = torch.load(PATH)
    #model.load_state_dict(checkpoint['state_dict'])
    #model.to('cuda')

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    
    #optimizer.load_state_dict(checkpoint['optimizer'])

    data_loader, valid_data_loader, test_data_loader, data_count = data_generator_np(training_files,
                                                                                     train_label, 
                                                                                     validation_files, 
                                                                                     val_label,
                                                                                     test_files,
                                                                                     test_label,
                                                                                     batch_size)
    #weights_for_each_class = calc_class_weight(data_count)
    weights_for_each_class = None


    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader = test_data_loader,
                      class_weights=weights_for_each_class)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
#    args.add_argument('-f', '--fold_id', type=str,
#                      help='fold_id')
#    args.add_argument('-da', '--np_data_dir', type=str,
#                      help='Directory containing numpy files')


    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    #fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, options)
#    if "shhs" in args2.np_data_dir:
#        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
#    else:
#        folds_data = load_folds_data(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    training_files = '/3_class/subject_1/X_train.npy'
    train_label = '/3_class/subject_1/y_train.npy'
    
    validation_files = '/3_class/subject_1/X_test.npy'
    val_label = '/3_class/subject_1/y_test.npy'

    test_files = '/3_class/subject_1/X_train.npy'
    test_label = '/3_class/subject_1/y_train.npy'

    #training_files = 'C:/Users/KimTS/Desktop/GSR_attention_based/wesad/overlap_5_120(30)/total/x_total.npy'
    #train_label = 'C:/Users/KimTS/Desktop/GSR_attention_based/wesad/overlap_5_120(30)/total/total_label.npy'

    #validation_files = 'C:/Users/KimTS/Desktop/Experiment_Protocol_1/MyDataset/AffectiveROAD_Data/Database/overlap_120(30)_sp/total/X_total.npy'
    #val_label = 'C:/Users/KimTS/Desktop/Experiment_Protocol_1/MyDataset/AffectiveROAD_Data/Database/overlap_120(30)_sp/total/y_total.npy'

    #test_files = 'C:/Users/KimTS/Desktop/Experiment_Protocol_1/MyDataset/DCU_NVT_EXP1/overlap_5.4(1)_sp/1th_dw/X_train.npy'
    #test_label = 'C:/Users/KimTS/Desktop/Experiment_Protocol_1/MyDataset/DCU_NVT_EXP1/overlap_5.4(1)_sp/1th_dw/y_train.npy'

    main(config, training_files, train_label, validation_files, val_label, test_files, test_label)
