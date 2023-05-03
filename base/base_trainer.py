import torch
from abc import abstractmethod
from numpy import inf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime, time

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        #for name, param in self.model.named_parameters():
        #    if name.split('.')[0] == 'fc' or name.split('.')[0] == 'tce' or name.split('.')[1] == 'AFR':
        #        pass
        #    else :
        #        param.requires_grad = False

        #for name, param in self.model.named_parameters():
        #    print(name, param.requires_grad)


        not_improved_count = 0
        all_outs = []
        all_trgs = []

        all_test_outs = []
        all_test_trgs = []
        train_acc_result=0

        best = False
        best_epoch = 0
        best_model = self.model
        best_optimizer = self.optimizer
        best_mnt_best = self.mnt_best
        best_config = self.config

        
        timings=[]
        for epoch in range(self.start_epoch, self.epochs + 1):
            
            train_acc_result, result, epoch_outs, epoch_trgs, epoch_test_outs, epoch_test_trgs, epoch_curr_time = self._train_epoch(epoch, self.epochs)
           
            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            all_outs.extend(epoch_outs)
            all_trgs.extend(epoch_trgs)
            all_test_outs.extend(epoch_test_outs)
            all_test_trgs.extend(epoch_test_trgs)

            if epoch == 1:
                learning_df = pd.DataFrame([log])
            
            else:
                learning_df = learning_df.append(log, ignore_index=True)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            

            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    best_epoch = epoch
                    best_model = self.model
                    best_optimizer = self.optimizer
                    best_mnt_best = self.mnt_best
                    best_config = self.config

                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            #if epoch % self.save_period == 0:
            #    self._save_checkpoint(epoch, save_best=best)
            timings.append(epoch_curr_time)
        timings = np.array(timings)
            
        np.save(self.config._save_dir / "timings.npy", timings)
        best = True
        #self._save_checkpoint(best_epoch, best_model, best_optimizer, best_mnt_best, best_config, save_best=best)
        self._save_checkpoint(epoch, self.model, self.optimizer, self.mnt_best, self.config, save_best=False)


        outs_name = "outs_" 
        trgs_name = "trgs_" 
        np.save(self.config._save_dir / outs_name, all_outs)
        np.save(self.config._save_dir / trgs_name, all_trgs)

        test_outs_name = "test_out_" 
        test_trgs_name = "test_trg_" 
        np.save(self.config._save_dir / test_outs_name, all_test_outs)
        np.save(self.config._save_dir / test_trgs_name, all_test_trgs)

        print('Train ACC: ', train_acc_result)


        learning_df['Training_time'] = timings
        learning_df.to_csv(self.config._save_dir / "learning_curve.csv", mode='w')

        loss = learning_df['loss']
        val_loss = learning_df['val_loss']


        epochs = range(1, len(loss) + 1)

        plt.plot(epochs, loss, 'r', label='Training loss')      # bo -> 파란색 점
        plt.plot(epochs, val_loss, 'b', label='Validation loss') # b -> 파란색 실선
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.config._save_dir / 'loss_curve.png')
        plt.show()


        acc = learning_df['accuracy']           # 훈련 정확도
        val_acc = learning_df['val_accuracy']   # 검증 정확도

        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(self.config._save_dir / 'accuarcy_curve.png')
        plt.show()

        self._calc_metrics()

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, best_epoch, best_model, best_optimizer, best_mnt_best, best_config, save_best=True):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': best_epoch,
            'state_dict': best_model.state_dict(),
            'optimizer': best_optimizer.state_dict(),
            'monitor_best': best_mnt_best,
            'config': best_config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(best_epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            #self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _calc_metrics(self):
        from sklearn.metrics import classification_report
        from sklearn.metrics import cohen_kappa_score
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score
        import pandas as pd
        import os
        from os import walk

        all_outs = []
        all_trgs = []

        all_test_outs = []
        all_test_trgs = []

        outs_list = []
        trgs_list = []

        test_outs_list = []
        test_trgs_list = []

        save_dir = os.path.abspath(os.path.join(self.checkpoint_dir, os.pardir))
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                if "outs" in file:
                     outs_list.append(os.path.join(root, file))
                if "trgs" in file:
                     trgs_list.append(os.path.join(root, file))

                if "test_out" in file:
                     test_outs_list.append(os.path.join(root, file))
                if "test_trg" in file:
                     test_trgs_list.append(os.path.join(root, file))


        if len(outs_list)==self.config["data_loader"]["args"]["num_folds"]:
            for i in range(len(outs_list)):
                outs = np.load(outs_list[i])
                trgs = np.load(trgs_list[i])
                all_outs.extend(outs)
                all_trgs.extend(trgs)

                test_outs = np.load(test_outs_list[i])
                test_trgs = np.load(test_trgs_list[i])
                all_test_outs.extend(test_outs)
                all_test_trgs.extend(test_trgs)

        all_trgs = np.array(all_trgs).astype(int)
        all_outs = np.array(all_outs).astype(int)

        all_test_trgs = np.array(all_test_trgs).astype(int)
        all_test_outs = np.array(all_test_outs).astype(int)

        r = classification_report(all_trgs, all_outs, digits=6, output_dict=True)
        cm = confusion_matrix(all_trgs, all_outs)
        df = pd.DataFrame(r)
        df["cohen"] = cohen_kappa_score(all_trgs, all_outs)
        df["accuracy"] = accuracy_score(all_trgs, all_outs)
        df = df * 100
        file_name = self.config["name"] + "_classification_report.xlsx"
        report_Save_path = os.path.join(save_dir, file_name)
        df.to_excel(report_Save_path)

        cm_file_name = self.config["name"] + "_confusion_matrix.torch"
        cm_Save_path = os.path.join(save_dir, cm_file_name)
        torch.save(cm, cm_Save_path)



        r_test = classification_report(all_test_trgs, all_test_outs, digits=6, output_dict=True)
        cm = confusion_matrix(all_test_trgs, all_test_outs)
        df = pd.DataFrame(r_test)
        df["cohen"] = cohen_kappa_score(all_test_trgs, all_test_outs)
        df["accuracy"] = accuracy_score(all_test_trgs, all_test_outs)
        df = df * 100
        file_name = self.config["name"] + "_test_classification_report.xlsx"
        report_Save_path = os.path.join(save_dir, file_name)
        df.to_excel(report_Save_path)

        cm_file_name = self.config["name"] + "_test_confusion_matrix.torch"
        cm_Save_path = os.path.join(save_dir, cm_file_name)
        torch.save(cm, cm_Save_path)


        # Uncomment if you want to copy some of the important files into the experiement folder
        # from shutil import copyfile
        # copyfile("model/model.py", os.path.join(self.checkpoint_dir, "model.py"))
        # copyfile("model/loss.py", os.path.join(self.checkpoint_dir, "loss.py"))
        # copyfile("trainer/trainer.py", os.path.join(self.checkpoint_dir, "trainer.py"))
        # copyfile("train_Kfold_CV.py", os.path.join(self.checkpoint_dir, "train_Kfold_CV.py"))
        # copyfile("config.json",  os.path.join(self.checkpoint_dir, "config.json"))
        # copyfile("data_loader/data_loaders.py",  os.path.join(self.checkpoint_dir, "data_loaders.py"))



