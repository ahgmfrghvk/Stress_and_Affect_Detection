import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn
import time

selected_d = {"outs": [], "trg": []}
selected_test = {'outs':[], 'trg':[]}
class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, test_data_loader=None, class_weights=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        self.test_data_loader = test_data_loader


        self.lr_scheduler = optimizer
        self.log_step = int(data_loader.batch_size) * 1  # reduce this if you want more logs

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

        self.selected = 0
        self.val_selected = 0
        self.class_weights = class_weights

    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
               total_epochs: Integer, the total number of epoch
        :return: A log that contains average loss and metric in this epoch.
        """

        #starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_time = time.time()
        

        self.model.train()
        self.train_metrics.reset()
        overall_outs = []
        overall_trgs = []

        overall_test_outs = []
        overall_test_trgs = []

        #starter.record()
        for batch_idx, (data1, target) in enumerate(self.data_loader):
            data1, target = data1.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data1)

            loss = self.criterion(output, target, self.class_weights, self.device)

            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item(), len(data1))
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target), len(data1))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                ))

            if batch_idx == self.len_epoch:
                break
        #ender.record()
        #torch.cuda.synchronize()
        #curr_time = starter.elapsed_time(ender)
        #curr_time = round(curr_time, 2)
        #print('Training time: ', str(curr_time)+'ms')
        end_time = time.time()
        curr_time = (end_time-start_time)
        print('Training time: ', str(curr_time)+'sec')
        

        log = self.train_metrics.result()

        if self.do_validation:
            
            val_log, outs, trgs = self._valid_epoch(epoch)
            test_log, test_outs, test_trgs = self._test_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

            

            #if val_log["accuracy"] > self.selected:
            #    self.selected = val_log["accuracy"]
            if np.round(val_log["accuracy"],6) >= np.round(self.val_selected,6):

                #self.selected = log["accuracy"]
                self.val_selected = val_log["accuracy"]
                selected_d["outs"] = outs
                selected_d["trg"] = trgs

                self.test_selected = test_log["accuracy"]
                selected_test['outs'] = test_outs
                selected_test['trg'] = test_trgs
                print('last check')


            if epoch == total_epochs:
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])

                overall_test_outs.extend(selected_test['outs'])
                overall_test_trgs.extend(selected_test['trg'])

            # THIS part is to reduce the learning rate after 10 epochs to 1e-4
            if epoch == 10:
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.0001

        return self.selected, log, overall_outs, overall_trgs, overall_test_outs, overall_test_trgs, curr_time

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            for batch_idx, (data1, target) in enumerate(self.valid_data_loader):
                data1, target = data1.to(self.device), target.to(self.device)
                output = self.model(data1)
                loss = self.criterion(output, target, self.class_weights, self.device)

                self.valid_metrics.update('loss', loss.item(), len(data1))
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target), len(data1))

                preds_ = output.data.max(1, keepdim=True)[1].cpu()

                outs = np.append(outs, preds_.cpu().numpy())
                trgs = np.append(trgs, target.data.cpu().numpy())


        return self.valid_metrics.result(), outs, trgs


    def _test_epoch(self, epoch):
        """
        Evaluate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about evaluation
        """
        self.model.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            for batch_idx, (data1,target) in enumerate(self.test_data_loader):
                data1, target = data1.to(self.device), target.to(self.device)
                output = self.model(data1)
                loss = self.criterion(output, target, self.class_weights, self.device)

                self.test_metrics.update('loss', loss.item(), len(data1))
                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(output, target), len(data1))

                preds_ = output.data.max(1, keepdim=True)[1].cpu()

                outs = np.append(outs, preds_.cpu().numpy())
                trgs = np.append(trgs, target.data.cpu().numpy())


        return self.test_metrics.result(), outs, trgs

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx 
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
