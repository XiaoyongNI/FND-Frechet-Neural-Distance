import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics as sm
import time
from training_func.MyspecialFunction import calculate_weighted_metrics
from sklearn.metrics import roc_auc_score
from scipy.special import expit  # sigmoid
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

def train(model, dataloader, optimizer, criterion, modeltype, device):
    model.train()
    f1_meter, total_loss = 0, 0
    cnt = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device) 
        optimizer.zero_grad()
        if modeltype == 'GRU_multic':
            outputs = model(inputs).squeeze()
            # outputs= torch.sigmoid(outputs.reshape(-1,1)).squeeze()

        elif modeltype == 'GRU_fullc':
            outputs = model(inputs).squeeze()
            # outputs= torch.sigmoid(outputs.reshape(-1,1)).squeeze()
        elif modeltype == 'REST':
            outputs = model(inputs)
            outputs = torch.mean(outputs, dim=-2)
            outputs= torch.sigmoid(outputs.flatten())

        else: #CNN
            outputs = model(inputs).squeeze()
        # y_pred_labels = (outputs >= 0.5).int()  # 如果概率 >= 0.5 则为类别1，否则为类别0

        labels = labels.long().to(device)
        # one_hot_label = F.one_hot(labels).to(device)

        loss = criterion(outputs , labels.type(torch.float32))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def train_new(model, dataloader, optimizer, criterion, modeltype, device):
    model.train()
    f1_meter, total_loss = 0, 0
    cnt = 0
    conv_fea1 = []
    conv_fea2 = []
    ref_labels = []
    for inputs, labels in dataloader:
        inputs = inputs.to(device) 
        optimizer.zero_grad()
        if modeltype == 'GRU_multic':
            outputs = model(inputs).squeeze()
            # outputs= torch.sigmoid(outputs.reshape(-1,1)).squeeze()

        elif modeltype == 'GRU_fullc':
            outputs = model(inputs).squeeze()
            # outputs= torch.sigmoid(outputs.reshape(-1,1)).squeeze()
        elif modeltype == 'REST':
            outputs = model(inputs)
            outputs = torch.mean(outputs, dim=-2)
            outputs= torch.sigmoid(outputs.flatten())

        else: #CNN
            outputs, conv1_fea, conv2_fea = model(inputs).squeeze()
            conv_fea1.append(conv1_fea)
            conv_fea2.append(conv2_fea)
            ref_labels.append(labels)
        # y_pred_labels = (outputs >= 0.5).int()  # 如果概率 >= 0.5 则为类别1，否则为类别0

        labels = labels.long().to(device)
        # one_hot_label = F.one_hot(labels).to(device)

        loss = criterion(outputs , labels.type(torch.float32))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader), conv_fea1, conv_fea2, ref_labels

def evaluate(model, dataloader, criterion, device, modeltype ):
    model.eval() 

    f1_meter, val_loss = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device) 
            if modeltype == 'GRU_multic':
                outputs = model(inputs).squeeze()
                # outputs= torch.sigmoid(outputs.reshape(-1,1)).squeeze()

            elif modeltype == 'GRU_fullc':
                outputs = model(inputs).squeeze()
                # outputs= torch.sigmoid(outputs.reshape(-1,1)).squeeze()

            elif modeltype == 'REST':
                outputs = model(inputs)
                outputs = torch.mean(outputs, dim=-2)
                outputs= torch.sigmoid(outputs.flatten())

            else: #CNN
                outputs = model(inputs).squeeze()

            labels = labels.long().to(device)
            # one_hot_label = F.one_hot(labels).to(device)

            loss = criterion(outputs , labels.type(torch.float32))
            val_loss += loss.item()
            y_pred_labels = (outputs >= 0.5).int()  # 如果概率 >= 0.5 则为类别1，否则为类别0
            precision, recall, f1, support = sm.precision_recall_fscore_support(labels.detach().cpu(), y_pred_labels.detach().cpu(), average='binary')
            f1_meter += f1

    return val_loss / len(dataloader), f1_meter / len(dataloader)

def test(model, dataloader, device, modeltype, sample_weight):
    model.eval()
    output_meter = []
    target_meter = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device) 
            if modeltype == 'GRU_multic':
                outputs = model(inputs)
                outputs = torch.round(torch.sigmoid(outputs).flatten()).long()

            elif modeltype == 'GRU_fullc':
                outputs = model(inputs)
                outputs = torch.round(torch.sigmoid(outputs).flatten()).long()
            elif modeltype == 'REST':
                outputs = model(inputs) #batch,channel,1
                outputs = torch.mean(outputs, dim=-2)#batch,1

                outputs= torch.round(torch.sigmoid(outputs.flatten())).long()

            else: #CNN
                outputs = model(inputs)
                outputs = torch.round(torch.sigmoid(outputs).flatten()).long()

            target_meter.append(labels.int())
            output_meter.append(outputs)

        target_meter = torch.cat(target_meter,dim=0).numpy()
        output_meter = torch.cat(output_meter,dim=0).detach().cpu().numpy()
        print(target_meter.shape)
        print(output_meter.shape)

        precision, recall, f1, support = sm.precision_recall_fscore_support(target_meter, output_meter, average='weighted')
        bca = sm.balanced_accuracy_score(target_meter, output_meter)
        # tn, fp, fn, tp = sm.confusion_matrix(target_meter, output_meter).ravel() #int, int, int, int
        spec, fpr = calculate_weighted_metrics(target_meter, output_meter, sample_weight)

        # probs = expit(output_meter)

        auc = roc_auc_score(target_meter, output_meter)
        accuracy = sm.accuracy_score(target_meter, output_meter)
        recall = recall_score(target_meter, output_meter, average='binary')

        cm = confusion_matrix(target_meter, output_meter)
        print("Confusion Matrix:")
        print(cm)

    return precision, recall, f1, bca, spec,fpr, auc, accuracy

def test_infer(model, dataloader, device, modeltype):
    model.eval()
    output_meter = []
    target_meter = []
    inference_all = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device) 
            if modeltype == 'GRU_multic':
                start_time = time.time()
                outputs = model(inputs)
                outputs = torch.round(torch.sigmoid(outputs).flatten()).long()
                end_time = time.time()
            elif modeltype == 'GRU_fullc':
                start_time = time.time()
                outputs = model(inputs)
                outputs = torch.round(torch.sigmoid(outputs).flatten()).long()
                end_time = time.time()
            elif modeltype == 'REST':
                start_time = time.time()
                outputs = model(inputs)
                outputs = torch.mean(outputs, dim=-2)
                outputs= torch.round(torch.sigmoid(outputs.flatten())).long()
                end_time = time.time()
            else: #CNN
                start_time = time.time()
                outputs = model(inputs)
                outputs = torch.round(torch.sigmoid(outputs).flatten()).long()
                end_time = time.time()

            inference_time = (end_time - start_time) * 1000# 转换为毫秒
            inference_all += inference_time

            target_meter.append(labels.int())
            output_meter.append(outputs)

        target_meter = torch.cat(target_meter,dim=0).numpy()
        output_meter = torch.cat(output_meter,dim=0).detach().cpu().numpy()
        print(target_meter)
        print(output_meter)
        inference_time_sample = inference_all / len(target_meter)
        print(f"Inference time for a single sample: {inference_time_sample:.3f} ms")
    return inference_time_sample

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# time consume
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


def test_detail(model, dataloader, device, modeltype, sample_weight):
    model.eval()
    output_meter = []
    target_meter = []
    conv_fea1 = []
    conv_fea2 = []
    ref_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device) 
            if modeltype == 'GRU_multic':
                outputs = model(inputs)
                outputs = torch.round(torch.sigmoid(outputs).flatten()).long()

            elif modeltype == 'GRU_fullc':
                outputs = model(inputs)
                outputs = torch.round(torch.sigmoid(outputs).flatten()).long()
            elif modeltype == 'REST':
                outputs = model(inputs) #batch,channel,1
                outputs = torch.mean(outputs, dim=-2)#batch,1

                outputs= torch.round(torch.sigmoid(outputs.flatten())).long()

            else: #CNN
                outputs, conv1_fea, conv2_fea = model(inputs)#.squeeze()
                conv_fea1.append(conv1_fea)
                conv_fea2.append(conv2_fea)
                ref_labels.append(labels)

                outputs = torch.round(torch.sigmoid(outputs).flatten()).long()

            target_meter.append(labels.int())
            output_meter.append(outputs)
        

        target_meter = torch.cat(target_meter,dim=0).numpy()
        output_meter = torch.cat(output_meter,dim=0).detach().cpu().numpy()
        print(target_meter)
        print(output_meter)

        precision, recall, f1, support = sm.precision_recall_fscore_support(target_meter, output_meter, average='weighted')
        bca = sm.balanced_accuracy_score(target_meter, output_meter)
        # tn, fp, fn, tp = sm.confusion_matrix(target_meter, output_meter).ravel() #int, int, int, int
        spec, fpr = calculate_weighted_metrics(target_meter, output_meter, sample_weight)

    return precision, recall, f1, bca, spec,fpr, conv_fea1,conv_fea2, ref_labels