import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics as sm
import time
import gc
from torch_geometric.nn import global_mean_pool
from training_func.MyspecialFunction import calculate_weighted_metrics

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    f1_meter, total_loss = 0, 0
    cnt = 0
    for mybatch in dataloader:
        inputs, labels, batch = mybatch.x, mybatch.y, mybatch.batch
        inputs = inputs.to(device)
        optimizer.zero_grad()
        # inputs = inputs.view(inputs.size(0), 1, inputs.size(1), inputs.size(2))

        outputs = model(inputs)  
        outputs = global_mean_pool(outputs, batch.to(device))
        outputs= torch.sigmoid(outputs.flatten())

        labels = labels.long().to(device)
        loss = criterion(outputs , labels.type(torch.float32))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def new_train_f(model, dataloader, channels, optimizer, criterion, device,flag=True):
    model.train()
    f1_meter, total_loss = 0, 0
    cnt = 0
    for mybatch in dataloader:
        inputs, labels, batch = mybatch.x, mybatch.y, mybatch.batch
        inputs = inputs.to(device)
        optimizer.zero_grad()
        inputs = inputs.view(-1, channels, inputs.size(1), inputs.size(2))

        outputs = model(inputs)  
        cnt += 1
        if cnt >3:
            flag = False

        outputs = global_mean_pool(outputs, batch.to(device))
        outputs= torch.sigmoid(outputs.flatten())

        labels = labels.long().to(device)
        loss = criterion(outputs , labels.type(torch.float32))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader), flag

def new_train(model, dataloader, channels, optimizer, criterion, device):
    model.train()
    f1_meter, total_loss = 0, 0
    cnt = 0
    conv_fea = []
    mlp_fea = []
    ref_labels = []
    for mybatch in dataloader:
        inputs, labels, batch = mybatch.x, mybatch.y, mybatch.batch
        inputs = inputs.to(device)
        optimizer.zero_grad()
        inputs = inputs.view(-1, channels, inputs.size(1), inputs.size(2))

        outputs = model(inputs)  

        outputs = global_mean_pool(outputs, batch.to(device))
        outputs= torch.sigmoid(outputs.flatten())

        labels = labels.long().to(device)
        loss = criterion(outputs , labels.type(torch.float32))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def KD_train(model, teacher_model,dataloader, optimizer, criterion, temp, alpha, device, batch_size):
    model.train()
    f1_meter, total_loss = 0, 0
    cnt = 0
    for mybatch in dataloader:

        inputs, labels, batch = mybatch.x, mybatch.y, mybatch.batch
        inputs = inputs.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)  
        outputs = global_mean_pool(outputs, batch.to(device))
        student_probs = torch.sigmoid(outputs.flatten() / temp)

        outputs= torch.sigmoid(outputs.flatten())
        
        teacher_inputs = torch.reshape(inputs, (batch_size, -1, inputs.shape[1], inputs.shape[2]))
        pred = teacher_model(teacher_inputs)

        teacher_probs = torch.sigmoid(pred / temp)
        distill_loss = F.binary_cross_entropy(student_probs, teacher_probs)

        labels = labels.long().to(device)
        BCEloss = criterion(outputs , labels.type(torch.float32))
        loss = alpha * BCEloss + (1 - alpha) * distill_loss 

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, channels,dataloader, criterion,device):
    model.eval() 

    f1_meter, val_loss = 0, 0
    with torch.no_grad():
        for mybatch in dataloader:

            inputs, labels, batch = mybatch.x, mybatch.y, mybatch.batch
            inputs = inputs.view(-1, channels, inputs.size(1), inputs.size(2))

            inputs = inputs.to(device) 
            outputs = model(inputs)

            outputs = global_mean_pool(outputs, batch.to(device))
            outputs= torch.sigmoid(outputs.flatten())

            labels = labels.to(device)
            loss = criterion(outputs , labels.type(torch.float32))
            val_loss += loss.item()

            y_pred_labels = (outputs >= 0.5).int()  # 如果概率 >= 0.5 则为类别1，否则为类别0
            precision, recall, f1, support = sm.precision_recall_fscore_support(labels.detach().cpu(), y_pred_labels.detach().cpu(), average='binary')
            f1_meter += f1

    return val_loss / len(dataloader), f1_meter / len(dataloader)

def test(model, dataloader, device, sample_weight,channels):
    model.eval()
    output_meter = []
    target_meter = []
    with torch.no_grad():
        for mybatch in dataloader:
            inputs, labels, batch  = mybatch.x, mybatch.y, mybatch.batch
            inputs = inputs.view(-1, channels, inputs.size(1), inputs.size(2))

            inputs = inputs.to(device) 

            outputs = model(inputs)
            outputs = global_mean_pool(outputs, batch.to(device))
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

    return precision, recall, f1, bca, spec,fpr

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

def test_infer(model, dataloader, channels,device):
    model.eval()
    output_meter = []
    target_meter = []
    inference_all = 0
    with torch.no_grad():
        for mybatch in dataloader:
            inputs, labels, batch  = mybatch.x, mybatch.y, mybatch.batch
            inputs = inputs.view(-1, channels, inputs.size(1), inputs.size(2))

            inputs = inputs.to(device) 

            start_time = time.time()
            outputs = model(inputs)

            outputs = global_mean_pool(outputs, batch.to(device))
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


def test_wrong_input(model, dataloader, device, sample_weight,channels):
    model.eval()
    output_meter = []
    target_meter = []
    wrong_input = []

    conv_fea = []
    mlp_fea = []
    ref_labels = []

    with torch.no_grad():
        for mybatch in dataloader:
            inputs, labels, batch  = mybatch.x, mybatch.y, mybatch.batch
            inputs = inputs.view(-1, channels, inputs.size(1), inputs.size(2))

            inputs = inputs.to(device) 

            outputs, conv_fea, mlp_fea = model(inputs)
            conv_fea.append(conv_fea)
            mlp_fea.append(mlp_fea)
            ref_labels.append(labels)
            outputs = global_mean_pool(outputs, batch.to(device))
            outputs = torch.round(torch.sigmoid(outputs).flatten()).long()

            target_meter.append(labels.int())
            output_meter.append(outputs)
            if labels.int().value() != ouputs.value():
                wrong_input.append(inputs)

        target_meter = torch.cat(target_meter,dim=0).numpy()
        output_meter = torch.cat(output_meter,dim=0).detach().cpu().numpy()
        print(target_meter)
        print(output_meter)

        precision, recall, f1, support = sm.precision_recall_fscore_support(target_meter, output_meter, average='weighted')
        bca = sm.balanced_accuracy_score(target_meter, output_meter)
         # tn, fp, fn, tp = sm.confusion_matrix(target_meter, output_meter).ravel() #int, int, int, int
        spec, fpr = calculate_weighted_metrics(target_meter, output_meter, sample_weight)

    return precision, recall, f1, bca, spec,fpr, wrong_input, conv_fea, mlp_fea, ref_labels