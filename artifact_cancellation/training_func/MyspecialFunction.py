import numpy as np
from scipy.signal import hilbert

def getLineLength(array):
    diffarray = array[1:array.shape[0]]
    diffarray2 = array[0:array.shape[0]-1]
    absarray = np.abs(diffarray - diffarray2)

    return np.sum(absarray)

def aggregation(array, num):
    arraySquare = array * array
    # print(np.ceil(array.shape[0]/num))
    paddingLength = np.ceil(array.shape[0]/num) * num - array.shape[0]
    arraySquare_padding = np.pad(arraySquare, pad_width=(0, int(paddingLength)), mode='constant', constant_values=0)
    sumPre = arraySquare_padding.reshape((-1, num))
    arraySum = np.sum(sumPre, axis=1)
    return arraySum

def aggregation_List(array, aggList):
    arraySquare = array * array
    sumArray = []
    for i in range(0, len(aggList)-1):
        sumArray.append(np.sum(arraySquare[aggList[i]:aggList[i+1]]))
    sumArray = np.array(sumArray)
    return sumArray

def getEnvelope(data):
    analytic_signal = hilbert(data)
    envelope = np.abs(analytic_signal)
    return envelope 
    
def getLBP(data):
    LBP = []
    for i in range(data.shape[0]):
        if ((i < 4) or (i >= data.shape[0]-4)):
            continue
        else:
            center = data[i]   
            compareArray0 = np.array(a[i-4:i])
            compareArray1 = np.array(a[i+1:i+5])
            compareArray = np.concatenate((compareArray0, compareArray1))
            binaryArray = compareArray > center
            boolArray = binaryArray.astype('int')
            binaryNum = ''.join(boolArray.astype('str'))
            print(binaryNum)
            print(int(binaryNum, 2))
            LBP.append(int(binaryNum, 2))
    LBP = np.array(LBP)
    return LBP    

def MA(data):
    MA_result = np.mean(data,axis=1)
    return MA_result

def last_relevant_pytorch(output, lengths, batch_first=True):
    lengths = lengths.cpu()

    # masks of the true seq lengths
    masks = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2))
    time_dimension = 1 if batch_first else 0
    masks = masks.unsqueeze(time_dimension)
    masks = masks.to(output.device)
    last_output = output.gather(time_dimension, masks).squeeze(time_dimension)
    last_output.to(output.device)

    return last_output

def calculate_weighted_metrics(y_true, y_pred, sample_weights):
    """
    计算加权特异性 (Specificity) 和加权假正率 (FPR)。

    参数:
    - y_true: list 或 numpy array，真实标签 (0 或 1)。
    - y_pred: list 或 numpy array，预测标签 (0 或 1)。
    - sample_weights: list 或 numpy array，每个样本的权重。

    返回:
    - weighted_specificity: 加权特异性。
    - weighted_fpr: 加权假正率。
    """
    # 转换为 numpy 数组以便操作
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sample_weights = np.array(sample_weights)

    # 确保输入一致
    assert len(y_true) == len(y_pred) == len(sample_weights), "输入长度不一致。"

    # 计算 TN 和 FP 的权重
    tn_weights = sample_weights[(y_true == 0) & (y_pred == 0)]  # TN 权重
    fp_weights = sample_weights[(y_true == 0) & (y_pred == 1)]  # FP 权重

    # 统计加权的 TN 和 FP
    weighted_tn = np.sum(tn_weights)
    weighted_fp = np.sum(fp_weights)

    # 计算加权特异性和 FPR
    denominator = weighted_tn + weighted_fp
    if denominator > 0:
        weighted_specificity = weighted_tn / denominator
        weighted_fpr = weighted_fp / denominator
    else:
        weighted_specificity = 0
        weighted_fpr = 0

    return weighted_specificity, weighted_fpr