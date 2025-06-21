'''
###Funcition:ablation_eperiments of manuscript:'Transformer-CNN Neural Network of Depth Domain
    Adaptive Interaction for Tumor Visually Interpretable Grading of Laryngeal Cancer
    IHC Histopathology Multi-Scale Images'.
###Author: Dr.Pan Huang
###E-mail: panhuang@cqu.edu.cn
###Creating Data:2022.01.24
###Department: College of Optoelectronic, Chongqing University, CHINA.
'''
import torch
import numpy as np
from torch import nn
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def save_model(model=None, file_path=None, file_name=None):
    weight_file = model.state_dict()
    torch.save(weight_file, file_path + '\\' + file_name)

def load_model(file_path=None):
    return torch.load(file_path)

def to_category(label_tensor=None, class_num=3):
    label_tensor = label_tensor.cpu().numpy()
    label_inter = np.zeros((label_tensor.size, class_num))
    for i in range(label_tensor.size):
        label_inter[i, int(label_tensor[i])] = 1
    return label_inter

def to_np_category(label_tensor=None, class_num=3):
    label_inter = np.zeros((label_tensor.size, class_num))
    for i in range(label_tensor.size):
        label_inter[i, int(label_tensor[i])] = 1
    return label_inter


def acc_scores(model=None, data_loader=None, gpu_device=0, out_mode='triplet', class_num = 3):
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    test_acc = []
    sum_label = torch.zeros(2).cuda(gpu_device)
    pre_label = torch.zeros(2).cuda(gpu_device)
    pre_value = torch.zeros((2, 3)).cuda(gpu_device)
    true_label = torch.zeros(2).cuda(gpu_device)
    for test_img, test_label in data_loader:
        test_img = test_img.cuda(gpu_device)
        test_label = test_label.cuda(gpu_device)
        with torch.no_grad():
            if out_mode == 'triplet':
                _, _, test_pre_y = model(test_img)
            elif out_mode == 'single':
                test_pre_y = model(test_img)
            test_loss = loss_fn(test_pre_y, test_label)
            test_pre_label = torch.argmax(test_pre_y, dim=1)
            test_acc.append(accuracy_score(test_label.detach().cpu().numpy(),
                                           test_pre_label.detach().cpu().numpy()))
            pre_label = torch.cat((pre_label, test_pre_label))
            sum_label = torch.cat((sum_label, test_label))
            pre_value = torch.cat((pre_value, test_pre_y))
    pre_label = pre_label[2:]
    sum_label = sum_label[2:]
    pre_value =  pre_value[2:]
    print('-----------------------------------------------------------------------')
    print(' test_acc:{:.4}'.format(np.mean(test_acc)))
    print('-----------------------------------------------------------------------')
    print('classification_report:', '\n', classification_report(sum_label.cpu().numpy(), pre_label.cpu().numpy(), digits=4))
    print('-----------------------------------------------------------------------')
    print('AUC:',roc_auc_score(to_category(sum_label, class_num = class_num), to_category(pre_label, class_num = class_num)))
    print('-----------------------------------------------------------------------')
    return pre_value, sum_label

def features_extraction(model=None, data_loader=None, gpu_device=0, out_mode='triplet'):
    model.eval()
    test_acc = []
    vit_features = torch.zeros(2, 768).cuda(gpu_device)
    fab_features = torch.zeros(2, 1024).cuda(gpu_device)
    fusion_features = torch.zeros(2, 1792).cuda(gpu_device)
    for test_img, test_label in data_loader:
        test_img = test_img.cuda(gpu_device)
        test_label = test_label.cuda(gpu_device)
        with torch.no_grad():
            if out_mode == 'triplet':
                vit_pre_y, fab_pre_y, fusion_pre_y = model(test_img)
            elif out_mode == 'single':
                test_pre_y = model(test_img)

            vit_features = torch.cat((vit_features, vit_pre_y))
            fab_features = torch.cat((fab_features, fab_pre_y))
            fusion_features = torch.cat((fusion_features, fusion_pre_y))

    vit_features = vit_features[2:]
    fab_features = fab_features[2:]
    fusion_features = fusion_features[2:]
    return vit_features, fab_features, fusion_features
