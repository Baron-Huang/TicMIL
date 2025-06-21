import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.utils import initialize_weights
import numpy as np
import os
from topk import SmoothTop1SVM
from Utils.Setup_Seed import setup_seed
from Read_Feats_Datasets import Read_Feats_Datasets
from torch.utils.data import DataLoader
from Training_Testing_for_SOTA.training_testing_for_clam import training_for_clam, testing_for_clam

"""
Lu, M.Y., Williamson, D.F., Chen, T.Y., Chen, R.J., Barbieri, M. and Mahmood, F., 2021. 
Data-efficient and weakly supervised computational pathology on whole-slide images. 
Nature biomedical engineering, 5(6), pp.555-570.
"""

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""


class CLAM_SB(nn.Module):
    def __init__(self, feat_dim=1024, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 subtyping=False, instance_eval=True):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [feat_dim, 256, 128], "big": [feat_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.instance_eval = instance_eval

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device, dtype=torch.uint8).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device, dtype=torch.uint8).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        # print(f"inst_eval top k: {A.shape}")
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)

        instance_loss_fn = SmoothTop1SVM(n_classes=self.n_classes).cuda(device)
        instance_loss = instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)

        instance_loss_fn = SmoothTop1SVM(n_classes=self.n_classes).cuda(device)
        instance_loss = instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h,label):
        h, label = h, label
        h = h.squeeze()
        # print(f"raw h: {h.shape}")
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        if self.instance_eval:
            # an ugly walkaround to deal with small bag size
            original_k_sample = self.k_sample
            if A.shape[1] < 2 * self.k_sample:
                self.k_sample = (A.shape[1] - 1) // 2

            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

            self.k_sample = original_k_sample

        M = torch.mm(A, h)
        logits = self.classifiers(M)

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if self.instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}

        return logits, Y_prob, Y_hat, results_dict


class CLAM_MB(CLAM_SB):
    def __init__(self, feat_dim=1024, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                  subtyping=False, instance_eval=True):
        nn.Module.__init__(self)
        self.size_dict = {"small": [feat_dim, 256, 128], "big": [feat_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in
                           range(n_classes)]  # use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.instance_eval = instance_eval

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, h, label):
        # h, label = batch[0], batch[1]
        device = h.device
        h = h.squeeze()
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        if self.instance_eval:
            # an ugly walkaround to deal with small bag size
            original_k_sample = self.k_sample
            if A.shape[1] < 2 * self.k_sample:
                self.k_sample = (A.shape[1] - 1) // 2

            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

            self.k_sample = original_k_sample

        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if self.instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}

        return logits, Y_prob, Y_hat, results_dict



if __name__ == "__main__":
    random_seed = 1
    batch_size = 2
    num_classes = 3
    feat_dim = 768
    epoch = 100
    size_arg = 'small'
    gpu_device = 0
    mode_stats = 'training'
    weight_path = '/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/CLAM_MB(small).pth'
    testing_weights_path = r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/CLAM_MB(small).pth'
    setup_seed(random_seed)
    train_dataset = Read_Feats_Datasets(data_size=[1025, 768],
                                        data_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Train/Feats',
                                        label_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Train/Labels')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    test_dataset = Read_Feats_Datasets(data_size=[1025, 768],
                                       data_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test/Feats',
                                       label_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test/Labels')
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    val_dataset = Read_Feats_Datasets(data_size=[1025, 768],
                                      data_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test/Feats',
                                      label_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test/Labels')
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)


    clammb_net = CLAM_MB(instance_eval=False, feat_dim=feat_dim, n_classes=num_classes, size_arg=size_arg).cuda(gpu_device)

    if mode_stats == 'training':
        training_for_clam(mil_net=clammb_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                          proba_mode=False, lr_fn='vit', epoch=epoch, gpu_device=gpu_device, onecycle_mr=1e-2, current_lr=None,
                          data_parallel=False, weight_path=weight_path, proba_value=0.85, class_num = num_classes,
                          bags_stat = True)
    elif mode_stats == 'testing':
        clammb_weight = torch.load(testing_weights_path, map_location='cuda:0')
        clammb_net.load_state_dict(clammb_weight, strict=True)
        testing_for_clam(test_model = clammb_net, train_loader=train_loader, val_loader=val_loader,
                           proba_value = None, test_loader=test_loader, gpu_device=gpu_device,
                           out_mode = None, proba_mode=False, class_num=num_classes,
                           roc_save_path = None, bags_stat=True, bag_relations_path = None)


