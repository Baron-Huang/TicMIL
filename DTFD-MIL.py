import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropout import Pdropout
from Utils.Setup_Seed import setup_seed
from torch.utils.data import DataLoader
from Training_Testing_for_SOTA.training_testing_for_dtfdmil import training_for_dtfdmil, testing_for_dtfdmil
from Read_Feats_Datasets import Read_Feats_Datasets

def order_F_to_C(n):
    idx = np.arange(0, n ** 2)
    idx = idx.reshape(n, n, order="F")
    idx = idx.reshape(n ** 2, order="C")
    idx = list(idx)
    return idx


def init_dct(n, m):
    """ Compute the Overcomplete Discrete Cosinus Transform. """
    oc_dictionary = np.zeros((n, m))
    for k in range(m):
        V = np.cos(np.arange(0, n) * k * np.pi / m)
        if k > 0:
            V = V - np.mean(V)
        oc_dictionary[:, k] = V / np.linalg.norm(V)
    oc_dictionary = np.kron(oc_dictionary, oc_dictionary)
    oc_dictionary = oc_dictionary.dot(np.diag(1 / np.sqrt(np.sum(oc_dictionary ** 2, axis=0))))

    idx = np.arange(0, n ** 2)
    idx = idx.reshape(n, n, order="F")
    idx = idx.reshape(n ** 2, order="C")
    oc_dictionary = oc_dictionary[idx, :]
    oc_dictionary = torch.from_numpy(oc_dictionary).float()
    return oc_dictionary


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                #Pdropout(0.2),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                #Pdropout(0.2)
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=529, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.resBlocks = []
        self.numRes = numLayer_Res

        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

        # self.fc = nn.Sequential(
        #     nn.Linear(n_channels, n_channels//2, bias=False),
        #     nn.ReLU(inplace=True),
        #     Pdropout(0.2),
        #     nn.Linear(n_channels//2, n_channels//2, bias=False),
        #     nn.ReLU(inplace=True),
        #     Pdropout(0.2),
        #     nn.Linear(n_channels//2, m_dim, bias=False),
        #     nn.ReLU(inplace=True),
        #     Pdropout(0.2),
        # )


    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x


class Attention2(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention2, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x, isNorm=True):
        ## x: N x L
        A = self.attention(x)  ## N x K
        A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
        return A  ### K x N


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=4, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)

    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        pred = self.classifier(afeat) ## K x num_cls
        return pred


if __name__ == "__main__":
    random_seed = 1
    batch_size = 2
    num_classes = 3
    epoch = 100
    gpu_device = 0
    mode_stats = 'training'
    weight_path = '/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/DTFD_MIL.pth'
    testing_weights_path = r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/DTFD_MIL.pth'
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



    dtfdmil_net = Attention_with_Classifier(L=768, num_cls=num_classes)
    #test_x = torch.randn((85, 768))
    #pre_y, _, _ = abmil_net(test_x)

    #print(abmil_net)

    #print(pre_y.shape)

    dtfdmil_net = dtfdmil_net.cuda(gpu_device)

    if mode_stats == 'training':
        training_for_dtfdmil(mil_net=dtfdmil_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                           proba_mode=False, lr_fn='vit', epoch=epoch, gpu_device=gpu_device, onecycle_mr=1e-2, current_lr=None,
                           data_parallel=False, weight_path=weight_path, proba_value=0.85, class_num = num_classes,
                           bags_stat = True)
    elif mode_stats == 'testing':
        dtfdmil_weight = torch.load(testing_weights_path, map_location='cuda:0')
        dtfdmil_net.load_state_dict(dtfdmil_weight, strict=True)
        testing_for_dtfdmil(test_model = dtfdmil_net, train_loader=train_loader, val_loader=val_loader,
                           proba_value = None, test_loader=test_loader, gpu_device=gpu_device,
                           out_mode = None, proba_mode=False, class_num=num_classes,
                           roc_save_path = None, bags_stat=True, bag_relations_path = None)

