import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention
from Utils.Setup_Seed import setup_seed
from Read_Feats_Datasets import Read_Feats_Datasets
from Training_Testing_for_SOTA.training_testing_for_abmil import training_for_abmil, testing_for_abmil
from torch.utils.data import DataLoader


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.4
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x



class AttentionMIL(nn.Module):
    def __init__(self, in_features, num_classes=2, L=768, D=192, n_leision = 5, attn_mode="gated", dropout_node=0.0):
        super().__init__()
        self.L = L
        self.D = D
        self.K = 1

        self.attn_mode = attn_mode


        self.MLP = nn.Sequential(
            nn.Linear(in_features, self.L),
            nn.ReLU(),
        )

        if attn_mode == 'gated':
            self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
            )

            self.attention_U = nn.Sequential(
                nn.Linear(self.L, self.D),
                nn.Sigmoid()
            )

            self.attention_weights = nn.Linear(self.D, self.K)
        else:
            self.attention = nn.Sequential(
                nn.Linear(self.L, self.D),
                nn.Tanh(),
                nn.Linear(self.D, self.K)
            )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_node) if dropout_node>0.0 else nn.Identity(),
            nn.Linear(self.L*self.K, num_classes)
        )

    def forward(self, x):
        H = self.MLP(x)  # NxL

        if self.attn_mode == 'gated':
            A_V = self.attention_V(H)  # NxD
            A_U = self.attention_U(H)  # NxD
            A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N
        else:
            A = self.attention(H)  # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL
        logits = self.classifier(M)

        return logits, A ,H


if __name__ == "__main__":
    random_seed = 1
    batch_size = 2
    num_classes = 3
    epoch = 100
    gpu_device = 0
    mode_stats = 'testing'
    weight_path = '/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/AB_MIL_Linear.pth'
    testing_weights_path = '/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/AB_MIL_Linear.pth'
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

    abmill_net = AttentionMIL(768, attn_mode='linear', dropout_node=0.1, num_classes=num_classes).cuda(gpu_device)
    #test_x = torch.randn((85, 768))
    #pre_y, _, _ = abmil_net(test_x)

    #print(abmil_net)

    #print(pre_y.shape)

    if mode_stats == 'training':
        training_for_abmil(mil_net=abmill_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                           proba_mode=False, lr_fn='vit', epoch=epoch, gpu_device=gpu_device, onecycle_mr=1e-2, current_lr=None,
                           data_parallel=False, weight_path=weight_path, proba_value=0.85, class_num = num_classes,
                           bags_stat = True)
    elif mode_stats == 'testing':
        abmill_weight = torch.load(testing_weights_path, map_location='cuda:0')
        abmill_net.load_state_dict(abmill_weight, strict=True)
        testing_for_abmil(test_model = abmill_net, train_loader=train_loader, val_loader=val_loader,
                           proba_value = None, test_loader=test_loader, gpu_device=gpu_device,
                           out_mode = None, proba_mode=False, class_num=num_classes,
                           roc_save_path = None, bags_stat=True, bag_relations_path = None)







































