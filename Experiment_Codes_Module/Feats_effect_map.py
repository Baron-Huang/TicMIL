from cProfile import label

import torch
import argparse
import numpy as np
import pandas as pd

from Training_Testing_for_SOTA.training_testing_for_frmil import testing_for_frmil


class Feats_Effect_Map_for_TicMIL:
    def __init__(self, save_path = r'/data/HP_Projects/TicMIL/Results/Feats_effect/PacMIL.csv',
                 test_read_path = r'/data/HP_Projects/TicMIL/Datasets/Cervix/New_Bags/Test'):
        super(Feats_Effect_Map_for_TicMIL, self)
        self.save_path = save_path
        self.test_read_path = test_read_path
        self.feat_list = []

    def create_swintt(self):
        from Models.SwinT_models.models.swin_transformer import SwinTransformer
        model_paras = argparse.ArgumentParser(description='SwinT Hyparameters')
        model_paras.add_argument('--random_seed', type=int, default=1)
        model_paras.add_argument('--gpu_device', type=int, default=0)
        model_paras.add_argument('--class_num', type=int, default=3)
        model_paras.add_argument('--num_workers', type=int, default=16)
        model_paras.add_argument('--bags_stat', type=bool, default=True)
        model_paras.add_argument('--bags_len', type=int, default=1025)
        model_paras.add_argument('--img_size', type=list, default=[96, 96])

        model_args = model_paras.parse_args()

        swinT_base = SwinTransformer(img_size=model_args.img_size[0], patch_size=4, in_chans=3,
                                     num_classes=model_args.class_num,
                                     embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                                     window_size=3, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                     norm_layer=torch.nn.LayerNorm, ape=False, patch_norm=True,
                                     use_checkpoint=False, fused_window_process=False)

        return swinT_base

    def create_loader(self):
        from Utils.Read_MIL_Datasets import Read_MIL_Datasets
        from torch.utils.data import DataLoader
        data_paras = argparse.ArgumentParser(description='Loader Hyparameters')
        data_paras.add_argument('--bags_len', type=int, default=1025)
        data_paras.add_argument('--batch_size', type=int, default=1)
        data_paras.add_argument('--img_size', type=list, default=[96, 96])
        data_paras.add_argument('--num_workers', type=int, default=16)

        data_args = data_paras.parse_args()

        test_data = Read_MIL_Datasets(read_path=self.test_read_path, img_size=data_args.img_size,
                                      bags_len=data_args.bags_len)
        test_loader = DataLoader(test_data, batch_size=data_args.batch_size, shuffle=False,
                                 num_workers=data_args.num_workers)
        return test_loader


    def hook_fn_for_ticmil(self, model, feats_in, feats_out):
        print(feats_in[0].shape)
        self.feat_list.append(feats_in)


    def fit_transform(self):
        from torchsummary import summary
        from Utils.Setup_Seed import setup_seed
        from Models.TicMIL_model_modules import TicMIL_Parallel_Feature, TicMIL_Parallel_Head
        from Utils.training_testing_for_TicMIL import testing_for_ticmil_parallel
        from torchvision.transforms.functional import to_pil_image
        from torch.nn.parallel import DataParallel
        ticmil_paras = argparse.ArgumentParser(description='TicMIL Hyparameters')
        ticmil_paras.add_argument('--random_seed', type=int, default=1)
        ticmil_paras.add_argument('--gpu_device', type=int, default=0)
        ticmil_paras.add_argument('--class_num', type=int, default=3)
        ticmil_paras.add_argument('--batch_size', type=int, default=1)
        ticmil_paras.add_argument('--data_parallel', type=bool, default=True)
        ticmil_paras.add_argument('--num_workers', type=int, default=16)
        ticmil_paras.add_argument('--bags_stat', type=bool, default=True)
        ticmil_paras.add_argument('--bags_len', type=int, default=1025)
        ticmil_paras.add_argument('--roc_save_path', type=str,
                                  default=r'/data/HP_Projects/TicMIL/Results/ROC/TicMIL.csv')
        ticmil_paras.add_argument('--confusion_save_path', type=str,
                                  default=r'/data/HP_Projects/TicMIL/Results/Confusion/TicMIL.csv')
        ticmil_paras.add_argument('--run_mode', type=str, default='test')  ## train or test
        ticmil_paras.add_argument('--test_weights_feature', type=str,
                                  default=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/TicMIL_Feature.pth')
        ticmil_paras.add_argument('--test_weights_head', type=str,
                                  default=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/TicMIL_Head.pth')

        ticmil_args = ticmil_paras.parse_args()
        setup_seed(1)

        swinT_base = self.create_swintt()
        test_loader = self.create_loader()

        with torch.no_grad():
            print('########################## PacMIL_summary #########################')
            summary(swinT_base, (3, 96, 96), device='cpu')
            print('\n', '########################## PacMIL #########################')
            print(swinT_base, '\n')

        ticmil_feature = TicMIL_Parallel_Feature(base_model=swinT_base)
        ticmil_head = TicMIL_Parallel_Head(base_model=swinT_base, class_num=ticmil_args.class_num,
                                           model_stats=ticmil_args.run_mode,
                                           batch_size=ticmil_args.batch_size, bags_len=ticmil_args.bags_len)
        ticmil_feature = ticmil_feature.cuda()
        ticmil_head = ticmil_head.cuda(ticmil_args.gpu_device)
        ticmil_feature = DataParallel(ticmil_feature, device_ids=[0])

        head_weight = torch.load(ticmil_args.test_weights_head, map_location='cuda:0')
        feature_weight = torch.load(ticmil_args.test_weights_feature, map_location='cuda:0')
        ticmil_feature.load_state_dict(feature_weight, strict=True)
        ticmil_head.load_state_dict(head_weight, strict=True)

        # print(ticmil_feature.layers[3].blocks[1].mlp.fc2.weight)
        testing_for_ticmil_parallel(mil_feature=ticmil_feature, mil_head=ticmil_head, train_loader=test_loader,
                                    data_parallel=ticmil_args.data_parallel, roc_save_path=ticmil_args.roc_save_path,
                                    batch_size=ticmil_args.batch_size,
                                    confusion_save_path=ticmil_args.confusion_save_path,
                                    bags_len=ticmil_args.bags_len, val_loader=test_loader, test_loader=test_loader)


        ############### hook
        ticmil_head.head.register_forward_hook(self.hook_fn_for_ticmil)

        label_sum = np.zeros((1, 1))
        with torch.no_grad():
            for data_i, label_i in test_loader:
                label_i = label_i.reshape(label_i.shape[0], 1)
                label_sum = np.concatenate((label_sum, label_i.detach().cpu().numpy()))
                for i in data_i:
                    y = ticmil_feature(i.cuda(ticmil_args.gpu_device))
                    y = ticmil_head(y)
                    # print(y.shape)


        print(self.feat_list[0][0].shape)
        feat_numpy = np.zeros((1, 768))

        for i in range(len(self.feat_list)):
            feat_numpy = np.concatenate((feat_numpy, self.feat_list[i][0].detach().cpu().numpy()))

        feat_numpy = feat_numpy[1:]
        label_sum = label_sum[1:]

        print(feat_numpy.shape)
        print(label_sum.shape)
        print(ticmil_head.head.weight.shape)

        class_wegts_mat = ticmil_head.head.weight[label_sum, :].reshape(feat_numpy.shape[0], feat_numpy.shape[1])
        class_wegts_mat = class_wegts_mat.detach().cpu().numpy()
        print(class_wegts_mat.shape)

        feat_effect_mat = feat_numpy * class_wegts_mat

        print(feat_effect_mat.shape)

        fe_mean = np.mean(feat_effect_mat, axis=0)
        fe_std = np.std(feat_effect_mat, axis=0)

        print(fe_mean.shape)
        print(fe_std.shape)

        x = np.argsort(-fe_mean)
        print(x)
        max_ord = x[0:100]

        fe_dict = {}
        fe_dict['no'] = max_ord
        fe_dict['mean'] = fe_mean[max_ord]
        fe_dict['error'] = fe_std[max_ord]
        fe_pd = pd.DataFrame(fe_dict)
        fe_pd.to_csv(self.save_path)


class Feats_Effect_Map_for_ILRA(Feats_Effect_Map_for_TicMIL):
    def __init__(self, save_path, test_read_path, test_wegts_path = None,
                 feat_len = 768, gpu_device = 0, ins_len = 1025):
        super(Feats_Effect_Map_for_ILRA, self).__init__(save_path, test_read_path)
        self.test_wegts_path = test_wegts_path
        self.testing_model = None
        self.register_layer = None
        self.test_loader = None
        self.feat_len = feat_len
        self.gpu_device = gpu_device
        self.ins_len = ins_len


    def creating_model_layers(self):
        from ILRA_MIL import ILRA
        from Training_Testing_for_SOTA.training_testing_for_ilramil import testing_for_ilramil

        self.testing_model = ILRA(feat_dim=768, n_classes=3, hidden_feat=256, num_heads=8, topk=1)
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(self.test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        self.register_layer = self.testing_model.classifier

        testing_for_ilramil(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)


    def creating_loader(self):
        from torch.utils.data import DataLoader
        from Read_Feats_Datasets import Read_Feats_Datasets
        two_steps_paras = argparse.ArgumentParser(description='Two Steps Hyparameters')
        two_steps_paras.add_argument('--batch_size', type=int, default=1)
        two_steps_paras.add_argument('--shuffle_stats', type=bool, default=False)
        two_steps_paras.add_argument('--num_workers', type=int, default=16)

        two_steps_args = two_steps_paras.parse_args()
        test_dataset = Read_Feats_Datasets(data_size=[1025, 768],
                                           data_path=self.test_read_path + r'/Feats',
                                           label_path=self.test_read_path + r'/Labels')
        test_loader = DataLoader(dataset=test_dataset, batch_size=two_steps_args.batch_size,
                                 shuffle=two_steps_args.shuffle_stats, num_workers=two_steps_args.num_workers)
        self.test_loader = test_loader


    def hook_fn(self, model, feats_in, feats_out):
        self.feat_list.append(feats_in)


    def fit_transform(self):
        from Utils.Setup_Seed import setup_seed

        setup_seed(1)

        ## 1-th step
        self.creating_loader()

        ## 2-th step
        self.creating_model_layers()

        ####
        self.register_layer.register_forward_hook(self.hook_fn)

        label_sum = np.zeros((1, 1))
        with torch.no_grad():
            for data_i, label_i in self.test_loader:
                y = self.testing_model(data_i.cuda(self.gpu_device))
                pre_label = torch.argmax(y[0], dim=1, keepdim=True)
                label_sum = np.concatenate((label_sum, pre_label.detach().cpu().numpy()))

        print(self.feat_list[0][0].shape)
        feat_numpy = np.zeros((1, 256))

        for i in range(len(self.feat_list)):
            kkk = self.feat_list[i][0].detach().cpu().numpy()
            feat_numpy = np.concatenate((feat_numpy, kkk.reshape(((1, 256)))))

        feat_numpy = feat_numpy[1:]
        label_sum = label_sum[1:]

        print(feat_numpy.shape)
        print(label_sum.shape)
        print(self.register_layer.weight.shape)

        class_wegts_mat = self.register_layer.weight[label_sum, :].reshape(feat_numpy.shape[0], feat_numpy.shape[1])
        class_wegts_mat = class_wegts_mat.detach().cpu().numpy()
        print(class_wegts_mat.shape)

        feat_effect_mat = feat_numpy * class_wegts_mat

        print(feat_effect_mat.shape)

        fe_mean = np.mean(feat_effect_mat, axis=0)
        fe_std = np.std(feat_effect_mat, axis=0)

        print(fe_mean.shape)
        print(fe_std.shape)

        x = np.argsort(-fe_mean)
        print(x)
        max_ord = x[0:100]

        fe_dict = {}
        fe_dict['no'] = max_ord
        fe_dict['mean'] = fe_mean[max_ord]
        fe_dict['error'] = fe_std[max_ord]
        fe_pd = pd.DataFrame(fe_dict)
        fe_pd.to_csv(self.save_path)

class Feats_effect_Map_for_ABMIL_Gated(Feats_Effect_Map_for_ILRA):
    def __init__(self, save_path, test_read_path, test_wegts_path = None,
                 feat_len = 768, gpu_device = 0):
        super(Feats_effect_Map_for_ABMIL_Gated, self).__init__(save_path, test_read_path, test_wegts_path,
                 feat_len, gpu_device)


    def creating_model_layers(self):
        from AB_MIL_Gated import AttentionMIL
        from Training_Testing_for_SOTA.training_testing_for_abmil import testing_for_abmil

        self.testing_model = AttentionMIL(768, attn_mode='gated', dropout_node=0.1, num_classes=3)
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(self.test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        self.register_layer = self.testing_model.classifier[1]

        testing_for_abmil(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                          proba_value=None, test_loader=self.test_loader, gpu_device=0,
                          out_mode=None, proba_mode=False, class_num=3,
                          roc_save_path=None, bags_stat=True, bag_relations_path=None)


    def fit_transform(self):
        from Utils.Setup_Seed import setup_seed

        setup_seed(1)

        ## 1-th step
        self.creating_loader()

        ## 2-th step
        self.creating_model_layers()

        ####
        self.register_layer.register_forward_hook(self.hook_fn)

        label_sum = np.zeros((1, 1))
        with torch.no_grad():
            for data_i, label_i in self.test_loader:
                data_i = data_i.reshape((self.ins_len, self.feat_len))
                y = self.testing_model(data_i.cuda(self.gpu_device))
                pre_label = torch.argmax(y[0], dim=1, keepdim=True)
                label_sum = np.concatenate((label_sum, pre_label.detach().cpu().numpy()))

        print(self.feat_list[0][0].shape)
        feat_numpy = np.zeros((1, self.feat_len))

        for i in range(len(self.feat_list)):
            kkk = self.feat_list[i][0].detach().cpu().numpy()
            feat_numpy = np.concatenate((feat_numpy, kkk.reshape(((1, self.feat_len)))))

        feat_numpy = feat_numpy[1:]
        label_sum = label_sum[1:]

        print(feat_numpy.shape)
        print(label_sum.shape)
        print(self.register_layer.weight.shape)

        class_wegts_mat = self.register_layer.weight[label_sum, :].reshape(feat_numpy.shape[0], feat_numpy.shape[1])
        class_wegts_mat = class_wegts_mat.detach().cpu().numpy()
        print(class_wegts_mat.shape)

        feat_effect_mat = feat_numpy * class_wegts_mat

        print(feat_effect_mat.shape)

        fe_mean = np.mean(feat_effect_mat, axis=0)
        fe_std = np.std(feat_effect_mat, axis=0)

        print(fe_mean.shape)
        print(fe_std.shape)

        x = np.argsort(-fe_mean)
        print(x)
        max_ord = x[0:100]

        fe_dict = {}
        fe_dict['no'] = max_ord
        fe_dict['mean'] = fe_mean[max_ord]
        fe_dict['error'] = fe_std[max_ord]
        fe_pd = pd.DataFrame(fe_dict)
        fe_pd.to_csv(self.save_path)

class Feats_effect_Map_for_ABMIL_Linear(Feats_effect_Map_for_ABMIL_Gated):
    def __init__(self, save_path, test_read_path, test_wegts_path = None,
                 feat_len = 768, gpu_device = 0):
        super(Feats_effect_Map_for_ABMIL_Linear, self).__init__(save_path, test_read_path, test_wegts_path,
                 feat_len, gpu_device)

    def creating_model_layers(self):
        from AB_MIL_Gated import AttentionMIL
        from Training_Testing_for_SOTA.training_testing_for_abmil import testing_for_abmil

        self.testing_model = AttentionMIL(768, attn_mode='linear', dropout_node=0.1, num_classes=3)
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(self.test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        self.register_layer = self.testing_model.classifier[1]

        testing_for_abmil(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                          proba_value=None, test_loader=self.test_loader, gpu_device=0,
                          out_mode=None, proba_mode=False, class_num=3,
                          roc_save_path=None, bags_stat=True, bag_relations_path=None)



class Feats_effect_Map_for_CLAM_SB_small(Feats_Effect_Map_for_ILRA):
    def __init__(self, save_path, test_read_path, test_wegts_path = None,
                 feat_len = 768, gpu_device = 0, out_feat_len = 256):
        super(Feats_effect_Map_for_CLAM_SB_small, self).__init__(save_path, test_read_path, test_wegts_path,
                 feat_len, gpu_device)
        self.out_feat_len = out_feat_len

    def creating_model_layers(self):
        from CLAM_SB import CLAM_SB
        from Training_Testing_for_SOTA.training_testing_for_clam import testing_for_clam

        self.testing_model = CLAM_SB(instance_eval=False, feat_dim=768, n_classes=3, size_arg='small')
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(self.test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        self.register_layer = self.testing_model.classifiers

        testing_for_clam(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                          proba_value=None, test_loader=self.test_loader, gpu_device=0,
                          out_mode=None, proba_mode=False, class_num=3,
                          roc_save_path=None, bags_stat=True, bag_relations_path=None)

    def fit_transform(self):
        from Utils.Setup_Seed import setup_seed

        setup_seed(1)

        ## 1-th step
        self.creating_loader()

        ## 2-th step
        self.creating_model_layers()

        ####
        self.register_layer.register_forward_hook(self.hook_fn)

        label_sum = np.zeros((1, 1))
        with torch.no_grad():
            for data_i, label_i in self.test_loader:
                data_i = data_i.reshape((self.ins_len, self.feat_len))
                y = self.testing_model(data_i.cuda(self.gpu_device), torch.zeros(1).cuda(self.gpu_device))
                pre_label = torch.argmax(y[0], dim=1, keepdim=True)
                label_sum = np.concatenate((label_sum, pre_label.detach().cpu().numpy()))

        print(self.feat_list[0][0].shape)
        feat_numpy = np.zeros((1, self.out_feat_len))

        for i in range(len(self.feat_list)):
            kkk = self.feat_list[i][0].detach().cpu().numpy()
            feat_numpy = np.concatenate((feat_numpy, kkk.reshape(((1, self.out_feat_len)))))

        feat_numpy = feat_numpy[1:]
        label_sum = label_sum[1:]

        print(feat_numpy.shape)
        print(label_sum.shape)
        print(self.register_layer.weight.shape)

        class_wegts_mat = self.register_layer.weight[label_sum, :].reshape(feat_numpy.shape[0], feat_numpy.shape[1])
        class_wegts_mat = class_wegts_mat.detach().cpu().numpy()
        print(class_wegts_mat.shape)

        feat_effect_mat = feat_numpy * class_wegts_mat

        print(feat_effect_mat.shape)

        fe_mean = np.mean(feat_effect_mat, axis=0)
        fe_std = np.std(feat_effect_mat, axis=0)

        print(fe_mean.shape)
        print(fe_std.shape)

        x = np.argsort(-fe_mean)
        print(x)
        max_ord = x[0:100]

        fe_dict = {}
        fe_dict['no'] = max_ord
        fe_dict['mean'] = fe_mean[max_ord]
        fe_dict['error'] = fe_std[max_ord]
        fe_pd = pd.DataFrame(fe_dict)
        fe_pd.to_csv(self.save_path)

class Feats_effect_Map_for_CLAM_SB_big(Feats_effect_Map_for_CLAM_SB_small):
    def __init__(self, save_path, test_read_path, test_wegts_path = None,
                 feat_len = 768, gpu_device = 0, out_feat_len = 512):
        super(Feats_effect_Map_for_CLAM_SB_big, self).__init__(save_path, test_read_path, test_wegts_path,
                 feat_len, gpu_device, out_feat_len)


    def creating_model_layers(self):
        from CLAM_SB import CLAM_SB
        from Training_Testing_for_SOTA.training_testing_for_clam import testing_for_clam

        self.testing_model = CLAM_SB(instance_eval=False, feat_dim=768, n_classes=3, size_arg='big')
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(self.test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        self.register_layer = self.testing_model.classifiers

        testing_for_clam(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                          proba_value=None, test_loader=self.test_loader, gpu_device=0,
                          out_mode=None, proba_mode=False, class_num=3,
                          roc_save_path=None, bags_stat=True, bag_relations_path=None)


#### ILRA 2，3纬度均可,对out变化不行，AB_MIB只能输入2D，而CLAM是双输入，而DGRMIL单个输入，2，3D均可，适应变化的out
class Feats_effect_Map_for_DGRMIL(Feats_Effect_Map_for_ILRA):
    def __init__(self, save_path, test_read_path, test_wegts_path = None,
                 feat_len = 768, gpu_device = 0, out_feat_len = 512):
        super(Feats_effect_Map_for_DGRMIL, self).__init__(save_path, test_read_path, test_wegts_path,
                 feat_len, gpu_device)

        self.out_feat_len = out_feat_len

    def creating_model_layers(self):
        from DGR_MIL import DGRMIL
        from Training_Testing_for_SOTA.training_testing_for_dgrmil import testing_for_dgrmil

        self.testing_model = DGRMIL(768, num_classes=3, attn_mode='linear', dropout_node=0.1)
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(self.test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        self.register_layer = self.testing_model.classifier[0]

        testing_for_dgrmil(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                          proba_value=None, test_loader=self.test_loader, gpu_device=0,
                          out_mode=None, proba_mode=False, class_num=3,
                          roc_save_path=None, bags_stat=True, bag_relations_path=None)


    def fit_transform(self):
        from Utils.Setup_Seed import setup_seed

        setup_seed(1)

        ## 1-th step
        self.creating_loader()

        ## 2-th step
        self.creating_model_layers()

        ####
        self.register_layer.register_forward_hook(self.hook_fn)

        label_sum = np.zeros((1, 1))
        with torch.no_grad():
            for data_i, label_i in self.test_loader:
                y = self.testing_model(data_i.cuda(self.gpu_device))
                pre_label = torch.argmax(y[0], dim=1, keepdim=True)
                label_sum = np.concatenate((label_sum, pre_label.detach().cpu().numpy()))

        print(self.feat_list[0][0].shape)
        feat_numpy = np.zeros((1, self.out_feat_len))

        for i in range(len(self.feat_list)):
            kkk = self.feat_list[i][0].detach().cpu().numpy()
            feat_numpy = np.concatenate((feat_numpy, kkk.reshape(((1, self.out_feat_len)))))

        feat_numpy = feat_numpy[1:]
        label_sum = label_sum[1:]

        print(feat_numpy.shape)
        print(label_sum.shape)
        print(self.register_layer.weight.shape)

        class_wegts_mat = self.register_layer.weight[label_sum, :].reshape(feat_numpy.shape[0], feat_numpy.shape[1])
        class_wegts_mat = class_wegts_mat.detach().cpu().numpy()
        print(class_wegts_mat.shape)

        feat_effect_mat = feat_numpy * class_wegts_mat

        print(feat_effect_mat.shape)

        fe_mean = np.mean(feat_effect_mat, axis=0)
        fe_std = np.std(feat_effect_mat, axis=0)

        print(fe_mean.shape)
        print(fe_std.shape)

        x = np.argsort(-fe_mean)
        print(x)
        max_ord = x[0:100]

        fe_dict = {}
        fe_dict['no'] = max_ord
        fe_dict['mean'] = fe_mean[max_ord]
        fe_dict['error'] = fe_std[max_ord]
        fe_pd = pd.DataFrame(fe_dict)
        fe_pd.to_csv(self.save_path)




#### ILRA 2，3纬度均可,对out变化不行，AB_MIB只能输入2D，而CLAM是双输入，而DGRMIL单个输入，2，3D均可，适应变化的out,
####DTFD_MIL AB_MIB只能输入2D,且适应out的变化, 而且只有一个输出
class Feats_effect_Map_for_DTFD_MIL(Feats_effect_Map_for_ABMIL_Gated):
    def __init__(self, save_path, test_read_path, test_wegts_path = None,
                 feat_len = 768, gpu_device = 0, out_feat_len = 768):
        super(Feats_effect_Map_for_DTFD_MIL, self).__init__(save_path, test_read_path, test_wegts_path,
                 feat_len, gpu_device)

        self.out_feat_len = out_feat_len


    def creating_model_layers(self):
        from DTFD_MIL import Attention_with_Classifier
        from Training_Testing_for_SOTA.training_testing_for_dtfdmil import testing_for_dtfdmil

        self.testing_model = Attention_with_Classifier(L=768, num_cls=3)
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(self.test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        ### need to revise
        self.register_layer = self.testing_model.classifier.fc

        testing_for_dtfdmil(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                          proba_value=None, test_loader=self.test_loader, gpu_device=0,
                          out_mode=None, proba_mode=False, class_num=3,
                          roc_save_path=None, bags_stat=True, bag_relations_path=None)

    def fit_transform(self):
        from Utils.Setup_Seed import setup_seed

        setup_seed(1)

        ## 1-th step
        self.creating_loader()

        ## 2-th step
        self.creating_model_layers()

        ####
        self.register_layer.register_forward_hook(self.hook_fn)

        label_sum = np.zeros((1, 1))
        with torch.no_grad():
            for data_i, label_i in self.test_loader:
                data_i = data_i.reshape((self.ins_len, self.feat_len))
                y = self.testing_model(data_i.cuda(self.gpu_device))
                pre_label = torch.argmax(y, dim=1, keepdim=True)
                label_sum = np.concatenate((label_sum, pre_label.detach().cpu().numpy()))

        print(self.feat_list[0][0].shape)
        feat_numpy = np.zeros((1, self.out_feat_len))

        for i in range(len(self.feat_list)):
            kkk = self.feat_list[i][0].detach().cpu().numpy()
            feat_numpy = np.concatenate((feat_numpy, kkk.reshape(((1, self.out_feat_len)))))

        feat_numpy = feat_numpy[1:]
        label_sum = label_sum[1:]

        print(feat_numpy.shape)
        print(label_sum.shape)
        print(self.register_layer.weight.shape)

        class_wegts_mat = self.register_layer.weight[label_sum, :].reshape(feat_numpy.shape[0], feat_numpy.shape[1])
        class_wegts_mat = class_wegts_mat.detach().cpu().numpy()
        print(class_wegts_mat.shape)

        feat_effect_mat = feat_numpy * class_wegts_mat

        print(feat_effect_mat.shape)

        fe_mean = np.mean(feat_effect_mat, axis=0)
        fe_std = np.std(feat_effect_mat, axis=0)

        print(fe_mean.shape)
        print(fe_std.shape)

        x = np.argsort(-fe_mean)
        print(x)
        max_ord = x[0:100]

        fe_dict = {}
        fe_dict['no'] = max_ord
        fe_dict['mean'] = fe_mean[max_ord]
        fe_dict['error'] = fe_std[max_ord]
        fe_pd = pd.DataFrame(fe_dict)
        fe_pd.to_csv(self.save_path)




#### ILRA 2，3纬度均可,对out变化不行，AB_MIB只能输入2D，而CLAM是双输入，而DGRMIL单个输入，2，3D均可，适应变化的out
class Feats_effect_Map_for_TransMIL(Feats_effect_Map_for_DGRMIL):
    def __init__(self, save_path, test_read_path, test_wegts_path = None,
                 feat_len = 768, gpu_device = 0, out_feat_len = 484):
        super(Feats_effect_Map_for_TransMIL, self).__init__(save_path, test_read_path, test_wegts_path,
                 feat_len, gpu_device, out_feat_len)

    def creating_model_layers(self):
        from TransMIL import TransMIL
        from Training_Testing_for_SOTA.training_testing_for_transmil import testing_for_transmil

        self.testing_model = TransMIL(768, n_classes=3, mDim=484)
        self.testing_model = self.testing_model.cuda(self.gpu_device)
        testing_model_wegts = torch.load(self.test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        self.register_layer = self.testing_model._fc2

        testing_for_transmil(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)



if __name__ == '__main__':
    #feats_transform = Feats_space_transform()
    #feats_transform.fit_transform_for_pacmil()

    #### 1. PacMIL

    fem_transform = Feats_Effect_Map_for_TicMIL(
                 save_path = r'/data/HP_Projects/TicMIL/Results/Feats_effect/TicMIL.csv',
                 test_read_path = r'/data/HP_Projects/TicMIL/Datasets/Cervix/New_Bags/Test')
    fem_transform.fit_transform()


    #### 2. ILRA_MIL
    '''
    fem_for_ilra_transform = Feats_Effect_Map_for_ILRA(
                 save_path = r'/data/HP_Projects/TicMIL/Results/Feats_effect/ILRA.csv',
                 test_read_path = r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
                 test_wegts_path=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/ILRA_MIL.pth')
    fem_for_ilra_transform.fit_transform()
    '''

    #### 3. AB_MIL_Gated
    '''
    fem_for_agmilg = Feats_effect_Map_for_ABMIL_Gated(
                    save_path=r'/data/HP_Projects/TicMIL/Results/Feats_effect/AB_MIL_Gated.csv',
                    test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
                    test_wegts_path=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/AB_MIL_Gated.pth')
    fem_for_agmilg.fit_transform()
    '''

    #### 4. AB_MIL_Linear
    '''
    fem_for_agmill = Feats_effect_Map_for_ABMIL_Linear(
        save_path=r'/data/HP_Projects/TicMIL/Results/Feats_effect/AB_MIL_Linear.csv',
        test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
        test_wegts_path=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/AB_MIL_Linear.pth')
    fem_for_agmill.fit_transform()
    '''

    #### 5. CLAM_SB_Small
    '''
    fem_for_clamsbs = Feats_effect_Map_for_CLAM_SB_small(
        save_path=r'/data/HP_Projects/TicMIL/Results/Feats_effect/CLAM_SB_small.csv',
        test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
        test_wegts_path=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/CLAM_SB(small).pth')
    fem_for_clamsbs.fit_transform()
    '''

    #### 6. CLAM_SB_Big
    '''
    fem_for_clamsbb = Feats_effect_Map_for_CLAM_SB_big(
        save_path=r'/data/HP_Projects/TicMIL/Results/Feats_effect/CLAM_SB_big.csv',
        test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
        test_wegts_path=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/CLAM_SB(Big).pth')
    fem_for_clamsbb.fit_transform()
    '''

    #### 7. DGR_MIL
    '''
    fem_for_dgrmil = Feats_effect_Map_for_DGRMIL(
        save_path=r'/data/HP_Projects/TicMIL/Results/Feats_effect/DGRMIL.csv',
        test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
        test_wegts_path=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/DGR_MIL.pth')
    fem_for_dgrmil.fit_transform()
    '''

    ### 8. DTFD_MIL
    '''
    fem_for_dtfdmil = Feats_effect_Map_for_DTFD_MIL(
        save_path=r'/data/HP_Projects/TicMIL/Results/Feats_effect/DTFD_MIL.csv',
        test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
        test_wegts_path=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/DTFD_MIL.pth')
    fem_for_dtfdmil.fit_transform()
    '''


    ### 9. TransMIL

    '''fem_for_transmil = Feats_effect_Map_for_TransMIL(
        save_path=r'/data/HP_Projects/TicMIL/Results/Feats_effect/TransMIL.csv',
        test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
        test_wegts_path=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/TransMIL.pth')
    fem_for_transmil.fit_transform()'''





