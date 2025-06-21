import torch
import argparse
import numpy as np
import pandas as pd

class Feats_space_transform_for_TicMIL:
    def __init__(self, D1_save_path = r'/data/HP_Projects/TicMIL/Results/Feats_space/1D/TicMIL_1D.csv',
                 D2_save_path = r'/data/HP_Projects/TicMIL/Results/Feats_space/2D/TicMIL_2D.csv',
                 D3_save_path = r'/data/HP_Projects/TicMIL/Results/Feats_space/3D/TicMIL_3D.csv',
                 test_read_path = r'/data/HP_Projects/TicMIL/Datasets/Cervix/New_Bags/Test'):
        super(Feats_space_transform_for_TicMIL, self)
        self.feat_list = []
        self.D1_save_path = D1_save_path
        self.D2_save_path = D2_save_path
        self.D3_save_path = D3_save_path
        self.test_read_path = test_read_path

    def create_swintt(self):
        from Models.SwinT_models.models.swin_transformer import SwinTransformer
        model_paras = argparse.ArgumentParser(description='SwinT Hyparameters')
        model_paras.add_argument('--random_seed', type=int, default=1)
        model_paras.add_argument('--gpu_device', type=int, default=0)
        model_paras.add_argument('--class_num', type=int, default=3)
        model_paras.add_argument('--num_workers', type=int, default=16)
        model_paras.add_argument('--bags_stat', type=bool, default=True)
        model_paras.add_argument('--bags_len', type=int, default=85)
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


    def visual_save_fn(self, feat_numpy=None, label_sum=None,
                  D1_save_path=None, D2_save_path=None, D3_save_path=None):

        from matplotlib import pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA

        trans_pca_3D = PCA(n_components=3)
        feats_3d = trans_pca_3D.fit_transform(feat_numpy)

        trans_pca_2D = PCA(n_components=2)
        feats_2d = trans_pca_2D.fit_transform(feat_numpy)

        trans_pca_1D = PCA(n_components=1)
        feats_1d = trans_pca_1D.fit_transform(feat_numpy)

        feats_pd_1d = pd.DataFrame(feats_1d)
        feats_pd_2d = pd.DataFrame(feats_2d)
        feats_pd_3d = pd.DataFrame(feats_3d)

        feats_pd_1d.to_csv(D1_save_path)
        feats_pd_2d.to_csv(D2_save_path)
        feats_pd_3d.to_csv(D3_save_path)

        print(feats_3d.shape)
        print(feats_2d.shape)
        print(feats_1d.shape)

        grade_1_order = np.where(label_sum == 0)
        grade_2_order = np.where(label_sum == 1)
        grade_3_order = np.where(label_sum == 2)

        plt.figure(1)
        plt.scatter(feats_2d[grade_1_order[0], 0], feats_2d[grade_1_order[0], 1], c='black')
        plt.scatter(feats_2d[grade_2_order[0], 0], feats_2d[grade_2_order[0], 1], c='blue')
        plt.scatter(feats_2d[grade_3_order[0], 0], feats_2d[grade_3_order[0], 1], c='red')

        fig_2 = plt.figure(2)
        ax = fig_2.add_subplot(111, projection='3d')

        ax.scatter(feats_3d[grade_1_order[0], 0], feats_2d[grade_1_order[0], 1],
                   feats_3d[grade_1_order[0], 2], c='black')
        ax.scatter(feats_3d[grade_2_order[0], 0], feats_2d[grade_2_order[0], 1],
                   feats_3d[grade_2_order[0], 2], c='blue')
        ax.scatter(feats_3d[grade_3_order[0], 0], feats_2d[grade_3_order[0], 1],
                   feats_3d[grade_3_order[0], 2], c='red')

        plt.figure(3)
        sns.kdeplot(feats_1d[grade_1_order[0], :], shade=True, color='red', label="Density")
        sns.kdeplot(feats_1d[grade_2_order[0], :], shade=True, color='blue', label="Density")
        sns.kdeplot(feats_1d[grade_3_order[0], :], shade=True, color='red', label="Density")

        plt.show()


    def fit_transform_for_ticmil(self):
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

        #print(ticmil_feature.layers[3].blocks[1].mlp.fc2.weight)
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
                    #print(y.shape)


        print(self.feat_list[0][0].shape)
        feat_numpy = np.zeros((1, 768))

        for i in range(len(self.feat_list)):
            feat_numpy = np.concatenate((feat_numpy, self.feat_list[i][0].detach().cpu().numpy()))

        feat_numpy = feat_numpy[1:]
        label_sum = label_sum[1:]

        print(feat_numpy.shape)
        print(label_sum.shape)

        self.visual_save_fn(feat_numpy=feat_numpy, label_sum=label_sum, D1_save_path=self.D1_save_path,
                            D2_save_path=self.D2_save_path, D3_save_path=self.D3_save_path)


#### 3-D tensor输入
class Feats_space_transform_for_ILRA(Feats_space_transform_for_TicMIL):
    def __init__(self, D1_save_path, D2_save_path, D3_save_path, test_read_path, test_wegts_path = None,
                 feat_len = 256):
        super(Feats_space_transform_for_ILRA, self).__init__(D1_save_path, D2_save_path,
                                                                 D3_save_path, test_read_path)
        self.testing_model = None
        self.register_layer = None
        self.test_wegts_path = test_wegts_path
        self.test_loader = None
        self.feat_len = feat_len


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


    def creating_loader_for_ts(self):
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
        self.creating_loader_for_ts()

        ## 2-th step
        self.creating_model_layers()

        ####
        self.register_layer.register_forward_hook(self.hook_fn)

        feats_sum = np.zeros((1, self.feat_len))
        labels_sum = np.zeros((1, 1))
        for data_i, label_i in self.test_loader:
            label_i = label_i.reshape(label_i.shape[0], 1)
            labels_sum = np.concatenate((labels_sum, label_i.detach().cpu().numpy()))
            with torch.no_grad():
                pre_y = self.testing_model(data_i.cuda(0))

        for i in range(len(self.feat_list)):
            feats_sum = np.concatenate((feats_sum,
                            self.feat_list[i][0].detach().cpu().numpy().reshape(1, self.feat_len)))

        self.visual_save_fn(feat_numpy=feats_sum, label_sum=labels_sum,
                  D1_save_path=self.D1_save_path, D2_save_path=self.D2_save_path, D3_save_path=self.D3_save_path)



### 2D matrix输入
class Feats_space_transform_for_AB_MIL_Gated(Feats_space_transform_for_ILRA):
    def __init__(self, D1_save_path, D2_save_path, D3_save_path, test_read_path, test_wegts_path = None,
                 feat_len = 768):
        super(Feats_space_transform_for_AB_MIL_Gated, self).__init__(D1_save_path, D2_save_path,
                                                                 D3_save_path, test_read_path, test_wegts_path, feat_len)

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
        self.creating_loader_for_ts()

        ## 2-th step
        self.creating_model_layers()

        ####
        self.register_layer.register_forward_hook(self.hook_fn)

        feats_sum = np.zeros((1, self.feat_len))
        labels_sum = np.zeros((1, 1))
        for data_i, label_i in self.test_loader:
            label_i = label_i.reshape(label_i.shape[0], 1)
            labels_sum = np.concatenate((labels_sum, label_i.detach().cpu().numpy()))
            with torch.no_grad():
                if data_i.dim() == 3:
                    data_i = data_i.reshape(data_i.shape[1], data_i.shape[2])
                pre_y = self.testing_model(data_i.cuda(0))

        for i in range(len(self.feat_list)):
            feats_sum = np.concatenate((feats_sum,
                            self.feat_list[i][0].detach().cpu().numpy().reshape(1, self.feat_len)))

        self.visual_save_fn(feat_numpy=feats_sum, label_sum=labels_sum,
                  D1_save_path=self.D1_save_path, D2_save_path=self.D2_save_path, D3_save_path=self.D3_save_path)


class Feats_space_transform_for_AB_MIL_Linear(Feats_space_transform_for_AB_MIL_Gated):
    def __init__(self, D1_save_path, D2_save_path, D3_save_path, test_read_path, test_wegts_path = None,
                 feat_len = 768):
        super(Feats_space_transform_for_AB_MIL_Linear, self).__init__(D1_save_path, D2_save_path,
                                                                 D3_save_path, test_read_path, test_wegts_path, feat_len)

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



class Feats_space_transform_for_CLAM_SB_small(Feats_space_transform_for_ILRA):
    def __init__(self, D1_save_path, D2_save_path, D3_save_path, test_read_path, test_wegts_path = None,
                 feat_len = 768, out_feat_len = 256):
        super(Feats_space_transform_for_CLAM_SB_small, self).__init__(D1_save_path, D2_save_path,
                                                                 D3_save_path, test_read_path, test_wegts_path, feat_len)
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
        self.creating_loader_for_ts()

        ## 2-th step
        self.creating_model_layers()

        ####
        self.register_layer.register_forward_hook(self.hook_fn)

        feats_sum = np.zeros((1, self.out_feat_len))
        labels_sum = np.zeros((1, 1))
        for data_i, label_i in self.test_loader:
            label_i = label_i.reshape(label_i.shape[0], 1)
            labels_sum = np.concatenate((labels_sum, label_i.detach().cpu().numpy()))
            with torch.no_grad():
                pre_y = self.testing_model(data_i.cuda(0), torch.zeros((1)).cuda(0))

        for i in range(len(self.feat_list)):
            feats_sum = np.concatenate((feats_sum,
                            self.feat_list[i][0].detach().cpu().numpy().reshape(1, self.out_feat_len)))

        self.visual_save_fn(feat_numpy=feats_sum, label_sum=labels_sum,
                  D1_save_path=self.D1_save_path, D2_save_path=self.D2_save_path, D3_save_path=self.D3_save_path)


class Feats_space_transform_for_CLAM_SB_big(Feats_space_transform_for_CLAM_SB_small):
    def __init__(self, D1_save_path, D2_save_path, D3_save_path, test_read_path, test_wegts_path = None,
                 feat_len = 768, out_feat_len = 256):
        super(Feats_space_transform_for_CLAM_SB_big, self).__init__(D1_save_path, D2_save_path,
                                                                 D3_save_path, test_read_path, test_wegts_path, feat_len,
                                                                    out_feat_len)

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

class Feats_space_transform_for_CLAM_MB_small(Feats_space_transform_for_CLAM_SB_small):
    def __init__(self, D1_save_path, D2_save_path, D3_save_path, test_read_path, test_wegts_path = None,
                 feat_len = 768, out_feat_len = 256):
        super(Feats_space_transform_for_CLAM_MB_small, self).__init__(D1_save_path, D2_save_path,
                                                                 D3_save_path, test_read_path, test_wegts_path, feat_len,
                                                                    out_feat_len)

    def creating_model_layers(self):
        from CLAM_MB import CLAM_MB
        from Training_Testing_for_SOTA.training_testing_for_clam import testing_for_clam

        self.testing_model = CLAM_MB(instance_eval=False, feat_dim=768, n_classes=3, size_arg='small')
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(self.test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        self.register_layer = self.testing_model.classifiers[0]

        testing_for_clam(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)




class Feats_space_transform_for_CLAM_MB_big(Feats_space_transform_for_CLAM_SB_small):
    def __init__(self, D1_save_path, D2_save_path, D3_save_path, test_read_path, test_wegts_path = None,
                 feat_len = 768, out_feat_len = 256):
        super(Feats_space_transform_for_CLAM_MB_big, self).__init__(D1_save_path, D2_save_path,
                                                                 D3_save_path, test_read_path, test_wegts_path, feat_len,
                                                                    out_feat_len)

    def creating_model_layers(self):
        from CLAM_MB import CLAM_MB
        from Training_Testing_for_SOTA.training_testing_for_clam import testing_for_clam

        self.testing_model = CLAM_MB(instance_eval=False, feat_dim=768, n_classes=3, size_arg='big')
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(self.test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        self.register_layer = self.testing_model.classifiers[0]

        testing_for_clam(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)


class Feats_space_transform_for_DGRMIL(Feats_space_transform_for_CLAM_SB_small):
    def __init__(self, D1_save_path, D2_save_path, D3_save_path, test_read_path, test_wegts_path = None,
                 feat_len = 768, out_feat_len = 512):
        super(Feats_space_transform_for_DGRMIL, self).__init__(D1_save_path, D2_save_path,
                                                                 D3_save_path, test_read_path, test_wegts_path, feat_len,
                                                                    out_feat_len)

    def creating_model_layers(self):
        from DGR_MIL import DGRMIL
        from Training_Testing_for_SOTA.training_testing_for_dgrmil import testing_for_dgrmil

        self.testing_model = DGRMIL(768, num_classes=3, attn_mode='gated', dropout_node=0.1)
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(self.test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        self.register_layer = self.testing_model.classifier[0]

        testing_for_dgrmil(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)


class Feats_space_transform_for_DTFD_MIL(Feats_space_transform_for_AB_MIL_Gated):
    def __init__(self, D1_save_path, D2_save_path, D3_save_path, test_read_path, test_wegts_path = None,
                 feat_len = 512):
        super(Feats_space_transform_for_DTFD_MIL, self).__init__(D1_save_path, D2_save_path,
                                                                 D3_save_path, test_read_path, test_wegts_path, feat_len)

    def creating_model_layers(self):
        from DTFD_MIL import Attention_with_Classifier
        from Training_Testing_for_SOTA.training_testing_for_dtfdmil import testing_for_dtfdmil

        self.testing_model = Attention_with_Classifier(L=768, num_cls=3)
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(self.test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        self.register_layer = self.testing_model.classifier.fc

        testing_for_dtfdmil(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)


class Feats_space_transform_for_FRMIL(Feats_space_transform_for_AB_MIL_Gated):
    def __init__(self, D1_save_path, D2_save_path, D3_save_path, test_read_path, test_wegts_path = None,
                 feat_len = 512):
        super(Feats_space_transform_for_FRMIL, self).__init__(D1_save_path, D2_save_path,
                                                                 D3_save_path, test_read_path, test_wegts_path, feat_len)

    def creating_model_layers(self):
        from FRMIL import MILNet
        from Training_Testing_for_SOTA.training_testing_for_frmil import testing_for_frmil
        import argparse
        frmil_paras = argparse.ArgumentParser(description='FRMIL Hyparameters')
        frmil_paras.add_argument('--num_feats', type=int, default=768)
        frmil_paras.add_argument('--output_class', type=int, default=3)

        frmil_args = frmil_paras.parse_args()
        self.testing_model = MILNet(frmil_args)
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(self.test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        self.register_layer = self.testing_model.b_classifier

        testing_for_frmil(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)

    def fit_transform(self):
        from Utils.Setup_Seed import setup_seed

        setup_seed(1)

        ## 1-th step
        self.creating_loader_for_ts()

        ## 2-th step
        self.creating_model_layers()

        ####
        self.register_layer.register_forward_hook(self.hook_fn)

        feats_sum = np.zeros((1, self.feat_len))
        labels_sum = np.zeros((1, 1))
        for data_i, label_i in self.test_loader:
            label_i = label_i.reshape(label_i.shape[0], 1)
            labels_sum = np.concatenate((labels_sum, label_i.detach().cpu().numpy()))
            with torch.no_grad():
                pre_y = self.testing_model(data_i.cuda(0))

        for i in range(len(self.feat_list)):
            kkk = torch.mean(self.feat_list[i][0], dim=0)
            feats_sum = np.concatenate((feats_sum,
                                        kkk.detach().cpu().numpy().reshape(1, self.feat_len)))

        self.visual_save_fn(feat_numpy=feats_sum, label_sum=labels_sum,
                            D1_save_path=self.D1_save_path, D2_save_path=self.D2_save_path,
                            D3_save_path=self.D3_save_path)



class Feats_space_transform_for_TransMIL(Feats_space_transform_for_AB_MIL_Gated):
    def __init__(self, D1_save_path, D2_save_path, D3_save_path, test_read_path, test_wegts_path = None,
                 feat_len = 484):
        super(Feats_space_transform_for_TransMIL, self).__init__(D1_save_path, D2_save_path,
                                                                 D3_save_path, test_read_path, test_wegts_path, feat_len)

    def creating_model_layers(self):
        from TransMIL import TransMIL
        from Training_Testing_for_SOTA.training_testing_for_transmil import testing_for_transmil

        self.testing_model = TransMIL(768, n_classes=3, mDim=484)
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(self.test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        self.register_layer = self.testing_model._fc2

        testing_for_transmil(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)

if __name__ == '__main__':

    #### 1. PacMIL
    '''
    feats_transform_for_pacmil = Feats_space_transform_for_TicMIL()
    feats_transform_for_pacmil.fit_transform_for_ticmil()
    '''

    ### 2. ILRA
    '''
    ts_feats_transform = Feats_space_transform_for_ILRA(
                 D1_save_path = r'/data/HP_Projects/TicMIL/Results/Feats_space/1D/ILRA_MIL_1D.csv',
                 D2_save_path = r'/data/HP_Projects/TicMIL/Results/Feats_space/2D/ILRA_MIL_2D.csv',
                 D3_save_path = r'/data/HP_Projects/TicMIL/Results/Feats_space/3D/ILRA_MIL_3D.csv',
                 test_read_path = r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
                 test_wegts_path = r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/ILRA_MIL.pth',
                 feat_len=256)
    ts_feats_transform.fit_transform()
    '''

    ### 3. AB_MIL_Gated
    '''
    ts_feats_transform_for_abmilg = Feats_space_transform_for_AB_MIL_Gated(
                 D1_save_path = r'/data/HP_Projects/TicMIL/Results/Feats_space/1D/AB_MIL_Gated_1D.csv',
                 D2_save_path = r'/data/HP_Projects/TicMIL/Results/Feats_space/2D/AB_MIL_Gated_2D.csv',
                 D3_save_path = r'/data/HP_Projects/TicMIL/Results/Feats_space/3D/AB_MIL_Gated_3D.csv',
                 test_read_path = r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
                 test_wegts_path = r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/AB_MIL_Gated.pth',
                 feat_len=768)
    ts_feats_transform_for_abmilg.fit_transform()
    '''

    ### 4. AB_MIL_Linear
    '''
    ts_feats_transform_for_abmill = Feats_space_transform_for_AB_MIL_Linear(
                 D1_save_path = r'/data/HP_Projects/TicMIL/Results/Feats_space/1D/AB_MIL_Linear_1D.csv',
                 D2_save_path = r'/data/HP_Projects/TicMIL/Results/Feats_space/2D/AB_MIL_Linear_2D.csv',
                 D3_save_path = r'/data/HP_Projects/TicMIL/Results/Feats_space/3D/AB_MIL_Linear_3D.csv',
                 test_read_path = r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
                 test_wegts_path = r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/AB_MIL_Linear.pth',
                 feat_len=768)
    ts_feats_transform_for_abmill.fit_transform()
    '''

    ### 5. CLAM_SB_small
    '''
    ts_feats_transform_for_clamsbs = Feats_space_transform_for_CLAM_SB_small(
                 D1_save_path = r'/data/HP_Projects/TicMIL/Results/Feats_space/1D/CLAM_SB_small_1D.csv',
                 D2_save_path = r'/data/HP_Projects/TicMIL/Results/Feats_space/2D/CLAM_SB_small_2D.csv',
                 D3_save_path = r'/data/HP_Projects/TicMIL/Results/Feats_space/3D/CLAM_SB_small_3D.csv',
                 test_read_path = r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
                 test_wegts_path = r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/CLAM_SB(small).pth',
                 feat_len=768)
    ts_feats_transform_for_clamsbs.fit_transform()
    '''

    ### 6. CLAM_SB_big
    '''
    ts_feats_transform_for_clamsbb = Feats_space_transform_for_CLAM_SB_big(
        D1_save_path=r'/data/HP_Projects/TicMIL/Results/Feats_space/1D/CLAM_SB_big_1D.csv',
        D2_save_path=r'/data/HP_Projects/TicMIL/Results/Feats_space/2D/CLAM_SB_big_2D.csv',
        D3_save_path=r'/data/HP_Projects/TicMIL/Results/Feats_space/3D/CLAM_SB_big_3D.csv',
        test_read_path= r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
        test_wegts_path=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/CLAM_SB(Big).pth',
        feat_len=768, out_feat_len=512)
    ts_feats_transform_for_clamsbb.fit_transform()
    '''

    ### 7. CLAM_MB_small
    '''
    ts_feats_transform_for_clammbs = Feats_space_transform_for_CLAM_MB_small(
        D1_save_path = r'/data/HP_Projects/TicMIL/Results/Feats_space/1D/CLAM_MB_small_1D.csv',
        D2_save_path = r'/data/HP_Projects/TicMIL/Results/Feats_space/2D/CLAM_MB_small_2D.csv',
        D3_save_path = r'/data/HP_Projects/TicMIL/Results/Feats_space/3D/CLAM_MB_small_3D.csv',
        test_read_path = r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
        test_wegts_path = r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/CLAM_MB(small).pth',
        feat_len=768)
    ts_feats_transform_for_clammbs.fit_transform()
    '''

    ### 8. CLAM_MB_big
    '''
    ts_feats_transform_for_clammbb = Feats_space_transform_for_CLAM_MB_big(
        D1_save_path=r'/data/HP_Projects/TicMIL/Results/Feats_space/1D/CLAM_MB_big_1D.csv',
        D2_save_path=r'/data/HP_Projects/TicMIL/Results/Feats_space/2D/CLAM_MB_big_2D.csv',
        D3_save_path=r'/data/HP_Projects/TicMIL/Results/Feats_space/3D/CLAM_MB_big_3D.csv',
        test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
        test_wegts_path=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/CLAM_MB(Big).pth',
        feat_len=768, out_feat_len=512)
    ts_feats_transform_for_clammbb.fit_transform()
    '''

    ### 9. DGRMIL
    '''
    ts_feats_transform_for_dgrmil = Feats_space_transform_for_DGRMIL(
        D1_save_path=r'/data/HP_Projects/TicMIL/Results/Feats_space/1D/DGRMIL_1D.csv',
        D2_save_path=r'/data/HP_Projects/TicMIL/Results/Feats_space/2D/DGRMIL_2D.csv',
        D3_save_path=r'/data/HP_Projects/TicMIL/Results/Feats_space/3D/DGRMIL_3D.csv',
        test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
        test_wegts_path=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/DGR_MIL.pth',
        feat_len=768, out_feat_len=512)
    ts_feats_transform_for_dgrmil.fit_transform()
    '''

    ### 10. DTFD_MIL
    '''
    ts_feats_transform_for_dtfdmil = Feats_space_transform_for_DTFD_MIL(
        D1_save_path=r'/data/HP_Projects/TicMIL/Results/Feats_space/1D/DTFD_MIL_1D.csv',
        D2_save_path=r'/data/HP_Projects/TicMIL/Results/Feats_space/2D/DTFD_MIL_2D.csv',
        D3_save_path=r'/data/HP_Projects/TicMIL/Results/Feats_space/3D/DTFD_MIL_3D.csv',
        test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
        test_wegts_path=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/DTFD_MIL.pth',
        feat_len=768)
    ts_feats_transform_for_dtfdmil.fit_transform()
    '''

    ### 11. FRMIL
    '''
    ts_feats_transform_for_frmil = Feats_space_transform_for_FRMIL(
        D1_save_path=r'/data/HP_Projects/TicMIL/Results/Feats_space/1D/FRMIL_1D.csv',
        D2_save_path=r'/data/HP_Projects/TicMIL/Results/Feats_space/2D/FRMIL_2D.csv',
        D3_save_path=r'/data/HP_Projects/TicMIL/Results/Feats_space/3D/FRMIL_3D.csv',
        test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
        test_wegts_path=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/FRMIL.pth',
        feat_len=768)
    ts_feats_transform_for_frmil.fit_transform()
    '''


    ### 12. TransMIL

    ts_feats_transform_for_transmil = Feats_space_transform_for_TransMIL(
        D1_save_path=r'/data/HP_Projects/TicMIL/Results/Feats_space/1D/TransMIL_1D.csv',
        D2_save_path=r'/data/HP_Projects/TicMIL/Results/Feats_space/2D/TransMIL_2D.csv',
        D3_save_path=r'/data/HP_Projects/TicMIL/Results/Feats_space/3D/TransMIL_3D.csv',
        test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
        test_wegts_path=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/TransMIL.pth',
        feat_len=484)
    ts_feats_transform_for_transmil.fit_transform()

