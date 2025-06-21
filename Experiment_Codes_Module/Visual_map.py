import numpy as np
import cv2
from termcolor import colored
import argparse
import torch
from skimage import io

class Recons_WSI():
    def __init__(self, read_path = r'/data/HP_Projects/TicMIL/Datasets/Cervix/Cervix_MIL_961/Test/I',
                       save_path = r'/data/HP_Projects/TicMIL/Datasets/Cervix/WSI_org1/Test/I'):
        super(Recons_WSI, self).__init__()
        self.read_path = read_path
        self.save_path = save_path

    def recons_tranform(self):
        import cv2
        from skimage import io
        import os
        import natsort
        import numpy as np
        path_name_list = natsort.natsorted(os.listdir(self.read_path), alg=natsort.ns.PATH)

        for img_name_i in path_name_list:
            count = 0
            wsi_list = []
            row_np = np.zeros((96, 96, 3))
            white_img = np.ones((96, 96, 3)) + 255
            for img_i in natsort.natsorted(os.listdir(self.read_path + r'/' + img_name_i), alg=natsort.ns.PATH):
                path_img = io.imread(self.read_path + r'/' + img_name_i + r'/' + img_i)
                if np.mean(path_img) == 0:
                    path_img = white_img
                print(path_img.shape)
                print(row_np.shape)
                row_np = np.concatenate((row_np, path_img), axis=1)
                #cv2.imshow('Image Window', path_img)
                #cv2.waitKey(0)
                count += 1
                if count % 31 == 0:
                    wsi_list.append(row_np[:, 96:])
                    row_np = np.zeros((96, 96, 3))
                else:
                    pass


            for i in wsi_list:
                recs_wsi_img = np.concatenate(wsi_list, axis=0)
            print(recs_wsi_img.shape)
            cv2.imwrite(self.save_path + r'/' + img_name_i + '.jpg', recs_wsi_img)



class Visual_Map_for_TicMIL():
    def __init__(self, show_size = (3000, 3000), save_path = None, end_no = 961, img_path = None,
                    vector_len = 961, map_size = (31, 31), test_read_path = None, show_stats = False):
        super(Visual_Map_for_TicMIL, self).__init__()
        self.show_size = show_size
        self.save_path = save_path
        self.vector_len = vector_len
        self.map_size = map_size
        self.testing_model_feature = None
        self.testing_model_head = None
        self.feat_list = []
        self.test_read_path = test_read_path
        self.show_stats = show_stats
        self.end_no = end_no
        self.img_path = img_path

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

    def create_img_loader(self):
        from torch.utils.data import DataLoader, Dataset
        from torchvision.datasets import ImageFolder
        from torchvision import transforms
        img_paras = argparse.ArgumentParser(description='Image Loader Hyparameters')
        img_paras.add_argument('--batch_size', type=int, default=1)
        img_paras.add_argument('--img_size', type=list, default=[3000, 3000])
        img_paras.add_argument('--num_workers', type=int, default=16)

        img_args = img_paras.parse_args()
        transform = transforms.Compose([transforms.Resize(img_args.img_size), transforms.ToTensor()])
        img_data = ImageFolder(self.img_path, transform=transform)

        img_loader = DataLoader(img_data, batch_size=img_args.batch_size, shuffle=False,
                                 num_workers=img_args.num_workers)
        return img_loader


    def create_model_layer(self):
        from torchsummary import summary
        from Utils.Setup_Seed import setup_seed
        from Models.TicMIL_model_modules import TicMIL_Parallel_Feature, TicMIL_Parallel_Head_for_visual
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
            print('########################## TicMIL_summary #########################')
            summary(swinT_base, (3, 96, 96), device='cpu')
            print('\n', '########################## TicMIL #########################')
            print(swinT_base, '\n')

        ticmil_feature = TicMIL_Parallel_Feature(base_model=swinT_base)
        ticmil_head = TicMIL_Parallel_Head_for_visual(base_model=swinT_base, class_num=ticmil_args.class_num,
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
        #testing_for_ticmil_parallel(mil_feature=ticmil_feature, mil_head=ticmil_head, train_loader=test_loader,
        #                            data_parallel=ticmil_args.data_parallel, roc_save_path=ticmil_args.roc_save_path,
        #                            batch_size=ticmil_args.batch_size,
        #                            confusion_save_path=ticmil_args.confusion_save_path,
        #                            bags_len=ticmil_args.bags_len, val_loader=test_loader, test_loader=test_loader)

        self.testing_model_feature = ticmil_feature
        self.testing_model_head = ticmil_head


    def visual_transform(self, img_1, relation_vector, alpha = 0.1):
        from skimage.transform import resize
        from cv2 import addWeighted
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        from skimage import io

        img_1_bw = relation_vector
        img_1_bw = img_1_bw - np.min(img_1_bw)
        img_1_bw = img_1_bw / (np.max(img_1_bw))

        img_1_bw = np.reshape(img_1_bw, (self.vector_len, 1))

        img_1_bw[img_1_bw < alpha] = 0
        img_1_bw[img_1_bw < 0.5] *= 5

        mask_1 = np.reshape(img_1_bw, self.map_size)

        if self.show_stats == True:
            cv2.imshow('Image Window', mask_1)
            cv2.waitKey(0)
        else:
            pass
        mask_1 = resize(mask_1, (self.show_size[0], self.show_size[1], 1))
        white_img = np.zeros((self.show_size[0], self.show_size[1], 1))
        mask_1 = np.concatenate((mask_1, mask_1, white_img), axis=2)

        img_1 = resize(img_1, (self.show_size[0], self.show_size[1]))


        if self.show_stats == True:
            cv2.imshow('Image Window', img_1)
            cv2.waitKey(0)
        else:
            pass
        #print(type(img_1[0, 0, 1]))
        #print(type(mask_1[0, 0, 1]))
        interpret_img = addWeighted(mask_1, 0.4, img_1, 0.6, 0.12)

        if self.show_stats == True:
            cv2.imshow('Image Window', interpret_img)
            cv2.waitKey(0)
        else:
            pass

        return interpret_img

    def get_visual_map(self):
        self.create_model_layer()
        test_loader = self.create_loader()
        img_loader = self.create_img_loader()

        with torch.no_grad():
            for data_i, label_i in test_loader:
                for i in data_i:
                    y = self.testing_model_feature(i.cuda(0))
                    y, y_rel, _ = self.testing_model_head(y)
                    y_rel = y_rel.reshape(y_rel.shape[1], y_rel.shape[2])
                    y_rel = y_rel[:self.end_no, :]
                    y_rel = torch.mean(y_rel, dim=1)
                    y_rel = y_rel.reshape((1, self.vector_len))
                    print(y_rel.shape)
                    self.feat_list.append(y_rel)

        feat_numpy = np.zeros((1, self.vector_len))
        for i in range(self.feat_list.__len__()):
            kk = self.feat_list[i].detach().cpu().numpy()
            print(kk.shape)
            feat_numpy = np.concatenate((feat_numpy, kk.reshape((1, self.vector_len))))
            print(feat_numpy.shape)
        feat_numpy = feat_numpy[1:, :]

        for count_i, (img_i, _) in enumerate(img_loader):
            #print(count_i)
            img_i = img_i.permute(0, 2, 3, 1)
            img_i = img_i.detach().cpu().numpy()
            img_i = np.reshape(img_i, (img_i.shape[1], img_i.shape[2], img_i.shape[3]))
            #print(img_i.shape)
            #print(feat_numpy.shape)
            interpret_img = self.visual_transform(img_i.astype('float64'), feat_numpy[count_i, :], alpha = 0.1)
            #print(interpret_img[0, 0, 1])
            interpret_img = (interpret_img / np.max(interpret_img)) * 255.0
            interpret_img = interpret_img.astype('uint8')
            io.imsave(self.save_path + r'/' + str(count_i + 1) + r'.png', interpret_img)
            count_i += 1
            print(count_i)


class Visual_Map_for_ILRA(Visual_Map_for_TicMIL):
    def __init__(self, show_size = (512, 512), save_path = None, end_no = 961, img_path = None,
                    vector_len = 961, map_size = (31, 31), test_read_path = None, show_stats = False, incrs_scale = 5,
                 shadow_val = 0.5, alpha = 0.1, pixel_scale = 240):
        super(Visual_Map_for_ILRA, self).__init__(show_size, save_path, end_no, img_path,
                    vector_len, map_size, test_read_path, show_stats)
        self.test_loader = None
        self.incrs_sclae = incrs_scale
        self.shadow_val = shadow_val
        self.alpha = alpha
        self.pixel_scale = pixel_scale

    def create_loader(self):
        from torch.utils.data import DataLoader
        from Read_Feats_Datasets import Read_Feats_Datasets
        two_steps_paras = argparse.ArgumentParser(description='Two Steps Hyparameters')
        two_steps_paras.add_argument('--batch_size', type=int, default=1)
        two_steps_paras.add_argument('--shuffle_stats', type=bool, default=False)
        two_steps_paras.add_argument('--num_workers', type=int, default=16)

        two_steps_args = two_steps_paras.parse_args()
        print(self.test_read_path)
        test_dataset = Read_Feats_Datasets(data_size=[1025, 768],
                                           data_path=self.test_read_path + r'/Feats',
                                           label_path=self.test_read_path + r'/Labels')
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=two_steps_args.batch_size,
                                 shuffle=two_steps_args.shuffle_stats, num_workers=two_steps_args.num_workers)


    def visual_transform(self, img_1, relation_vector, alpha = 0.1):
        from skimage.transform import resize
        from cv2 import addWeighted
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        from skimage import io

        img_1_bw = relation_vector
        img_1_bw = img_1_bw - np.min(img_1_bw)
        img_1_bw = img_1_bw / (np.max(img_1_bw))

        img_1_bw = np.reshape(img_1_bw, (self.vector_len, 1))

        img_1_bw[img_1_bw < self.alpha] = 0
        img_1_bw[img_1_bw < self.shadow_val] *= self.incrs_sclae

        mask_1 = np.reshape(img_1_bw, self.map_size)

        if self.show_stats == True:
            cv2.imshow('Image Window', mask_1)
            cv2.waitKey(0)
        else:
            pass
        mask_1 = resize(mask_1, (self.show_size[0], self.show_size[1], 1))
        white_img = np.zeros((self.show_size[0], self.show_size[1], 1))
        mask_1 = np.concatenate((mask_1, mask_1, white_img), axis=2)

        img_1 = resize(img_1, (self.show_size[0], self.show_size[1]))


        if self.show_stats == True:
            cv2.imshow('Image Window', img_1)
            cv2.waitKey(0)
        else:
            pass
        #print(type(img_1[0, 0, 1]))
        #print(type(mask_1[0, 0, 1]))
        interpret_img = addWeighted(mask_1, 0.4, img_1, 0.6, 0.12)

        if self.show_stats == True:
            cv2.imshow('Image Window', interpret_img)
            cv2.waitKey(0)
        else:
            pass

        return interpret_img

    def hook_fn(self, model, feats_in, feats_out):
        kkk = feats_in[0].detach().cpu().numpy()
        if np.ndim(kkk) == 3:
            kkk = kkk.reshape((kkk.shape[1], kkk.shape[2]))
        kkk = np.mean(kkk, axis=1)
        self.feat_list.append(kkk[:self.end_no])

    def create_model_layer(self):
        from ILRA_MIL import ILRA
        from Training_Testing_for_SOTA.training_testing_for_ilramil import testing_for_ilramil
        test_wegts_path = r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/ILRA_MIL.pth'
        self.testing_model = ILRA(feat_dim=768, n_classes=3, hidden_feat=256, num_heads=8, topk=1)
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)

        testing_for_ilramil(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)

        self.register_layer = self.testing_model.pooling

    def get_visual_map(self):
        self.create_loader()
        self.create_model_layer()
        img_loader = self.create_img_loader()
        self.register_layer.register_forward_hook(self.hook_fn)

        with torch.no_grad():
            count = 0
            for data_i, label_i in self.test_loader:
                y = self.testing_model(data_i.cuda(0))
                #print(self.feat_list[count].shape)
                count += 1

        #print(self.vector_len)
        feat_numpy = np.zeros((1, self.vector_len))
        for i in range(len(self.feat_list)):
            print(self.feat_list[i].shape)
            kk = self.feat_list[i]
            feat_numpy = np.concatenate((feat_numpy, kk.reshape((1, self.vector_len))))
        feat_numpy = feat_numpy[1:, :]

        for count_i, (img_i, _) in enumerate(img_loader):
            # print(count_i)
            img_i = img_i.permute(0, 2, 3, 1)
            img_i = img_i.detach().cpu().numpy()
            img_i = np.reshape(img_i, (img_i.shape[1], img_i.shape[2], img_i.shape[3]))
            interpret_img = self.visual_transform(img_i.astype('float64'), feat_numpy[count_i, :], alpha=0.1)
            # print(interpret_img[0, 0, 1])
            interpret_img = interpret_img * self.pixel_scale
            interpret_img = interpret_img.astype('uint8')
            io.imsave(self.save_path + r'/' + str(count_i + 1) + r'.png', interpret_img)
            count_i += 1
            print(count_i)



class Visual_Map_for_AB_MIL_Gated(Visual_Map_for_ILRA):
    def __init__(self, show_size = (512, 512), save_path = None, end_no = 961, img_path = None,
                    vector_len = 961, map_size = (31, 31), test_read_path = None, show_stats = False, incrs_scale = 5,
                 shadow_val = 0.5, alpha = 0.1, pixel_scale = 240):
        super(Visual_Map_for_AB_MIL_Gated, self).__init__(show_size, save_path, end_no, img_path,
                    vector_len, map_size, test_read_path, show_stats, incrs_scale,
                 shadow_val, alpha, pixel_scale)


    def hook_fn(self, model, feats_in, feats_out):
        print(feats_out.shape)
        kkk = feats_out.detach().cpu().numpy()
        if np.ndim(kkk) == 3:
            kkk = kkk.reshape((kkk.shape[1], kkk.shape[2]))
        kkk = np.mean(kkk, axis=1)
        self.feat_list.append(kkk[:self.end_no])


    def create_model_layer(self):
        from AB_MIL_Gated import AttentionMIL
        from Training_Testing_for_SOTA.training_testing_for_abmil import testing_for_abmil
        test_wegts_path = r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/AB_MIL_Gated.pth'
        self.testing_model = AttentionMIL(768, attn_mode='gated', dropout_node=0.1, num_classes=3)
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        testing_for_abmil(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)

        self.register_layer = self.testing_model.attention_weights


    def get_visual_map(self):
        self.create_loader()
        self.create_model_layer()
        img_loader = self.create_img_loader()
        self.register_layer.register_forward_hook(self.hook_fn)

        with torch.no_grad():
            count = 0
            for data_i, label_i in self.test_loader:
                data_i = data_i.reshape((data_i.shape[1], data_i.shape[2]))
                y = self.testing_model(data_i.cuda(0))
                print(self.feat_list[count].shape)
                count += 1

        print(self.vector_len)
        feat_numpy = np.zeros((1, self.vector_len))
        for i in range(len(self.feat_list)):
            kk = self.feat_list[i]
            # print(kk.shape)
            feat_numpy = np.concatenate((feat_numpy, kk.reshape((1, self.vector_len))))
        feat_numpy = feat_numpy[1:, :]

        for count_i, (img_i, _) in enumerate(img_loader):
            # print(count_i)
            img_i = img_i.permute(0, 2, 3, 1)
            img_i = img_i.detach().cpu().numpy()
            img_i = np.reshape(img_i, (img_i.shape[1], img_i.shape[2], img_i.shape[3]))
            interpret_img = self.visual_transform(img_i.astype('float64'), feat_numpy[count_i, :], alpha=0.3)
            # print(interpret_img[0, 0, 1])
            interpret_img = interpret_img * self.pixel_scale
            interpret_img = interpret_img.astype('uint8')
            io.imsave(self.save_path + r'/' + str(count_i + 1) + r'.png', interpret_img)
            count_i += 1
            print(count_i)


class Visual_Map_for_AB_MIL_Linear(Visual_Map_for_AB_MIL_Gated):
    def __init__(self, show_size=(512, 512), save_path=None, end_no=961, img_path=None,
                 vector_len=961, map_size=(31, 31), test_read_path=None, show_stats=False, incrs_scale=5,
                 shadow_val=0.5, alpha=0.1, pixel_scale=240):
        super(Visual_Map_for_AB_MIL_Linear, self).__init__(show_size, save_path, end_no, img_path,
                                                          vector_len, map_size, test_read_path, show_stats, incrs_scale,
                                                          shadow_val, alpha, pixel_scale)


    def create_model_layer(self):
        from AB_MIL_Gated import AttentionMIL
        from Training_Testing_for_SOTA.training_testing_for_abmil import testing_for_abmil
        test_wegts_path = r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/AB_MIL_Linear.pth'
        self.testing_model = AttentionMIL(768, attn_mode='linear', dropout_node=0.1, num_classes=3)
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        testing_for_abmil(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)

        self.register_layer = self.testing_model.attention




class Visual_Map_for_CLAM_SB_small(Visual_Map_for_AB_MIL_Gated):
    def __init__(self, show_size=(512, 512), save_path=None, end_no=961, img_path=None,
                 vector_len=961, map_size=(31, 31), test_read_path=None, show_stats=False, incrs_scale=5,
                 shadow_val=0.5, alpha=0.1, pixel_scale=240):
        super(Visual_Map_for_CLAM_SB_small, self).__init__(show_size, save_path, end_no, img_path,
                                                          vector_len, map_size, test_read_path, show_stats, incrs_scale,
                                                          shadow_val, alpha, pixel_scale)


    def create_model_layer(self):
        from CLAM_SB import CLAM_SB
        from Training_Testing_for_SOTA.training_testing_for_clam import testing_for_clam
        test_wegts_path = r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/CLAM_SB(small).pth'
        self.testing_model = CLAM_SB(instance_eval=False, feat_dim=768, n_classes=3, size_arg='small')
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        testing_for_clam(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)

        self.register_layer = self.testing_model.attention_net

    def hook_fn(self, model, feats_in, feats_out):
        kkk = feats_out[0].detach().cpu().numpy()
        if np.ndim(kkk) == 3:
            kkk = kkk.reshape((kkk.shape[1], kkk.shape[2]))
        self.feat_list.append(kkk[:self.end_no])

    def get_visual_map(self):
        self.create_loader()
        self.create_model_layer()
        img_loader = self.create_img_loader()
        self.register_layer.register_forward_hook(self.hook_fn)

        with torch.no_grad():
            count = 0
            for data_i, label_i in self.test_loader:
                data_i = data_i.reshape((data_i.shape[1], data_i.shape[2]))
                y = self.testing_model(data_i.cuda(0), torch.zeros((1)).cuda(0))
                print(self.feat_list[count].shape)
                print(y[2].shape)
                count += 1


        feat_numpy = np.zeros((1, self.vector_len))
        for i in range(len(self.feat_list)):
            kk = self.feat_list[i]
            feat_numpy = np.concatenate((feat_numpy, kk.reshape((1, self.vector_len))))
        feat_numpy = feat_numpy[1:, :]

        for count_i, (img_i, _) in enumerate(img_loader):
            # print(count_i)
            img_i = img_i.permute(0, 2, 3, 1)
            img_i = img_i.detach().cpu().numpy()
            img_i = np.reshape(img_i, (img_i.shape[1], img_i.shape[2], img_i.shape[3]))
            interpret_img = self.visual_transform(img_i.astype('float64'), feat_numpy[count_i, :])
            # print(interpret_img[0, 0, 1])
            interpret_img = interpret_img * 225.0
            interpret_img = interpret_img.astype('uint8')
            io.imsave(self.save_path + r'/' + str(count_i + 1) + r'.png', interpret_img)
            count_i += 1
            print(count_i)



class Visual_Map_for_CLAM_SB_big(Visual_Map_for_CLAM_SB_small):
    def __init__(self, show_size=(512, 512), save_path=None, end_no=961, img_path=None,
                 vector_len=961, map_size=(31, 31), test_read_path=None, show_stats=False, incrs_scale=5,
                 shadow_val=0.5, alpha=0.1, pixel_scale=240):
        super(Visual_Map_for_CLAM_SB_small, self).__init__(show_size, save_path, end_no, img_path,
                                                           vector_len, map_size, test_read_path, show_stats,
                                                           incrs_scale, shadow_val, alpha, pixel_scale)


    def create_model_layer(self):
        from CLAM_SB import CLAM_SB
        from Training_Testing_for_SOTA.training_testing_for_clam import testing_for_clam
        test_wegts_path = r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/CLAM_SB(Big).pth'
        self.testing_model = CLAM_SB(instance_eval=False, feat_dim=768, n_classes=3, size_arg='big')
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        testing_for_clam(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)

        self.register_layer = self.testing_model.attention_net



class Visual_Map_for_CLAM_MB_small(Visual_Map_for_CLAM_SB_small):
    def __init__(self, show_size=(512, 512), save_path=None, end_no=961, img_path=None,
                 vector_len=961, map_size=(31, 31), test_read_path=None, show_stats=False, incrs_scale=5,
                 shadow_val=0.5, alpha=0.1, pixel_scale=240):
        super(Visual_Map_for_CLAM_MB_small, self).__init__(show_size, save_path, end_no, img_path,
                                                           vector_len, map_size, test_read_path, show_stats,
                                                           incrs_scale, shadow_val, alpha, pixel_scale)

    def hook_fn(self, model, feats_in, feats_out):
        kkk = feats_out[0].detach().cpu().numpy()
        if np.ndim(kkk) == 3:
            kkk = kkk.reshape((kkk.shape[1], kkk.shape[2]))
        kkk = np.mean(kkk, axis=1)
        self.feat_list.append(kkk[:self.end_no])


    def create_model_layer(self):
        from CLAM_MB import CLAM_MB
        from Training_Testing_for_SOTA.training_testing_for_clam import testing_for_clam
        test_wegts_path = r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/CLAM_MB(small).pth'
        self.testing_model = CLAM_MB(instance_eval=False, feat_dim=768, n_classes=3, size_arg='small')
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        testing_for_clam(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)

        self.register_layer = self.testing_model.attention_net


class Visual_Map_for_CLAM_MB_big(Visual_Map_for_CLAM_MB_small):
    def __init__(self, show_size=(512, 512), save_path=None, end_no=961, img_path=None,
                 vector_len=961, map_size=(31, 31), test_read_path=None, show_stats=False, incrs_scale=5,
                 shadow_val=0.5, alpha=0.1, pixel_scale=240):
        super(Visual_Map_for_CLAM_MB_big, self).__init__(show_size, save_path, end_no, img_path,
                                                           vector_len, map_size, test_read_path, show_stats,
                                                           incrs_scale, shadow_val, alpha, pixel_scale)


    def create_model_layer(self):
        from CLAM_MB import CLAM_MB
        from Training_Testing_for_SOTA.training_testing_for_clam import testing_for_clam
        test_wegts_path = r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/CLAM_MB(Big).pth'
        self.testing_model = CLAM_MB(instance_eval=False, feat_dim=768, n_classes=3, size_arg='big')
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        testing_for_clam(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)

        self.register_layer = self.testing_model.attention_net




class Visual_Map_for_DGRMIL(Visual_Map_for_ILRA):
    def __init__(self, show_size=(512, 512), save_path=None, end_no=961, img_path=None,
                 vector_len=961, map_size=(31, 31), test_read_path=None, show_stats=False, incrs_scale=5,
                 shadow_val=0.5, alpha=0.1, pixel_scale=240):
        super(Visual_Map_for_DGRMIL, self).__init__(show_size, save_path, end_no, img_path,
                                                  vector_len, map_size, test_read_path, show_stats, incrs_scale,
                                                    shadow_val, alpha, pixel_scale)

    def hook_fn(self, model, feats_in, feats_out):
        kkk = feats_out[0].detach().cpu().numpy()
        print('my_debug: ', kkk.shape)
        if np.ndim(kkk) == 3:
            kkk = kkk.reshape((kkk.shape[1], kkk.shape[2]))
        #print(kkk.shape)
        kkk = np.mean(kkk, axis=1, keepdims=True)
        kkk = kkk[:self.end_no]
        if kkk.shape[0] > 100:
            self.feat_list.append(kkk)

    def create_model_layer(self):
        from DGR_MIL import DGRMIL
        from Training_Testing_for_SOTA.training_testing_for_dgrmil import testing_for_dgrmil
        test_wegts_path = r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/DGR_MIL.pth'
        self.testing_model = DGRMIL(768, num_classes=3, attn_mode='linear', dropout_node=0.1)
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        testing_for_dgrmil(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)

        self.register_layer = self.testing_model.triple_optimizer




class Visual_Map_for_DTFD_MIL(Visual_Map_for_AB_MIL_Gated):
    def __init__(self, show_size=(512, 512), save_path=None, end_no=961, img_path=None,
                 vector_len=961, map_size=(31, 31), test_read_path=None, show_stats=False, incrs_scale=5,
                 shadow_val=0.5, alpha=0.1, pixel_scale=240):
        super(Visual_Map_for_DTFD_MIL, self).__init__(show_size, save_path, end_no, img_path,
                                                           vector_len, map_size, test_read_path, show_stats,
                                                           incrs_scale, shadow_val, alpha, pixel_scale)

    def hook_fn(self, model, feats_in, feats_out):
        kkk = feats_out[0].detach().cpu().numpy()
        print(kkk.shape)
        if np.ndim(kkk) == 3:
            kkk = kkk.reshape((kkk.shape[1], kkk.shape[2]))
        kkk = kkk[:self.end_no]
        if kkk.shape[0] > 100:
            self.feat_list.append(kkk)


    def create_model_layer(self):
        from DTFD_MIL import Attention_with_Classifier
        from Training_Testing_for_SOTA.training_testing_for_dtfdmil import testing_for_dtfdmil
        test_wegts_path = r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/DTFD_MIL.pth'
        self.testing_model = Attention_with_Classifier(L=768, num_cls=3)
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        testing_for_dtfdmil(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)

        self.register_layer = self.testing_model.attention


class Visual_Map_for_FRMIL(Visual_Map_for_ILRA):
    def __init__(self, show_size=(512, 512), save_path=None, end_no=961, img_path=None,
                 vector_len=961, map_size=(31, 31), test_read_path=None, show_stats=False, incrs_scale=5,
                 shadow_val=0.5, alpha=0.1, pixel_scale=240):
        super(Visual_Map_for_FRMIL, self).__init__(show_size, save_path, end_no, img_path,
                                                          vector_len, map_size, test_read_path, show_stats, incrs_scale,
                                                          shadow_val, alpha, pixel_scale)

    def hook_fn(self, model, feats_in, feats_out):
        kkk = feats_out[1].detach().cpu().numpy()
        print(kkk.shape)
        if np.ndim(kkk) == 3:
            kkk = kkk.reshape((kkk.shape[1], kkk.shape[2]))
        #print(kkk.shape)
        kkk = np.mean(kkk, axis=1, keepdims=True)
        kkk = kkk[:self.end_no]
        if kkk.shape[0] > 100:
            self.feat_list.append(kkk)

    def create_model_layer(self):
        from FRMIL import MILNet
        from Training_Testing_for_SOTA.training_testing_for_frmil import testing_for_frmil
        test_wegts_path = r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/FRMIL.pth'
        import argparse
        frmil_paras = argparse.ArgumentParser(description='FRMIL Hyparameters')
        frmil_paras.add_argument('--num_feats', type=int, default=768)
        frmil_paras.add_argument('--output_class', type=int, default=3)
        frmil_args = frmil_paras.parse_args()
        self.testing_model = MILNet(frmil_args)
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        testing_for_frmil(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)

        self.register_layer = self.testing_model.b_classifier



class Visual_Map_for_TransMIL(Visual_Map_for_AB_MIL_Gated):
    def __init__(self, show_size=(512, 512), save_path=None, end_no=961, img_path=None,
                 vector_len=961, map_size=(31, 31), test_read_path=None, show_stats=False, incrs_scale=5,
                 shadow_val=0.5, alpha=0.1, pixel_scale=240):
        super(Visual_Map_for_TransMIL, self).__init__(show_size, save_path, end_no, img_path,
                                                           vector_len, map_size, test_read_path, show_stats,
                                                           incrs_scale,
                                                           shadow_val, alpha, pixel_scale)

    def hook_fn(self, model, feats_in, feats_out):
        kkk = feats_out.detach().cpu().numpy()
        print(kkk.shape)
        if np.ndim(kkk) == 3:
            kkk = kkk.reshape((kkk.shape[1], kkk.shape[2]))
        #print(kkk.shape)
        kkk = np.mean(kkk, axis=1, keepdims=True)
        kkk = kkk[:self.end_no]
        if kkk.shape[0] > 100:
            self.feat_list.append(kkk)

    def create_model_layer(self):
        from TransMIL import TransMIL
        from Training_Testing_for_SOTA.training_testing_for_transmil import testing_for_transmil
        test_wegts_path = r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Other_SOTA/TransMIL.pth'
        self.testing_model = TransMIL(768, n_classes=3, mDim=484)
        self.testing_model = self.testing_model.cuda(0)
        testing_model_wegts = torch.load(test_wegts_path, map_location='cuda:0')
        self.testing_model.load_state_dict(testing_model_wegts, strict=True)
        testing_for_transmil(test_model=self.testing_model, train_loader=self.test_loader, val_loader=self.test_loader,
                            proba_value=None, test_loader=self.test_loader, gpu_device=0,
                            out_mode=None, proba_mode=False, class_num=3,
                            roc_save_path=None, bags_stat=True, bag_relations_path=None)

        self.register_layer = self.testing_model._fc1


if __name__ == '__main__':
    ## alpha :像素置零的阈值，shadow_val:放大的阈值大小
    ### 1. TicMIL
    '''
    ours_visual = Visual_Map_for_TicMIL(show_size = (3000, 3000), show_stats=False,
                save_path = r'/data/HP_Projects/TicMIL/Results/Visual/TicMIL',
                img_path = r'/data/HP_Projects/TicMIL/Datasets/Cervix/WSI_org/Test',
                test_read_path = r'/data/HP_Projects/TicMIL/Datasets/Cervix/New_Bags/Test')
    ours_visual.get_visual_map()
    '''

    ### 2. ILRA_MIL
    '''
    sota_visual_for_ilra = Visual_Map_for_ILRA(show_size = (3000, 3000), show_stats=False,
                                      save_path = r'/data/HP_Projects/TicMIL/Results/Visual/ILRA_MIL',
                                      img_path =  r'/data/HP_Projects/TicMIL/Datasets/Cervix/WSI_org/Test',
                                      test_read_path = r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test')

    sota_visual_for_ilra.get_visual_map()
    '''

    ### 3. AB_MIL_Gated
    '''
    sota_visual_for_abmilg = Visual_Map_for_AB_MIL_Gated(show_size=(3000, 3000), show_stats=False,
                                               save_path = r'/data/HP_Projects/TicMIL/Results/Visual/AB_MIL_Gated',
                                               img_path = r'/data/HP_Projects/TicMIL/Datasets/Cervix/WSI_org/Test',
                                               test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
                                               alpha=0.3, incrs_scale=1.0, shadow_val=0.5, pixel_scale=250)

    sota_visual_for_abmilg.get_visual_map()
    '''

    ### 4. AB_MIL_Linear
    '''
    sota_visual_for_abmill = Visual_Map_for_AB_MIL_Linear(show_size=(3000, 3000), show_stats=False,
                                                save_path=r'/data/HP_Projects/TicMIL/Results/Visual/AB_MIL_Linear',
                                                img_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/WSI_org/Test',
                                                test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
                                                          alpha=0.3, incrs_scale=1.0, shadow_val=0.5, pixel_scale=250)

    sota_visual_for_abmill.get_visual_map()
    '''

    ### 5. CLAM_SB_small
    '''
    sota_visual_for_clamsbs = Visual_Map_for_CLAM_SB_small(show_size=(3000, 3000), show_stats=False,
                                                save_path=r'/data/HP_Projects/TicMIL/Results/Visual/CLAM_SB_small',
                                                img_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/WSI_org/Test',
                                                test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
                                                alpha=0.3, incrs_scale=1.0, shadow_val=0.5, pixel_scale=250)

    sota_visual_for_clamsbs.get_visual_map()
    '''

    ### 6. CLAM_SB_big
    '''
    sota_visual_for_clamsbb = Visual_Map_for_CLAM_SB_big(show_size=(3000, 3000), show_stats=False,
                                                save_path=r'/data/HP_Projects/TicMIL/Results/Visual/CLAM_SB_big',
                                                img_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/WSI_org/Test',
                                                test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
                                                alpha=0.5, incrs_scale=1.0, shadow_val=0.65, pixel_scale=250)

    sota_visual_for_clamsbb.get_visual_map()
    '''

    ### 7. CLAM_MB_small
    '''
    sota_visual_for_clammbs = Visual_Map_for_CLAM_MB_small(show_size=(3000, 3000), show_stats=False,
                                                save_path=r'/data/HP_Projects/TicMIL/Results/Visual/CLAM_MB_small',
                                                img_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/WSI_org/Test',
                                                test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
                                                alpha=0.5, incrs_scale=1.0, shadow_val=0.65, pixel_scale=250)

    sota_visual_for_clammbs.get_visual_map()
    '''

    ### 8. CLAM_MB_big
    '''
    sota_visual_for_clammb = Visual_Map_for_CLAM_MB_big(show_size=(3000, 3000), show_stats=False,
                                                save_path=r'/data/HP_Projects/TicMIL/Results/Visual/CLAM_MB_big',
                                                img_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/WSI_org/Test',
                                                test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
                                                alpha=0.5, incrs_scale=1.0, shadow_val=0.65, pixel_scale=250)

    sota_visual_for_clammb.get_visual_map()
    '''

    ### 9. DGRMIL
    '''
    sota_visual_for_dgrmil = Visual_Map_for_DGRMIL(show_size=(3000, 3000), show_stats=False,
                                                save_path=r'/data/HP_Projects/TicMIL/Results/Visual/DGRMIL',
                                                img_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/WSI_org/Test',
                                                test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
                                                alpha=0.5, incrs_scale=1.0, shadow_val=0.65, pixel_scale=250)

    sota_visual_for_dgrmil.get_visual_map()
    '''

    ### 10. DTFD_MIL
    '''
    sota_visual_for_dtfdmil = Visual_Map_for_DTFD_MIL(show_size=(3000, 3000), show_stats=False,
                                                save_path=r'/data/HP_Projects/TicMIL/Results/Visual/DTFD_MIL',
                                                img_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/WSI_org/Test',
                                                test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
                                                alpha=0.1, incrs_scale=4, shadow_val=0.3, pixel_scale=220)

    sota_visual_for_dtfdmil.get_visual_map()
    '''

    ### 11. FRMIL
    '''
    sota_visual_for_frmil = Visual_Map_for_FRMIL(show_size=(3000, 3000), show_stats=False,
                                                save_path=r'/data/HP_Projects/TicMIL/Results/Visual/FRMIL',
                                                img_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/WSI_org/Test',
                                                test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
                                                alpha=0.1, incrs_scale=4, shadow_val=0.3, pixel_scale=220)

    sota_visual_for_frmil.get_visual_map()
    '''


    ### 12. TransMIL

    sota_visual_for_transmil = Visual_Map_for_TransMIL(show_size=(3000, 3000), show_stats=False,
                                                save_path=r'/data/HP_Projects/TicMIL/Results/Visual/TransMIL',
                                                img_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/WSI_org1/Test',
                                                test_read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_WSI/Test',
                                                alpha=0.3, incrs_scale=1, shadow_val=0.5, pixel_scale=220)

    sota_visual_for_transmil.get_visual_map()


    ### Recons WSI
    '''
    recs_wsi = Recons_WSI(read_path = r'/data/HP_Projects/TicMIL/Datasets/Cervix/Cervix_MIL_961/Test/III',
                       save_path = r'/data/HP_Projects/TicMIL/Datasets/Cervix/WSI_org/Test/III')
    recs_wsi.recons_tranform()
    '''

















