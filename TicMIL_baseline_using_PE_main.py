############################# TicMIL Demo main function ##############################
#### Author: Dr.Pan Huang
#### Email: XXx
#### Department: CQU
#### Attempt: Testing TicMIL Demo
#### TicMIL: A Tumor-guiding Instance Clustered Multi-instance Learning Network for Cervix Grading from Whole-slide Images

########################## API Section #########################
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import random
from torchsummary import summary
from Models.SwinT_models.models.swin_transformer import SwinTransformer
from Models.TicMIL_model_modules import (TicMIL_for_ablation, TicMIL_Parallel_Head_for_ablation,
                                         TicMIL_Parallel_Feature_for_ablation)
from Utils.fit_functions import training_for_baseline_parallel, testing_for_parallel_baseline
from Utils.Setup_Seed import setup_seed
from Utils.Read_MIL_Datasets import Read_MIL_Datasets
import seaborn as sns
from torch.nn.parallel import DataParallel
import argparse


sns.set(font='Times New Roman', font_scale=0.6)

def worker_init_fn(worker_id):
    random.seed(7 + worker_id)
    np.random.seed(7 + worker_id)
    torch.manual_seed(7 + worker_id)
    torch.cuda.manual_seed(7 + worker_id)
    torch.cuda.manual_seed_all(7 + worker_id)

########################## main_function #########################
if __name__ == '__main__':
    ########################## Hyparameters #########################
    paras = argparse.ArgumentParser(description='TicMIL Hyparameters')
    paras.add_argument('--random_seed', type=int, default=1)
    paras.add_argument('--gpu_device', type=int, default=0)
    paras.add_argument('--class_num', type=int, default=3)
    paras.add_argument('--batch_size', type=int, default=2)
    paras.add_argument('--epochs', type=int, default=100)
    paras.add_argument('--img_size', type=list, default=[96, 96])
    paras.add_argument('--bags_len', type=int, default=1025)
    paras.add_argument('--num_workers', type=int, default=16)
    paras.add_argument('--worker_time_out', type=int, default=0)
    paras.add_argument('--data_parallel', type=bool, default=True)
    paras.add_argument('--run_mode', type=str, default='train')
    paras.add_argument('--parallel_gpu_ids', type=list, default=[0, 1])
    paras.add_argument('--train_read_path', type=str,
                default=r'/data/HP_Projects/TicMIL/Datasets/Cervix/New_Bags/Train')
    paras.add_argument('--test_read_path', type=str,
                default=r'/data/HP_Projects/TicMIL/Datasets/Cervix/New_Bags/Test')
    paras.add_argument('--val_read_path', type=str,
                default=r'/data/HP_Projects/TicMIL/Datasets/Cervix/New_Bags/Test')

    paras.add_argument('--weights_save_path', type=str,
                default=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Cervix_WSI_Baseline_20240817.pth')
    paras.add_argument('--test_weights_path', type=str,
                default=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Cervix_WSI_SOTA_20240813.pth')


    ### Parallel save
    paras.add_argument('--weights_save_feature', type=str,
                        default=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Baseline_Feature(using PE).pth')
    paras.add_argument('--weights_save_head', type=str,
                        default=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Baseline_Head(using PE).pth')

    ### Parallel test
    paras.add_argument('--test_weights_feature', type=str,
                        default=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Baseline_Feature(using PE).pth')
    paras.add_argument('--test_weights_head', type=str,
                        default=r'/data/HP_Projects/TicMIL/Weights_Result_Text/WSI/Baseline_Head(using PE).pth')

    ### Pretrained
    paras.add_argument('--pretrained_weights_path', type=str,
                default=r'/data/HP_Projects/TicMIL/Weights/SwinT/swin_tiny_patch4_window7_224_22k.pth')

    args = paras.parse_args()
    setup_seed(args.random_seed)

    ########################## reading datas and processing datas #########################
    print('########################## reading datas and processing datas #########################')
    train_data = Read_MIL_Datasets(read_path=args.train_read_path ,img_size=args.img_size, bags_len=args.bags_len)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              timeout=args.worker_time_out)

    test_data = Read_MIL_Datasets(read_path=args.test_read_path, img_size=args.img_size, bags_len=args.bags_len)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              timeout=args.worker_time_out)

    val_data = Read_MIL_Datasets(read_path=args.val_read_path, img_size=args.img_size, bags_len=args.bags_len)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              timeout=args.worker_time_out)
    print('train_data:', '\n', train_data, '\n')

    ########################## creating models and visuling models #########################
    print('########################## creating models and visuling models #########################')
    swinT_base = SwinTransformer(img_size=args.img_size[0], patch_size=4, in_chans=3, num_classes=args.class_num,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=3, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False)

    checkpoint = torch.load(args.pretrained_weights_path, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = swinT_base.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            #logger.warning(f"Error in loading {k}, passing......")
            pass
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = swinT_base.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            #logger.warning(f"Error in loading {k}, passing......")
            pass
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = swinT_base.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            #logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(swinT_base.head.bias, 0.)
            torch.nn.init.constant_(swinT_base.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']

    swinT_base.load_state_dict(state_dict, strict=False)

    nn.init.trunc_normal_(swinT_base.head.weight, std=.02)
    print(swinT_base.layers[0].blocks[0].mlp.fc2.weight)

    ### creating a SPE-MIL model
    if args.data_parallel == False:
        ticmil_net = TicMIL_for_ablation(base_model=swinT_base, class_num=args.class_num)
        with torch.no_grad():
            print('########################## SwinT_summary #########################')
            summary(ticmil_net, (3, 96, 96), device='cpu')
            print('\n', '########################## SwinT_net #########################')
            print(ticmil_net, '\n')

            print(ticmil_net(torch.zeros(10, 3, 96, 96)).shape)
        ticmil_net = ticmil_net.cuda(args.gpu_device)
    else:
        ticmil_feature = TicMIL_Parallel_Feature_for_ablation(base_model=swinT_base)
        ticmil_head = TicMIL_Parallel_Head_for_ablation(base_model=swinT_base, class_num=args.class_num,
                                                        batch_size=args.batch_size, bags_len=args.bags_len)
        with torch.no_grad():
            print('########################## SwinT_summary #########################')
            summary(ticmil_feature, (3, 96, 96), device='cpu')
            summary(ticmil_head, (768,), device='cpu')
            print('\n', '########################## SwinT_net #########################')
            print(ticmil_feature, '\n')
            print(ticmil_head, '\n')
        ticmil_feature = ticmil_feature.cuda()
        ticmil_feature = DataParallel(ticmil_feature, device_ids=args.parallel_gpu_ids)
        ticmil_head = ticmil_head.cuda()
        #ticmil_head = DataParallel(ticmil_head, device_ids=[0, 1])


    ########################## fitting models and testing models #########################
    if args.run_mode == 'train':
        print('########################## fitting models #########################')
        if args.data_parallel == False:
            training_for_baseline_parallel(mil_net=ticmil_net, train_loader=train_loader, val_loader=val_loader,
                           proba_mode=args.proba_mode, test_loader=test_loader, lr_fn='vit', epoch=args.epochs,
                           gpu_device=args.gpu_device, weight_path=args.weights_save_path,
                           data_parallel=args.data_parallel, proba_value=args.proba_value,
                           class_num=args.class_num, bags_stat=args.bags_stat)
        else:
            training_for_baseline_parallel(mil_feature=ticmil_feature, mil_head=ticmil_head, train_loader=train_loader,
                                    val_loader=val_loader, test_loader=test_loader,
                                    lr_fn='vit', epoch=args.epochs, gpu_device=args.gpu_device,
                                    weight_path=args.weights_save_feature,
                                    bags_len=args.bags_len, batch_size=args.batch_size,
                                    weight_head_path=args.weights_save_head)

    if args.run_mode == 'test':
        print('########################## testing function #########################')
        if args.data_parallel == False:
            larynx_weight = torch.load(args.test_weights_path, map_location='cuda:0')
            ticmil_net.load_state_dict(larynx_weight, strict=True)
            testing_for_parallel_baseline(test_model=ticmil_net, train_loader=test_loader, val_loader=val_loader,
                              proba_value=args.proba_value, test_loader=test_loader, gpu_device=args.gpu_device,
                              out_mode=None, proba_mode=args.proba_mode, class_num=args.class_num,
                              roc_save_path=args.roc_save_path, bags_stat=args.bags_stat,
                              bag_relations_path=args.bag_relations_path)
        elif args.data_parallel == True:
            head_weight = torch.load(args.test_weights_head, map_location='cuda:0')
            feature_weight = torch.load(args.test_weights_feature, map_location='cuda:0')
            ticmil_feature.load_state_dict(feature_weight, strict=True)
            ticmil_head.load_state_dict(head_weight, strict=True)
            testing_for_parallel_baseline(mil_feature=ticmil_feature, mil_head=ticmil_head, train_loader=train_loader,
                                   data_parallel=args.data_parallel,
                                    batch_size=args.batch_size,
                                   bags_len=args.bags_len, val_loader=val_loader, test_loader=test_loader)





