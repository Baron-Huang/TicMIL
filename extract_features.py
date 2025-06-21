import torch
import argparse
from Utils.Read_MIL_Datasets import Read_MIL_Datasets
from torch.utils.data import DataLoader
from Models.SwinT_models.models.swin_transformer import SwinTransformer
from torch import nn
from Utils.load_pretrained_weight import load_swint_pretrained
from torchsummary import summary
import numpy as np

def other_paras_cfg():
    ####other paras
    paras = argparse.ArgumentParser(description='Other parameters')
    paras.add_argument('--bags_len', type=int, default=961)
    paras.add_argument('--img_size', type=list, default=[96, 96])
    paras.add_argument('--num_workers', type=int, default=16)
    paras.add_argument('--batch_size', type=int, default=1)
    paras.add_argument('--loader_random', type=bool, default=False)
    paras.add_argument('--pretrained_weights_path', type=str,
                       default='/data/HP_Projects/TicMIL/Weights/SwinT/swin_tiny_patch4_window7_224_22k.pth')
    return paras.parse_args()

def base_paras_cfg():
    ####base model paras
    paras = argparse.ArgumentParser(description='Base model parameters')
    paras.add_argument('--embed_dim', type=int, default=96)
    paras.add_argument('--patch_size', type=int, default=4)
    paras.add_argument('--in_chans', type=int, default=3)
    paras.add_argument('--num_classes', type=int, default=3)
    paras.add_argument('--depths', type=list, default=[2, 2, 6, 2])
    paras.add_argument('--num_heads', type=list, default=[3, 6, 12, 24])
    paras.add_argument('--mlp_ratio', type=int, default=4.)
    paras.add_argument('--drop_rate', type=float, default=0.)
    paras.add_argument('--attn_drop_rate', type=float, default=0.)
    paras.add_argument('--drop_path_rate', type=float, default=0.1)
    paras.add_argument('--qkv_bias', type=bool, default=True)
    paras.add_argument('--ape', type=bool, default=False)
    paras.add_argument('--patch_norm', type=bool, default=True)
    paras.add_argument('--use_checkpoint', type=bool, default=False)
    paras.add_argument('--fused_window_process', type=bool, default=False)
    paras.add_argument('--window_size', type=int, default=3)

    return paras.parse_args()

class SwinT_feats_encoder(nn.Module):
    def __init__(self, base_model=None, final_avp_kernel_size=9, final_avp_tride=9):
        super(SwinT_feats_encoder, self).__init__()
        self.layers_0 = base_model.layers[0]
        self.layers_1 = base_model.layers[1]
        self.layers_2 = base_model.layers[2]
        self.layers_3 = base_model.layers[3]
        self.patch_embed = base_model.patch_embed
        self.pos_drop = base_model.pos_drop
        self.norm = base_model.norm
        self.avgp = nn.AvgPool1d(kernel_size=final_avp_kernel_size, stride=final_avp_tride)

    def forward(self, x):
        y = self.patch_embed(x)
        y = self.pos_drop(y)
        y = self.layers_0(y)
        y = self.layers_1(y)
        y = self.layers_2(y)
        y = self.layers_3(y)
        y = self.norm(y)
        y = self.avgp(y.permute(0, 2, 1))
        y = torch.reshape(y, (y.shape[0], y.shape[1]))
        return y


def extract_features_for_swint(base_args=None, other_args=None, save_path=None, read_path = None):
    train_data = Read_MIL_Datasets(read_path=read_path ,img_size=other_args.img_size, bags_len=other_args.bags_len)
    train_loader = DataLoader(train_data, batch_size=other_args.batch_size, shuffle=other_args.loader_random,
                              num_workers=other_args.num_workers)

    swinT_base = SwinTransformer(img_size=other_args.img_size[0], patch_size=base_args.patch_size, in_chans=base_args.in_chans,
                 num_classes=base_args.num_classes, embed_dim=base_args.embed_dim, depths=base_args.depths,
                 num_heads=base_args.num_heads, window_size=base_args.window_size, mlp_ratio=4., qkv_bias=base_args.qkv_bias,
                 qk_scale=None, drop_rate=base_args.drop_rate, attn_drop_rate=base_args.attn_drop_rate,
                 drop_path_rate=base_args.drop_path_rate, norm_layer=nn.LayerNorm, ape=base_args.ape,
                 patch_norm=base_args.patch_norm, use_checkpoint=base_args.use_checkpoint,
                 fused_window_process=base_args.fused_window_process)

    ##### load pretrained weights
    checkpoint = torch.load(other_args.pretrained_weights_path, map_location='cpu')
    state_dict = checkpoint['model']

    load_swint_pretrained(state_dict=state_dict, swinT_base=swinT_base)

    swinT_base.load_state_dict(state_dict, strict=False)

    nn.init.trunc_normal_(swinT_base.head.weight, std=.02)

    feats_extract_model = SwinT_feats_encoder(base_model=swinT_base)

    feats_extract_model.eval()
    with torch.no_grad():
        print('########################## feats_extract_model_summary #########################')
        summary(feats_extract_model, (3, 96, 96), device='cpu')
        print(feats_extract_model(torch.randn((3, 3, 96, 96), device='cpu')).shape)
    #print(feats_extract_model)

    with torch.no_grad():
        count_save = 0
        for org_data, label in train_loader:
            count_save += 1
            org_data = torch.reshape(org_data, (org_data.shape[1], org_data.shape[2], org_data.shape[3], org_data.shape[4]))
            swint_feats_mat = feats_extract_model(org_data)
            swint_feats_mat = swint_feats_mat.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            np.save(save_path + '/Feats/' + str(count_save) + '.npy', swint_feats_mat)
            np.save(save_path + '/Labels/' + str(count_save) + '.npy', label)
            print(count_save)

    print(swinT_base.layers[0].blocks[0].mlp.fc2.weight)

    return 0


if __name__ == '__main__':
    new_tensor = extract_features_for_swint(base_args=base_paras_cfg(), other_args=other_paras_cfg(),
                    save_path='/data/HP_Projects/TicMIL/Datasets/Cervix/Feats_Non_PE/Train',
                    read_path=r'/data/HP_Projects/TicMIL/Datasets/Cervix/Cervix_MIL_961/Train')

    #test_label = np.load('/media/hp/765AE65C5AE6191F/HP_Files/PacMIL/Datasets/Cervix_pretrained_feats/Test/Labels/126.npy',
                            #allow_pickle=True)
    #print(test_label)

