import natsort
from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np


class Read_Feats_Datasets(Dataset):
    def __init__(self, data_path = None, label_path = None, data_size = [85, 768], label_size = [1, 1]):
        super(Read_Feats_Datasets, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.data_list = natsort.natsorted(os.listdir(self.data_path), alg=natsort.ns.PATH)
        self.label_list = natsort.natsorted(os.listdir(self.label_path), alg=natsort.ns.PATH)
        self.data_size = data_size
        self.label_size = label_size


    def __getitem__(self, item):

        data_inter = np.load(self.data_path + '/' + self.data_list[item])
        data_inter_tensor = torch.tensor(data_inter).reshape(self.data_size[0], self.data_size[1])

            ### read labels
        label_inter = np.load(self.label_path + '/' + self.label_list[item])

        return data_inter_tensor, label_inter

    def __len__(self):
        return len(self.label_list)


if __name__ == '__main__':
    new_dataset = Read_Feats_Datasets(
                    data_path = r'/media/hp/765AE65C5AE6191F/HP_Files/PacMIL/Datasets/Cervix_pretrained_feats/Train/Feats',
                    label_path = r'/media/hp/765AE65C5AE6191F/HP_Files/PacMIL/Datasets/Cervix_pretrained_feats/Train/Labels')
    print(new_dataset)

    new_loader = DataLoader(dataset=new_dataset, batch_size=2, shuffle=False, num_workers=16)

    for i, j in new_loader:
        print(i.shape)
        print(j)





