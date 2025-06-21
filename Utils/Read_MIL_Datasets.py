############################# reading Multiple instance learing datasets for tensor training ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: read MIL file datasets (including instances and bags).

########################## API Section #########################
import natsort
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
import numpy as np
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
#from Setup_Seed import setup_seed


########################## Read_MIL_Datasets #########################
class Read_MIL_Datasets(Dataset):
    def __init__(self, read_path = None, img_size = [96, 96], bags_len = 100):
        super(Read_MIL_Datasets, self).__init__()
        self.read_path = read_path
        self.img_size = img_size
        self.transform_rgb = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()
                                                    , transforms.Normalize(mean=0.5, std=0.5)])
        self.bags_img_path, self.bags_label_path = self.form_bags()
        self.bags_len = bags_len


    def _default_loader(path):
        return (Image.open(path))


    def form_bags(self):
        bags_img_list_inter = []
        class_path = natsort.natsorted(os.listdir(self.read_path), alg=natsort.ns.PATH)
        for class_files_name in class_path:
            bags_name_list = natsort.natsorted(os.listdir(self.read_path + r'//' + class_files_name),
                                               alg=natsort.ns.PATH)
            bags_img_list_inter.append(
                [self.read_path + r'//' + class_files_name + r'//' + bags_name for bags_name in bags_name_list])

        bag_label_npa = np.zeros((1))
        for label_num, label_search in enumerate(bags_img_list_inter):
            bag_label_npa = np.concatenate((bag_label_npa, np.zeros(len(label_search)) + label_num))
        bag_img_list = []
        for i in bags_img_list_inter:
            bag_img_list += i

        return bag_img_list, bag_label_npa[1:]


    def __getitem__(self, item):
        bags_label_arr = self.bags_label_path[item]
        bags_label_tensor = torch.tensor(bags_label_arr, dtype=torch.long)
        bags_img_batch_path = self.bags_img_path[item]
        transforms_rgb = transforms.Compose([transforms.Resize(self.img_size), transforms.ToTensor()
                                                , transforms.Normalize(mean=0.5, std=0.5)])

        bags_img_name_list = natsort.natsorted(os.listdir(bags_img_batch_path), alg=natsort.ns.PATH)
        if len(bags_img_name_list) > self.bags_len:
            img_ord = 0
            img_word_sum = torch.zeros((self.bags_len, 3, self.img_size[0], self.img_size[1]))
            for x_1_name in bags_img_name_list[int((len(bags_img_name_list) - self.bags_len) / 2) + 1:
                                        (len(bags_img_name_list)-int((len(bags_img_name_list) - self.bags_len) / 2))]:
                img_word = Image.open(bags_img_batch_path + r'//' + x_1_name)
                img_word_sum[img_ord, :, :, :] = transforms_rgb(img_word)
                img_ord += 1
        else:
            img_word_sum = torch.zeros((len(bags_img_name_list), 3, self.img_size[0], self.img_size[1]))
            for img_ord, x_1_name in enumerate(bags_img_name_list):
                img_word = Image.open(bags_img_batch_path + r'//' + x_1_name)
                img_word_sum[img_ord, :, :, :] = transforms_rgb(img_word)

        return img_word_sum, bags_label_tensor


    def __len__(self):
        return len(self.bags_img_path)


########################## demo testing main #########################
if __name__ == '__main__':
    #setup_seed(0)
    read_path = r'E:\SPE_MIL\Datasets\MIL_Larynx_china\MIL_IMG\Train'
    read_mil_obj = Read_MIL_Datasets(read_path=read_path)

    mil_train_loader = DataLoader(dataset=read_mil_obj, batch_size=1, shuffle=False, worker_init_fn=1)
    for batch_ord, (i, j) in enumerate(mil_train_loader):
        if batch_ord >= 0:
            break
        print(batch_ord)
    test_img = i[15, 28, :, :, :].permute(1, 2, 0)
    test_img = test_img.numpy()

    plt.figure()
    plt.imshow(test_img)
    plt.show()

    pass
