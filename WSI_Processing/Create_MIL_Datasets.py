############################# Creating Multiple instance learing datasets ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: creating & reconstructing MIL file path and datasets (including instances and bags).
import cv2
########################## API Section #########################
import numpy as np
from skimage import io, transform, color
from cv2 import imread
import os
import natsort
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = \
    '/data/HP_Projects/Anaconda/envs/ISBI_ENV/lib/python3.9/site-packages/PyQt5/Qt5/plugins'


########################## Create_MIL_Datasets #########################
class Create_MIL_Datasets():
    def __init__(self, h_step = 2, w_step = 2, read_path = None, save_path = None, resize_shape = (96, 96),
                 resize_stat = False, del_white_map = False, del_white_thod = 0.9):
        self.h_step = h_step
        self.w_step = w_step
        self.read_path = read_path
        self.save_path = save_path
        self.resize_shape = resize_shape
        self.resize_stat = resize_stat
        self.del_white_map = del_white_map
        self.del_white_thod = del_white_thod

    def single_img_mode(self):
        test_img = io.imread(self.read_path)
        file_name = self.read_path[-8:-4]
        search_str = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_']
        file_name = [i for i in file_name if i in search_str]
        new_file_name = ''
        for i in file_name:
            new_file_name += i

        os.makedirs(self.save_path + '//' + new_file_name)

        h_len = test_img.shape[0]
        w_len = test_img.shape[1]
        w_step_len = int(w_len / self.w_step)
        h_step_len = int(h_len / self.h_step)

        for i in range(self.h_step):
            for j in range(self.w_step):
                inter_img = test_img[(i * h_step_len):((i + 1) * h_step_len),
                            (j * w_step_len):((j + 1) * w_step_len), :]
                if self.resize_stat == True:
                    inter_img = cv2.resize(inter_img, self.resize_shape)
                #if self.del_white_map == True:
                #    if np.mean(color.rgb2gray(inter_img)) < self.del_white_thod:
                #        inter_img = np.zeros_like(inter_img)

                if np.mean(color.rgb2gray(inter_img)) < self.del_white_thod:
                    io.imsave(self.save_path + '//' + new_file_name + '//' + str(((i * self.w_step) + j)) + '.jpg',
                            inter_img.astype('uint8'))
                print(str(((i * self.w_step) + j)))

    def files_mode(self):
        img_name_list = natsort.natsorted(os.listdir(self.read_path), alg=natsort.ns.PATH)
        for img_name in img_name_list:
            test_img = imread(self.read_path + r'//' + img_name)
            file_name = img_name
            search_str = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_']
            file_name = [i for i in file_name if i in search_str]
            new_file_name = ''
            for i in file_name:
                new_file_name += i

            os.makedirs(self.save_path + '//' + new_file_name)

            h_len = test_img.shape[0]
            w_len = test_img.shape[1]
            w_step_len = int(w_len / self.w_step)
            h_step_len = int(h_len / self.h_step)

            for i in range(self.h_step):
                for j in range(self.w_step):
                    inter_img = test_img[(i * h_step_len):((i + 1) * h_step_len),
                                (j * w_step_len):((j + 1) * w_step_len), :]
                    if self.resize_stat == True:
                        inter_img = cv2.resize(inter_img, self.resize_shape)
                    #if self.del_white_map == True:
                    #    if np.mean(color.rgb2gray(inter_img)) > self.del_white_thod:
                    #        inter_img = np.zeros_like(inter_img)
                    if np.mean(color.rgb2gray(inter_img)) < self.del_white_thod:
                        io.imsave(self.save_path + '//' + new_file_name + '//' + str(((i * self.w_step) + j)) + '.jpg',
                                inter_img.astype('uint8'))
                    print(img_name + '  ' + str(((i * self.w_step) + j)))


########################## Reconst_OrgIMG_Datasets #########################
class Reconst_OrgIMG_Datasets():
    def __init__(self, read_path = None, save_path = None, reconst_w_len = 10):
        self.read_path = read_path
        self.save_path = save_path
        self.reconst_w_len = reconst_w_len

    def single_IMG_mode(self):
        img_name = natsort.natsorted(os.listdir(self.read_path), alg=natsort.ns.PATH)
        img_list = [io.imread(self.read_path + '//' + i) for i in img_name]
        step_len = int(len(img_name) /self.reconst_w_len)
        new_sub_img = [np.concatenate(img_list[(self.reconst_w_len * i): ((i + 1) * self.reconst_w_len)], axis=1)
                       for i in range(step_len)]
        new_img = np.concatenate(new_sub_img, axis=0)

        plt.figure(1)
        plt.imshow(new_img)
        plt.show()

        #io.imsave(save_path, new_img)

    def files_mode(self):
        files_name_list = natsort.natsorted(os.listdir(self.read_path), alg=natsort.ns.PATH)
        for file_name in files_name_list:
            img_name = natsort.natsorted(os.listdir(self.read_path + '//' + file_name), alg=natsort.ns.PATH)
            img_list = [io.imread(self.read_path + '//' + file_name + '//' + i) for i in img_name]
            step_len = int(len(img_name) / self.reconst_w_len)
            new_sub_img = [np.concatenate(img_list[(self.reconst_w_len * i): ((i + 1) * self.reconst_w_len)], axis=1)
                           for i in range(step_len)]
            new_img = np.concatenate(new_sub_img, axis=0)

            io.imsave(self.save_path + '//' + file_name + '.jpg', new_img)
            print(file_name + '.jpg')


########################## testing demo main #########################
if __name__ == '__main__':
    read_path = r'/data/HP_Projects/SPE_MIL/Datasets/WSI_Larynx_China/WSI_IMG_961_96x96_Patients_ColorN/Train/I/0164'
    save_path = r'/data/HP_Projects/SPE_MIL/Datasets/WSI_Larynx_China/WSI_IMG_961_96x96_Patients_ColorN/Train/I'
    #create_mil_obj = Create_MIL_Datasets(read_path = read_path, save_path = save_path,
    #                                   w_step= 50 , h_step = 50, del_white_map=True, del_white_thod=0.8,
    #                                     resize_stat=True, resize_shape=(96, 96))
    #create_mil_obj.files_mode()

    rec_obj = Reconst_OrgIMG_Datasets(read_path = read_path, save_path = save_path, reconst_w_len = 31)
    rec_obj.single_IMG_mode()




