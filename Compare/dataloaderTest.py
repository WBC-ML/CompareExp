import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2
import torchvision.transforms.functional as TF


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# seed_torch(1143)


def populate_train_list(orig_images_path, hazy_images_path, perc=11):
    train_list = []
    val_list = []

    image_list_haze = glob.glob(hazy_images_path + "*.jpg")

    tmp_dict = {}

    for image in image_list_haze:
        # image_this = image.split("/")[-1]
        # image_this = image.split("\\")[-1]
        # key = image.split("_")[0] + "_" + image.split("_")[1] + ".jpg"
        key = image # orig_images_path + image_this.split('/')[-1].split("_")[0] + "_" + image_this.split("_")[1] + ".jpg"
        # key = image  # .split('_')[0] + '.jpg'
        if key in tmp_dict.keys():
            tmp_dict[key].append(image)
        else:
            tmp_dict[key] = []
            tmp_dict[key].append(image)

    train_keys = []
    val_keys = []

    len_keys = len(tmp_dict.keys())
    for i in range(len_keys):
        if i < len_keys * perc / 10:
            train_keys.append(list(tmp_dict.keys())[i])
        else:
            val_keys.append(list(tmp_dict.keys())[i])

    for key in list(tmp_dict.keys()):

        if key in train_keys:
            for hazy_image in tmp_dict[key]:
                # train_list.append([orig_images_path + key, hazy_images_path + hazy_image])
                train_list.append([key, hazy_image])

        else:
            for hazy_image in tmp_dict[key]:
                val_list.append([key, hazy_image])
                # val_list.append([orig_images_path + key, hazy_images_path + hazy_image])
        # val_list.append([key, hazy_image])

    random.shuffle(train_list)
    random.shuffle(val_list)

    return train_list, val_list

class dehazing_loader(data.Dataset):

    def __init__(self, orig_images_path, hazy_images_path, mode='train', perc=11, ps=256):
        self.mode = mode  # if test then perc <= 0
        self.train_list, self.val_list = populate_train_list(orig_images_path, hazy_images_path, perc=perc)
        self.ps = ps

        if mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total testing examples:", len(self.val_list))

    def __getitem__(self, index):

        data_orig_path, data_hazy_path = self.data_list[index]

        data_orig = Image.open(data_orig_path).convert('RGB')
        data_hazy = Image.open(data_hazy_path).convert('RGB')

        data_orig = data_orig.resize((self.ps, self.ps), Image.BICUBIC)
        data_hazy = data_hazy.resize((self.ps, self.ps), Image.BICUBIC)

        #################
        # Crop patch
        # ps = 256
        # inp_img = TF.to_tensor(data_hazy)
        # tar_img = TF.to_tensor(data_orig)
        #
        # hh, ww = tar_img.shape[1], tar_img.shape[2]
        #
        # rr = random.randint(0, hh - ps)
        # cc = random.randint(0, ww - ps)
        #
        # data_hazy = inp_img[:, rr:rr + ps, cc:cc + ps]
        # data_orig = tar_img[:, rr:rr + ps, cc:cc + ps]

        #################
        if self.mode == 'train':
            data_orig, data_hazy = self.aug(data_orig, data_hazy)
        else:
            data_orig, data_hazy = TF.to_tensor(data_orig), TF.to_tensor(data_hazy)

        return data_orig, data_hazy

    def __len__(self):
        return len(self.data_list)

    def aug(self, inp_img, tar_img):
        # ps = 256
        aug = random.randint(0, 2)
        if aug == 1:
            inp_img = TF.adjust_gamma(inp_img, 1)
            tar_img = TF.adjust_gamma(tar_img, 1)

        aug = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
            inp_img = TF.adjust_saturation(inp_img, sat_factor)
            tar_img = TF.adjust_saturation(tar_img, sat_factor)

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        # hh, ww = tar_img.shape[1], tar_img.shape[2]
		#
        # rr = random.randint(0, hh - ps)
        # cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # # Crop patch
        # inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        # tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        return inp_img, tar_img

class dehazing_loader_unsupervisied(data.Dataset):

    def __init__(self, orig_images_path, hazy_images_path, mode='train', perc=11, ps=256):
        self.mode = mode  # if test then perc <= 0
        self.train_list, self.val_list = populate_train_list(orig_images_path, hazy_images_path, perc=perc)
        self.ps = ps
        self.hazy_path = hazy_images_path

        if mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total testing examples:", len(self.val_list))

    def __getitem__(self, index):

        data_orig_path, _ = self.data_list[index]
        data_hazy_path = os.path.join(self.hazy_path, self.data_list[random.randint(0, self.__len__()-1)][1])

        data_orig = Image.open(data_orig_path).convert('RGB')
        data_hazy = Image.open(data_hazy_path).convert('RGB')

        data_orig = data_orig.resize((self.ps, self.ps), Image.BICUBIC)
        data_hazy = data_hazy.resize((self.ps, self.ps), Image.BICUBIC)

        #################
        # Crop patch
        # ps = 256
        # inp_img = TF.to_tensor(data_hazy)
        # tar_img = TF.to_tensor(data_orig)
        #
        # hh, ww = tar_img.shape[1], tar_img.shape[2]
        #
        # rr = random.randint(0, hh - ps)
        # cc = random.randint(0, ww - ps)
        #
        # data_hazy = inp_img[:, rr:rr + ps, cc:cc + ps]
        # data_orig = tar_img[:, rr:rr + ps, cc:cc + ps]

        #################
        if self.mode == 'train':
            data_orig, data_hazy = self.aug(data_orig, data_hazy)
        else:
            data_orig, data_hazy = TF.to_tensor(data_orig), TF.to_tensor(data_hazy)

        return data_orig, data_hazy

    def __len__(self):
        return len(self.data_list)

    def aug(self, inp_img, tar_img):
        # ps = 256
        aug = random.randint(0, 2)
        if aug == 1:
            inp_img = TF.adjust_gamma(inp_img, 1)
            tar_img = TF.adjust_gamma(tar_img, 1)

        aug = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
            inp_img = TF.adjust_saturation(inp_img, sat_factor)
            tar_img = TF.adjust_saturation(tar_img, sat_factor)

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        # hh, ww = tar_img.shape[1], tar_img.shape[2]
		#
        # rr = random.randint(0, hh - ps)
        # cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # # Crop patch
        # inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        # tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        return inp_img, tar_img

class dehazing_loader_wival(data.Dataset):

    def __init__(self, orig_images_path, hazy_images_path, mode='train', perc=11, ps=256):
        self.mode = mode  # if test then perc <= 0
        self.train_list, self.val_list = populate_train_list(orig_images_path, hazy_images_path, perc=perc)
        self.ps = ps

        if mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total testing examples:", len(self.val_list))

    def __getitem__(self, index):

        data_orig_path, data_hazy_path = self.data_list[index]

        data_orig = Image.open(data_orig_path).convert('RGB')
        data_hazy = Image.open(data_hazy_path).convert('RGB')

        data_orig = data_orig.resize((self.ps, self.ps), Image.BICUBIC)
        data_hazy = data_hazy.resize((self.ps, self.ps), Image.BICUBIC)

        #################
        # Crop patch
        # ps = 256
        # inp_img = TF.to_tensor(data_hazy)
        # tar_img = TF.to_tensor(data_orig)
        #
        # hh, ww = tar_img.shape[1], tar_img.shape[2]
        #
        # rr = random.randint(0, hh - ps)
        # cc = random.randint(0, ww - ps)
        #
        # data_hazy = inp_img[:, rr:rr + ps, cc:cc + ps]
        # data_orig = tar_img[:, rr:rr + ps, cc:cc + ps]

        #################
        if self.mode == 'train':
            data_orig, data_hazy = self.aug(data_orig, data_hazy)
        else:
            data_orig, data_hazy = TF.to_tensor(data_orig), TF.to_tensor(data_hazy)

        return data_orig, data_hazy

    def __len__(self):
        return len(self.data_list)

    def aug(self, inp_img, tar_img):
        # ps = 256
        aug = random.randint(0, 2)
        if aug == 1:
            inp_img = TF.adjust_gamma(inp_img, 1)
            tar_img = TF.adjust_gamma(tar_img, 1)

        aug = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
            inp_img = TF.adjust_saturation(inp_img, sat_factor)
            tar_img = TF.adjust_saturation(tar_img, sat_factor)

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        # hh, ww = tar_img.shape[1], tar_img.shape[2]
		#
        # rr = random.randint(0, hh - ps)
        # cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # # Crop patch
        # inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        # tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        return inp_img, tar_img
