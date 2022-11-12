import torch
import torch.utils.data as data
import os
import imageio.v2
import numpy as np
from TermProjectDataAugmentation import random_crop_and_pad_image_and_labels, random_flip

# 因為需要指定哪一張是input image,哪一張是reference image,以及做preprocessing,所以自己定義一個Data Set
class DataSet(data.Dataset):
    def __init__(self, path="data/vimeo_septuplet/test.txt", im_height=256, im_width=256):
        self.image_input_list, self.image_ref_list = self.get_vimeo(filefolderlist=path)  # 得到input image file name和他的reference image file name
        self.im_height = im_height
        self.im_width = im_width

        print("dataset find image: ", len(self.image_input_list))

    def get_vimeo(self, rootdir="data/vimeo_septuplet/sequences/", filefolderlist="data/vimeo_septuplet/test.txt"):
        with open(filefolderlist) as f:
            fname = f.readlines()

        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(fname):
            input_fname = os.path.join(rootdir, line.rstrip())  # line.rstrip() => 將後面的'\n'移除
            fns_train_input += [input_fname]
            refnumber = int(input_fname[-5:-4]) - 2
            refname = input_fname[0:-5] + str(refnumber) + '.png'  # 得到reference image file name
            fns_train_ref += [refname]

        return fns_train_input, fns_train_ref

    def __getitem__(self, index):
        # imread()回傳array型態,所以可以用astype()將資料型態轉成float32
        input_image = imageio.v2.imread(self.image_input_list[index])
        ref_image = imageio.v2.imread(self.image_ref_list[index])

        #cur_image = torch.from_numpy(cur_image).float()
        #ref_image = torch.from_numpy(ref_image).float()
        #cur_image = torch.div(cur_image, 255.0)
        #ref_image = torch.div(ref_image, 255.0)
        #cur_image = torch.transpose(cur_image, 2, 0)
        #cur_image = torch.transpose(cur_image, 1, 2)
        #ref_image = torch.transpose(ref_image, 2, 0)
        #ref_image = torch.transpose(ref_image, 1, 2)

        input_image = input_image.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0

        # [h, w, c] => [c, h, w]
        # 想成將RGB三個拆開處理, 每一張(w x h) image都只有R、G、B值
        input_image = input_image.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)

        # 將numpy的array轉成torch.tensor
        input_image = torch.from_numpy(input_image).float()
        ref_image = torch.from_numpy(ref_image).float()

        # augmentation => 替current image和reference image增加變化
        #input_image, ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image, [self.im_height, self.im_width])
        #input_image, ref_image = random_flip(input_image, ref_image)

        return input_image, ref_image

    def __len__(self):
        return len(self.image_input_list)




