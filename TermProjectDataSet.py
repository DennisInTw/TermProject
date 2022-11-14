import torch
import torch.utils.data as data
import os
import imageio.v2
import numpy as np
from TermProjectDataAugmentation import random_crop_and_pad_image_and_labels, random_flip
from flowlib import read_flow
import torchvision
import random
import skimage
from typing import Union, Tuple


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        o = args
        for t in self.transforms:
            o = t(*o)
        return o

# 對兩張image做normalization, 加快convergence
class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img1, img2, gt_optical_flow):
        img1 = torchvision.transforms.functional.normalize(img1, self.mean, self.std)
        img2 = torchvision.transforms.functional.normalize(img2, self.mean, self.std)

        return img1, img2, gt_optical_flow

# 對兩張image、ground truth optical flow做rotation
class RandomRotate(object):
    def __init__(self, minmax_angle: Union[Tuple[int, int], int]):
        self.rotate_angle = minmax_angle

        if isinstance(minmax_angle, int):
            self.rotate_angle = (-minmax_angle, minmax_angle)

    def __call__(self, img1, img2, gt_optical_flow):
        angle = random.randint(*self.rotate_angle)
        img1 = torchvision.transforms.functional.rotate(img1, angle)
        img2 = torchvision.transforms.functional.rotate(img2, angle)
        gt_optical_flow = skimage.transform.rotate(gt_optical_flow, angle)

        return img1, img2, gt_optical_flow

class RandomFlip(object):
    def __init__(self, flip_vertical=True, flip_horizontal=True):
        self.flip_v = flip_vertical
        self.flip_h = flip_horizontal

    def __call__(self, img1, img2, gt_optical_flow):
        # 注意 : img1、img2、gt_optical_flow的data type和dimension
        # img1、img2都是tensor type => [channel, height, width]
        # gt_optical_flow是numpy.ndarray => [height, width, channel]

        if (self.flip_v is True) and (random.randint(0, 1) == 1):
            img1 = torch.flip(img1, [1])
            img2 = torch.flip(img2, [1])
            gt_optical_flow = torch.from_numpy(gt_optical_flow)
            gt_optical_flow = torch.flip(gt_optical_flow, [0])

        if (self.flip_h is True) and (random.randint(0, 1) == 1):
            img1 = torch.flip(img1, [0])
            img2 = torch.flip(img2, [0])

            # 如果vertical沒有flip, 則需要在horizontal這裡將gt_optical_flow轉成tensor type
            if torch.is_tensor(gt_optical_flow) is False:
                gt_optical_flow = torch.from_numpy(gt_optical_flow)
            gt_optical_flow = torch.flip(gt_optical_flow, [1])

        # 最後將gt_optical_flow轉回numpy.ndarray,不然會有問題,不過我還不清楚為什麼??
        if torch.is_tensor(gt_optical_flow) is True:
            gt_optical_flow = gt_optical_flow.detach().cpu().numpy()

        return img1, img2, gt_optical_flow



# 因為需要指定哪一張是input image,哪一張是reference image,以及做preprocessing,所以自己定義一個Data Set
class DataSet(data.Dataset):
    def __init__(self, filelist, im_height, im_width, transform):
        self.image_input_list, self.image_ref_list, self.ground_truth_list = self.get_frame(framelist=filelist)  # 得到input image file name和他的reference image file name
        self.im_height = im_height
        self.im_width = im_width
        self.transform = transform

        print("dataset find image: ", len(self.image_input_list))

    def get_frame(self, framelist, rootdir="data/"):
        with open(framelist) as f:
            fname = f.readlines()

        input_frames = []
        reference_frames = []
        ground_truth_frame_flo = []


        for n, line in enumerate(fname):
            prefix_filename = os.path.join(rootdir, line.rstrip())  # line.rstrip() => 將後面的'\n'移除
            #input_fname = os.path.join(rootdir, line.rstrip())  # line.rstrip() => 將後面的'\n'移除
            #input_frames += [input_fname]
            #refnumber = int(input_fname[-6:-4]) - 1  # reference frame在input frame的前面第2張
            #if refnumber < 10:
            #    refname = input_fname[0:-5] + str(refnumber) + '.png'  # 得到reference image file name
            #else:
            #    refname = input_fname[0:-6] + str(refnumber) + '.png'  # 得到reference image file name

            input_fname = prefix_filename + '_img1.ppm'
            refname = prefix_filename + '_img2.ppm'
            gt_flo_name = prefix_filename + '_flow.flo'
            input_frames += [input_fname]
            reference_frames += [refname]
            #gt_flo_name = input_fname[0:-4] + '.flo'  # 得到input frame的ground truth optical flow file name
            ground_truth_frame_flo += [gt_flo_name]

            #print(f"input_fname: {input_fname}  refname: {refname}  gt_flo_name: {gt_flo_name}")

        return input_frames, reference_frames, ground_truth_frame_flo

    def __getitem__(self, index):
        # imread()回傳array型態,所以可以用astype()將資料型態轉成float32
        input_image = imageio.v2.imread(self.image_input_list[index])
        ref_image = imageio.v2.imread(self.image_ref_list[index])
        gt_optical_flow = read_flow(self.ground_truth_list[index])
        #print(f"input_image: {self.image_input_list[index]}  ref_image: {self.image_ref_list[index]}  gt_optical_flow: {self.ground_truth_list[index]}")


        #cur_image = torch.from_numpy(cur_image).float()
        #ref_image = torch.from_numpy(ref_image).float()
        #cur_image = torch.div(cur_image, 255.0)
        #ref_image = torch.div(ref_image, 255.0)
        #cur_image = torch.transpose(cur_image, 2, 0)
        #cur_image = torch.transpose(cur_image, 1, 2)
        #ref_image = torch.transpose(ref_image, 2, 0)
        #ref_image = torch.transpose(ref_image, 1, 2)

        #input_image = input_image.astype(np.float32) / 255.0
        #ref_image = ref_image.astype(np.float32) / 255.0

        # [h, w, c] => [c, h, w]
        # 想成將RGB三個拆開處理, 每一張(w x h) image都只有R、G、B值
        input_image = input_image.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)

        # 將numpy的array轉成torch.tensor
        input_image = torch.from_numpy(input_image).float()
        ref_image = torch.from_numpy(ref_image).float()

        if self.transform is not None:
            input_image, ref_image, gt_optical_flow = self.transform(input_image, ref_image, gt_optical_flow)

        # augmentation => 替current image和reference image增加變化
        #input_image, ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image, [self.im_height, self.im_width])
        #input_image, ref_image, gt_optical_flow = random_flip(input_image, ref_image, gt_optical_flow)

        return input_image, ref_image, gt_optical_flow

    def __len__(self):
        return len(self.image_input_list)



