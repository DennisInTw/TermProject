import torch
from TermProjectDataSet import DataSet
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
import sys
import torch.nn.functional as F
from flowlib import flow_to_image

IMG_WIDTH = 448
IMG_HEIGHT = 256
EPOCHS = 1
BATCH_SIZE = 5
LEARNING_RATE = 1e-4
PYRAMID_LEVEL = 5


def showTensorImg(tenImg):
    img = tenImg
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # 將channel移到最後
    plt.imshow(img)
    plt.show()

def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    # print(inputfeature.size())
    outfeature = F.interpolate(inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear')
    # print(outfeature.size())
    return outfeature


def warp(image, optical_flow, device=torch.device('cpu')):
    b, c, im_h, im_w = image.size()

    hor = torch.linspace(-1.0, 1.0, im_w).view(1, 1, 1, im_w)
    hor = hor.expand(b, -1, im_h, -1)

    vert = torch.linspace(-1.0, 1.0, im_h).view(1, 1, im_h, 1)
    vert = vert.expand(b, -1, -1, im_w)

    grid = torch.cat([hor, vert], 1).to(device)
    optical_flow = torch.cat([optical_flow[:, 0:1, :, :] / ((im_w - 1.0) / 2.0), optical_flow[:, 1:2, :, :] / ((im_h - 1.0) / 2.0)], dim=1)

    # Channels last (which corresponds to optical flow vectors coordinates)
    grid = (grid + optical_flow).permute(0, 2, 3, 1)
    return F.grid_sample(image, grid=grid, padding_mode='border', align_corners=True)

def flow_warp(im, flow):
    device = im.device
    warp_result = warp(im, flow, device)

    return warp_result

# MEBasic對應到paper裡的G_k convnets
class MEBasic(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv5(x)
        return x

class ME_Spynet(nn.Module):
    def __init__(self, pyramid_level=5):
        super().__init__()
        self.pyramid_level = pyramid_level  # level = 0 ~ pyramid_level-1
        #建立G_k convnets, 要注意的是MEBasic(intLevel)要傳參數進去,否則會build error,但是現在還不清楚為什麼
        self.moduleBasic = torch.nn.ModuleList([MEBasic(intLevel) for intLevel in range(pyramid_level)])

    # im1 : current frame
    # im2 : reference frame
    def forward(self, im1, im2):
        logging.info(f'ME_Spynet => start working...')

        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2
        #showTensorImg(im1[0])
        #showTensorImg(im2[0])

        # 建立pyramid images
        # im1list, im2list => index=0存放level=4的pyramid image
        #                     index=1存放level=3的pyramid image
        # 所以底下迴圈會建立level=3 ~ level=0的pyramid image
        im1list = [im1_pre]
        im2list = [im2_pre]
        for intLevel in range(self.pyramid_level - 1):
            im1list.append(F.avg_pool2d(im1list[intLevel], kernel_size=2, stride=2))  # , count_include_pad=False))
            im2list.append(F.avg_pool2d(im2list[intLevel], kernel_size=2, stride=2))  # , count_include_pad=False))

        # shape_fine => 為level=0下的im2 images
        # zero shape => 建立一個都是0的初始flow estimation
        shape_fine = im2list[self.pyramid_level - 1].size()
        zeroshape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        device_id = im1.device.index
        #print(f"device_id: {device_id}")
        flowfileds = torch.zeros(zeroshape, dtype=torch.float32, device=device_id)

        for intLevel in range(self.pyramid_level):
            flowfiledsUpsample = bilinearupsacling(flowfileds) * 2.0
            flowfileds = flowfiledsUpsample + self.moduleBasic[intLevel](torch.cat([im1list[self.pyramid_level - 1 - intLevel], flow_warp(im2list[self.pyramid_level - 1 - intLevel], flowfiledsUpsample),flowfiledsUpsample], 1))  # residualflow

            u = flowfileds[0].detach().cpu().numpy()
            print(f"{type(u)} , {u.shape} , {np.max(u)}")
            img = flow_to_image(u, display=True)
            plt.imshow(img)
            plt.show()

        # flowfileds => 為pyramid最頂端的optical flow
        return flowfileds



class VideoCompressor(nn.Module):
    def __init__(self):
        super().__init__()
        # self.imageCompressor = ImageCompressor()
        self.opticFlow = ME_Spynet(pyramid_level=PYRAMID_LEVEL)

    def forward(self, input_img, ref_img):
        logging.info(f'VideoCompressor => start working...')
        self.opticFlow(input_img, ref_img)


# model => VideoCompressor()
def training(model, device, train_loader):
    for batch_idx, input in enumerate(train_loader):
        input_img, ref_img = input[0].to(device), input[1].to(device)
        logging.info(f'input_img.shape: {input_img.shape} \tref_img.shape: {ref_img.shape}')
        model(input_img, ref_img)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.__version__)
    print(torch.cuda.is_available())

    print(f"Confirm image size is changed or not !!!!!!")

    train_dataset = DataSet(path="data/vimeo_septuplet/test.txt", im_height=IMG_HEIGHT, im_width=IMG_WIDTH)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, num_workers=2, batch_size=BATCH_SIZE, pin_memory=True)

    model = VideoCompressor().to(device)
#    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    for epoch in range(EPOCHS):
        print(f'epoch: {epoch} start...')
        training(model, device, train_loader)



if __name__ == '__main__':
    # Set debug level
    # Print logging.info by level=logging.INFO
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
    main()
