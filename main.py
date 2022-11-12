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


class Spynet(nn.Module):
    def __init__(self, pyramid_level):
        super().__init__()
        self.pyramid_level = pyramid_level

        self.conv1 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)

    # Vk_1 => 上一個pyramid (level=k-1)得到的predicted optical flow
    # 但是未做upsampling
    def forward(self, im1, im2, VK_1):
        logging.info(f'Spynet => start working...')

        batchsize = im1.size()[0]

        img_width = im1.size()[3]
        img_height = im1.size()[2]
        device_id = im1.device
        print(f"device_id = {device_id}")

        if self.pyramid_level == 0:
            # 產生一個都是0的optical flow
            zeroshape = [batchsize, 2, img_height, img_width]
            flowfileds = torch.zeros(zeroshape, dtype=torch.float32, device=device_id)

            net_input = torch.cat([im1, im2, flowfileds], 1)
            print(f"net_input.shape: {net_input.shape}  type(net_input): {type(net_input)}")

            out = self.relu1(self.conv1(net_input))
            out = self.relu2(self.conv2(out))
            out = self.relu3(self.conv3(out))
            out = self.relu4(self.conv4(out))
            out = self.conv5(out)

            numpy_optical_flow = out[0].detach().cpu().numpy()
            print(f"{type(numpy_optical_flow)} , {numpy_optical_flow.shape} , {np.max(numpy_optical_flow)}")
            optical_flow_img = flow_to_image(numpy_optical_flow, display=True)
            plt.imshow(optical_flow_img)
            plt.show()

            flowfileds += out
        # pyramid level > 0
        else:
            flowfileds = VK_1

            flowfiledsUpsample = bilinearupsacling(flowfileds) * 2.0
            warp_result = flow_warp(im2, flowfiledsUpsample)
            net_input = torch.cat([im1, warp_result, flowfiledsUpsample], 1)
            print(f"net_input.shape: {net_input.shape}  type(net_input): {type(net_input)}")

            out = self.relu1(self.conv1(net_input))
            out = self.relu2(self.conv2(out))
            out = self.relu3(self.conv3(out))
            out = self.relu4(self.conv4(out))
            out = self.conv5(out)

            numpy_optical_flow = out[0].detach().cpu().numpy()
            print(f"{type(numpy_optical_flow)} , {numpy_optical_flow.shape} , {np.max(numpy_optical_flow)}")
            optical_flow_img = flow_to_image(numpy_optical_flow, display=True)
            plt.imshow(optical_flow_img)
            plt.show()

            flowfileds = flowfiledsUpsample + out


        return flowfileds

# models => Spynet()
def training(models, device, train_loader):
    print(f"Start training...")
    for batch_idx, input in enumerate(train_loader):
        im1, im2 = input[0].to(device), input[1].to(device)

        # 從最小的pyramid開始train
        predicted_flow = None
        for k in range(PYRAMID_LEVEL):
            print(f'train_one_level => current level = {k}')

            # 建立pyramid k的optimizer以及改成training狀態
            optimizer = torch.optim.Adam(models[k].parameters(), lr=LEARNING_RATE)
            models[k].train()
            optimizer.zero_grad()

            # 準備當下pyramid的image1和image2, 所以對原始大小的image做down sampling
            im1_down_sample = im1
            im2_down_sample = im2
            for intLevel in range(PYRAMID_LEVEL - 1 - k):
                im1_down_sample = F.avg_pool2d(im1_down_sample, kernel_size=2, stride=2)
                im2_down_sample = F.avg_pool2d(im2_down_sample, kernel_size=2, stride=2)
            print(f'type(im1_down_sample): {type(im1_down_sample)}  im1_down_sample.shape: {im1_down_sample.shape} \t type(im2_down_sample): {type(im2_down_sample)}  im2_down_sample.shape: {im2_down_sample.shape}')
            showTensorImg(im1_down_sample[0])
            showTensorImg(im2_down_sample[0])

            predicted_flow = models[k](im1_down_sample, im2_down_sample, predicted_flow)
            models[k].eval()


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.__version__)
    print(torch.cuda.is_available())

    print(f"Confirm image size is changed or not !!!!!!")

    train_dataset = DataSet(path="data/vimeo_septuplet/test.txt", im_height=IMG_HEIGHT, im_width=IMG_WIDTH)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, num_workers=2, batch_size=BATCH_SIZE, pin_memory=True)

    # 建立所有Spynet
    models = torch.nn.ModuleList([Spynet(intLevel).to(device) for intLevel in range(PYRAMID_LEVEL)])

    for eporch in range(EPOCHS):
        training(models, device, train_loader)




if __name__ == '__main__':
    # Set debug level
    # Print logging.info by level=logging.INFO
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
    main()
