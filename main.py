import torch
from TermProjectDataSet import DataSet, Compose, Normalize, RandomRotate, RandomFlip
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
import sys
import torchvision
from flowlib import flow_to_image
import imageio

IMG_WIDTH = 512
IMG_HEIGHT = 384
EPOCHS = 60
BATCH_SIZE = 10
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0#4e-5
PYRAMID_LEVEL = 5
TRAINING_DATA_PATH = "data/training/training_frames_list.txt"
LOSS_PIC_PATH = "./result/Loss.png"
MODEL_PATH = "./result/"
OPTICAL_FLOW_IMG_PATH = "./result/"
SHOW_FLOW_IMG = False


def showTensorImg(tenImg):
    img = tenImg
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # 將channel移到最後
    plt.imshow(img)
    plt.show()


class EPELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, target):
        dist = (target - predict).pow(2).sum().sqrt()
        return dist.mean()


def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    # print(inputfeature.size())
    outfeature = torch.nn.functional.interpolate(inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear')
    # print(outfeature.size())
    return outfeature

def bilineardownsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    # print(inputfeature.size())
    outfeature = torch.nn.functional.interpolate(inputfeature, (inputheight // 2, inputwidth // 2), mode='bilinear')
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
    return torch.nn.functional.grid_sample(image, grid=grid, padding_mode='border', align_corners=True)

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
        logging.info(f"Spynet => start working...")

        batchsize = im1.size()[0]

        img_width = im1.size()[3]
        img_height = im1.size()[2]
        device_id = im1.device
        logging.info(f"device_id = {device_id}")

        if self.pyramid_level == 0:
            # 產生一個都是0的optical flow
            zeroshape = [batchsize, 2, img_height, img_width]
            flowfileds = torch.zeros(zeroshape, dtype=torch.float32, device=device_id)

            net_input = torch.cat([im1, im2, flowfileds], 1)
            logging.info(f"net_input.shape: {net_input.shape}  type(net_input): {type(net_input)}")

            out = self.relu1(self.conv1(net_input))
            out = self.relu2(self.conv2(out))
            out = self.relu3(self.conv3(out))
            out = self.relu4(self.conv4(out))
            out = self.conv5(out)

            flowfileds += out


        # pyramid level > 0
        else:
            flowfileds = VK_1

            # 將VK_1做upsampling 2倍
            flowfiledsUpsample = bilinearupsacling(flowfileds) * 2.0
            warp_result = flow_warp(im2, flowfiledsUpsample)
            net_input = torch.cat([im1, warp_result, flowfiledsUpsample], 1)
            logging.info(f"net_input.shape: {net_input.shape}  type(net_input): {type(net_input)}")

            out = self.relu1(self.conv1(net_input))
            out = self.relu2(self.conv2(out))
            out = self.relu3(self.conv3(out))
            out = self.relu4(self.conv4(out))
            out = self.conv5(out)

            flowfileds = flowfiledsUpsample + out

            global SHOW_FLOW_IMG
            if SHOW_FLOW_IMG is True and self.pyramid_level == PYRAMID_LEVEL-1:
                numpy_optical_flow = out[0].detach().cpu().numpy()
                logging.info(f"{type(numpy_optical_flow)} , {numpy_optical_flow.shape} , {np.max(numpy_optical_flow)}")
                optical_flow_img = flow_to_image(numpy_optical_flow)
                imageio.imwrite('optical_flow.png', optical_flow_img)
                plt.imshow(optical_flow_img)
                plt.show()
                SHOW_FLOW_IMG = False


        return flowfileds


# models => Spynet()
def training(models, device, train_loader, criterion):
    logging.info(f"=== Start training ===")
    train_loss = 0.0
    last_gt_optical_flow = None
    last_predicted_optical_flow = None

    for batch_idx, input in enumerate(train_loader):
        # 得到image1, image2, ground truth optical flow
        im1, im2, gt_optical_flow = input[0].to(device), input[1].to(device), input[2].to(device)

        # 將ground truth optical flow的維度改掉
        # [batch, img_h, img_w, 2] ==> [batch, 2, img_h, img_w]
        gt_optical_flow = torch.transpose(gt_optical_flow, 1, 3)
        gt_optical_flow = torch.transpose(gt_optical_flow, 3, 2)

        # 從最小的pyramid開始train
        predicted_optical_flow = None
        for k in range(PYRAMID_LEVEL):
            logging.info(f"batch_idx: {batch_idx}   /   current level = {k}")

            # 建立pyramid level=k的optimizer以及改成training狀態
            optimizer = torch.optim.Adam(models[k].parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            models[k].train()
            optimizer.zero_grad()

            # 準備當下pyramid的image1和image2, 所以對原始大小的image做down sampling
            # 以及對原始的ground truth optical flow做down sampling
            im1_down_sample = im1
            im2_down_sample = im2
            gt_flow_down_sample = gt_optical_flow
            for intLevel in range(PYRAMID_LEVEL - 1 - k):
                im1_down_sample = torch.nn.functional.avg_pool2d(im1_down_sample, kernel_size=2, stride=2)
                im2_down_sample = torch.nn.functional.avg_pool2d(im2_down_sample, kernel_size=2, stride=2)
                gt_flow_down_sample = bilineardownsacling(gt_flow_down_sample)

            logging.info(f"type(im1_down_sample): {type(im1_down_sample)}  im1_down_sample.shape: {im1_down_sample.shape} \t type(im2_down_sample): {type(im2_down_sample)}  im2_down_sample.shape: {im2_down_sample.shape}")
            logging.info(f"type(gt_flow_down_sample): {type(gt_flow_down_sample)}  gt_flow_down_sample.shape: {gt_flow_down_sample.shape}")
            #showTensorImg(im1_down_sample[0])
            #showTensorImg(im2_down_sample[0])

            # 在多個model下training, 一定要在model的forward之前將資料detach(),不然再loss.backward()會有問題
            im1_down_sample = im1_down_sample.detach()
            im2_down_sample = im2_down_sample.detach()

            im1_down_sample = im1_down_sample.to(device)
            im2_down_sample = im2_down_sample.to(device)

            if predicted_optical_flow != None:
                predicted_optical_flow = predicted_optical_flow.detach()
                predicted_optical_flow = predicted_optical_flow.to(device)
            predicted_optical_flow = models[k](im1_down_sample, im2_down_sample, predicted_optical_flow)


            # 再來要補上計算loss, 然後更新參數
            #pred_flow_up_sample = bilinearupsacling(predicted_optical_flow) * 2.0

            logging.info(f"predicted_optical_flow.shape: {predicted_optical_flow.shape}  gt_flow_down_sample.shape: {gt_flow_down_sample.shape}")

            loss = criterion(predicted_optical_flow, gt_flow_down_sample)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            models[k].eval()
        #end loop for PYRAMID_LEVEL


        # 紀錄最後一個batch的最後一個ground true和predicted的optical flow, 然後回傳回去儲存成.png
        # len(train_loader) => 表示有多少個batch
        if batch_idx == len(train_loader) - 1:
            last_index = gt_optical_flow.size()[0] - 1
            logging.info(f"last_index: {last_index}")
            last_gt_optical_flow = gt_optical_flow[last_index]
            last_predicted_optical_flow = predicted_optical_flow[last_index]

    #end loop for train_loader

    train_loss /= len(train_loader)
    print(f"loss : {train_loss}")

    return train_loss, last_gt_optical_flow, last_predicted_optical_flow


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.__version__)
    print(torch.cuda.is_available())

    train_transform = Compose([
        RandomRotate(minmax_angle=17),
        RandomFlip(flip_vertical=True, flip_horizontal=True),
        Normalize(mean=[0.485, 0.406, 0.456], std=[0.229, 0.225, 0.224])
    ])

    print(f"Confirm image size is changed or not !!!!!!")

    train_dataset = DataSet(filelist=TRAINING_DATA_PATH, im_height=IMG_HEIGHT, im_width=IMG_WIDTH, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    # 建立所有Spynet
    models = torch.nn.ModuleList([Spynet(intLevel).to(device) for intLevel in range(PYRAMID_LEVEL)])

    criterion = EPELoss()

    # 蒐集每一個eporch的training loss,看看是不是有收斂
    train_loss = [0.0] * EPOCHS

    for eporch in range(EPOCHS):
        print(f"============ eporch [#{eporch}] =============")

        train_loss[eporch], last_gt_optical_flow, last_predicted_optical_flow = training(models, device, train_loader, criterion)

        # 將最後一個batch的最後一個ground true和predicted的optical flow儲存成image比對
        numpy_optical_flow = last_gt_optical_flow.detach().cpu().numpy()
        logging.info(f"{type(numpy_optical_flow)} , {numpy_optical_flow.shape} , {np.max(numpy_optical_flow)}")
        optical_flow_img = flow_to_image(numpy_optical_flow)
        imageio.imwrite(OPTICAL_FLOW_IMG_PATH + 'eporch_' + str(eporch) + '_gt.png', optical_flow_img)

        numpy_optical_flow = last_predicted_optical_flow.detach().cpu().numpy()
        logging.info(f"{type(numpy_optical_flow)} , {numpy_optical_flow.shape} , {np.max(numpy_optical_flow)}")
        optical_flow_img = flow_to_image(numpy_optical_flow)
        imageio.imwrite(OPTICAL_FLOW_IMG_PATH + 'eporch_' + str(eporch) + '_pred.png', optical_flow_img)


    # 將model裡的參數儲存起來
    for i in range(PYRAMID_LEVEL):
        torch.save(models[i].state_dict(), MODEL_PATH+"model_"+str(i)+".pt")

    plt.plot(train_loss, label="Training Loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(LOSS_PIC_PATH)
    plt.close()



if __name__ == '__main__':
    # Set debug level
    # Print logging.info by level=logging.INFO
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
    main()
