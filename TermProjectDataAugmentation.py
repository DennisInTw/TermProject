import torch
import random

# 為什麼這裡的reference image用labels??
# image和labels都是做完transpose的tensor
def random_crop_and_pad_image_and_labels(image, labels, size):
    combined_img = torch.cat([image, labels], 0)  # 從原本3個channel變成6個channel, 都是256x256
    input_img_channel = image.size()[0]
    image_shape = image.size()

    # Padding 0 at the left/right/top/bottom of combined_image
    # 不過目前看起來padding並沒有作用......
    padding_left, padding_right = (0, max(size[1], image_shape[2]) - image_shape[2])
    padding_top, padding_bottom = (0, max(size[0], image_shape[1]) - image_shape[1])
    padding = (padding_left, padding_right, padding_top, padding_bottom)
    combined_pad = torch.nn.functional.pad(combined_img, padding)

    freesize0 = random.randint(0, max(size[0], image_shape[1]) - size[0])
    freesize1 = random.randint(0, max(size[1], image_shape[2]) - size[1])
    combined_crop = combined_pad[:, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]

    return (combined_crop[:input_img_channel, :, :], combined_crop[input_img_channel:, :, :])


def random_flip(frame1, frame2, optical_flow):
    # augmentation setting....
    horizontal_flip = 1
    vertical_flip = 1
    transforms = 1

    # 需要將optical flow也做flip,但是現在還不能work
    if transforms and vertical_flip and random.randint(0, 1) == 1:
        frame1 = torch.flip(frame1, [1])
        frame2 = torch.flip(frame2, [1])
        #optical_flow = torch.flip(optical_flow, [1])
    if transforms and horizontal_flip and random.randint(0, 1) == 1:
        frame1 = torch.flip(frame1, [2])
        frame2 = torch.flip(frame2, [2])
        #optical_flow = torch.flip(optical_flow, [2])

    return frame1, frame2, optical_flow
