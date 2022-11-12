import torch
import random
import torch.nn.functional as F

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
    combined_pad = F.pad(combined_img, padding)

    freesize0 = random.randint(0, max(size[0], image_shape[1]) - size[0])
    freesize1 = random.randint(0, max(size[1], image_shape[2]) - size[1])
    combined_crop = combined_pad[:, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]

    return (combined_crop[:input_img_channel, :, :], combined_crop[input_img_channel:, :, :])


def random_flip(images, labels):
    # augmentation setting....
    horizontal_flip = 1
    vertical_flip = 1
    transforms = 1

    if transforms and vertical_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [1])
        labels = torch.flip(labels, [1])
    if transforms and horizontal_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [2])
        labels = torch.flip(labels, [2])

    return images, labels
