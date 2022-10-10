import os
import torch
import argparse
import torch.optim
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from model import EncoderDecoder, FCN
from utils import *
from SSIM import SSIMLoss

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--blurred_images_dir', type=str, default='./datasets/lai/uniform_ycbcr',
                        help='path of blurred images')
    parser.add_argument('--deblurred_image_output_dir', type=str, default='./deblurred_results/lai/x',
                        help='path of deblurred result')
    parser.add_argument('--kernel_output_dir', type=str, default='./deblurred_results/lai/k',
                        help='path of blur kernel')

    parser.add_argument('--iterations', type=int, default=5000, help='number of iterations')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # set auto-tuner
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # iterate through images folder
    image_names = os.listdir(args.blurred_images_dir)
    for image_name in image_names:
        # reading blurred image
        print('Deblurring image: ', image_name)
        blurred_image_path = os.path.join(args.blurred_images_dir, image_name)
        blurred_image = read_image_to_torch(blurred_image_path)
        blurred_image = blurred_image.to(device)
        blurred_image_shape = blurred_image.size()

        # determine kernel size according to kernel number
        kernel_number = image_name.split('_')[-1]
        if kernel_number == '01.png':
            kernel_size = [31, 31]
        elif kernel_number == '02.png':
            kernel_size = [51, 51]
        elif kernel_number == '03.png':
            kernel_size = [55, 55]
        elif kernel_number == '04.png':
            kernel_size = [75, 75]

        padh = (kernel_size[0] - 1) // 2
        padw = (kernel_size[1] - 1) // 2


        # Gx
        Gx = EncoderDecoder().to(device)

        # Gk
        Gk_input_size = 200
        Gk = FCN(Gk_input_size, kernel_size[0],kernel_size[1]).to(device)

        # optimizer
        optimizer = torch.optim.Adam([{'params': Gx.parameters()}, {'params': Gk.parameters(), 'lr': 1e-4}], lr=0.01)
        scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)

        # Loss
        mse = nn.MSELoss()
        ssim = SSIMLoss()

        # random x and k
        input_channels = 8
        sampling_tensor_size = [blurred_image_shape[2] + kernel_size[0] - 1, blurred_image_shape[3] + kernel_size[1] - 1]
        sampling_x = sample_from_distribution(input_channels, sampling_tensor_size).to(device)
        sampling_k = sample_from_distribution(Gk_input_size, [1,1]).squeeze().to(device)
        reg = 0.0001
        for i in tqdm(range(args.iterations)):
            scheduler.step(i)
            optimizer.zero_grad()
            # add noise
            noise = sample_from_distribution(input_channels, sampling_tensor_size, var=1).to(device)
            input_x = sampling_x + reg * noise

            # get the network output
            out_x = Gx(input_x)
            out_k = Gk(sampling_k)

            out_k_r = out_k.view(-1, 1, kernel_size[0], kernel_size[1])

            out_y = nn.functional.conv2d(out_x, out_k_r, bias=None)

            if i < 500:
                total_loss = mse(out_y, blurred_image)
            else:
                total_loss = 1 - ssim(out_y, blurred_image)

            total_loss.backward()
            optimizer.step()


        # save results
        save_path_x = os.path.join(args.deblurred_image_output_dir, image_name)
        out_x = out_x[: , :, padh : padh + blurred_image_shape[2], padw : padw + blurred_image_shape[3]]
        torch_to_np_save_image(save_path_x, out_x)

        save_path_k = os.path.join(args.kernel_output_dir, image_name)
        out_k_r = out_k_r / torch.max(out_k_r)
        torch_to_np_save_image(save_path_k, out_k_r)

