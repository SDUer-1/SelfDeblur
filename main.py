import os
import torch
import argparse
import torch.optim
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from model import EncoderDecoder, FCN
from utils import *

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

    parser.add_argument('--blurred_images_dir', type=str, default='./datasets/lai/uniform_ycbcr', help='path of blurred images')
    parser.add_argument('--deblurred_image_output_dir', type=str, default='./deblurred_results/lai/x',
                        help='path of deblurred result')
    parser.add_argument('--kernel_output_dir', type=str, default='./deblurred_results/lai/k',
                        help='path of blur kernel')
    parser.add_argument('--kernel_size', type=int, default=[31,31], help='size of kernel')

    parser.add_argument('--iterations', type=int, default=1000, help='number of iterations')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # set seed
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    # iterate through images folder
    image_names = os.listdir(args.blurred_images_dir)
    for image_name in image_names:
        # reading blurred image
        blurred_image_path = os.path.join(args.blurred_images_dir, image_name)
        blurred_image = read_image_to_torch(blurred_image_path)
        blurred_image = blurred_image.to(device)
        blurred_image_shape = blurred_image.size()

        # Gx
        Gx = EncoderDecoder().to(device)

        # Gk
        Gk = FCN(args.kernel_size[0],args.kernel_size[1]).to(device)

        # optimizer
        optimizer = torch.optim.Adam([{'params': Gx.parameters()}, {'params': Gk.parameters(), 'lr': 1e-4}], lr=0.01)
        scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)

        # Loss
        mse = nn.MSELoss()

        # random x and k
        input_channels = 8
        uniform_random_x = torch.rand((input_channels, blurred_image_shape[2] + args.kernel_size[0] - 1, blurred_image_shape[3] + args.kernel_size[1] - 1)).unsqueeze(0).to(device)
        uniform_random_k = torch.rand((200)).to(device)
        reg = 0.0001
        for i in range(5000):
            optimizer.zero_grad()
            # add noise
            input_x = uniform_random_x + reg * torch.zeros(uniform_random_x.shape).type_as(uniform_random_x.data).normal_()

            # get the network output
            out_x = Gx(input_x)
            out_k = Gk(uniform_random_k)

            out_k_r = out_k.view(-1, 1, args.kernel_size[0], args.kernel_size[1])

            out_y = nn.functional.conv2d(out_x, out_k_r, bias=None).squeeze()

            total_loss = mse(out_y, blurred_image)

            total_loss.backward()
            optimizer.step()


        # save results
        save_path_x = os.path.join(args.deblurred_image_output_dir, image_name)
        torch_to_np_save_image(save_path_x, out_x)
        save_path_k = os.path.join(args.kernel_output_dir, image_name)
        torch_to_np_save_image(save_path_k, out_k_r)
