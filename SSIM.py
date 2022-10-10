import torch
import torch.nn.functional as F
from utils import gaussian_kernel


class SSIMLoss(torch.nn.Module):
    def __init__(self, size=11, sigma=1.5, channels=1):
        super(SSIMLoss, self).__init__()
        self.size = size
        self.channels = channels
        self.sigma = sigma
        self.kernel = gaussian_kernel(self.size, self.sigma, self.channels)

    def forward(self, img_1, img_2):

        # calculate mu of two images
        self.kernel = self.kernel.to(img_1.device)
        mu_x = F.conv2d(img_1, self.kernel, padding=self.size // 2, groups=self.channels)
        mu_y = F.conv2d(img_2, self.kernel, padding=self.size // 2, groups=self.channels)

        mu_x_2 = mu_x ** 2
        mu_y_2 = mu_y ** 2

        # calculate sigma of two images
        sigma_x_2 = F.conv2d(img_1 * img_1, self.kernel, padding=self.size // 2, groups=self.channels) - mu_x_2
        sigma_y_2 = F.conv2d(img_2 * img_2, self.kernel, padding=self.size // 2, groups=self.channels) - mu_y_2

        # covariance
        sigma_xy = F.conv2d(img_1 * img_2, self.kernel, padding=self.size // 2, groups=self.channels) - mu_x * mu_y

        # constant c_1 and c_2
        c_1 = 0.01 ** 2
        c_2 = 0.03 ** 2

        # calculate ssim
        ssim = ((2 * mu_x * mu_y + c_1) * (2 * sigma_xy + c_2)) / ((mu_x_2 + mu_y_2 + c_1) * (sigma_x_2 + sigma_y_2 + c_2))
        return ssim.mean()
