import os
import cv2
import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from collections import defaultdict

def align_image(result_img, ground_img, max_shift=5):
    shifts = np.arange(2 * max_shift + 1)
    result_img = result_img[10:-10, 10:-10]
    ground_img = ground_img[15:-15, 15:-15]

    min_ssd = math.inf
    shift_result = None
    for i in range(len(shifts)):
        for j in range(len(shifts)):
            if -10 + shifts[i] == 0 and -10 + shifts[j] == 0:
                shift_img = result_img[shifts[i]:, shifts[j]:]
            elif -10 + shifts[i] != 0 and -10 + shifts[j] == 0:
                shift_img = result_img[shifts[i]:-10 + shifts[i], shifts[j]:]
            elif -10 + shifts[i] == 0 and -10 + shifts[j] != 0:
                shift_img = result_img[shifts[i]:, shifts[j]:-10 + shifts[j]]
            else:
                shift_img = result_img[shifts[i]:-10 + shifts[i], shifts[j]:-10 + shifts[j]]
            ssd = np.sum((shift_img - ground_img) ** 2)

            if ssd < min_ssd:
                min_ssd = ssd
                shift_result = shift_img

    return shift_result, ground_img


if __name__ == '__main__':

    grount_truth_path = './datasets/levin/groundtruth'
    deblurred_results_x_path = './deblurred_results/levin/x'
    deblurred_results_k_path = './deblurred_results/levin/k'

    deblurred_imgs_list = os.listdir(deblurred_results_x_path)
    psnr_results = []
    ssim_results = []
    for deblurred_img_name in deblurred_imgs_list:
        print(deblurred_img_name)
        category = deblurred_img_name.split('_')[0]

        deblurred_img_path = os.path.join(deblurred_results_x_path, deblurred_img_name)
        gt_img_name = deblurred_img_name.split('_')[0] + '.png'
        gt_img_path = os.path.join(grount_truth_path, gt_img_name)

        deblurred_img = cv2.imread(deblurred_img_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)

        shift_deblurred_img, shift_gt = align_image(deblurred_img, gt_img)

        psnr = compare_psnr(shift_deblurred_img, shift_gt, data_range=255)
        ssim = compare_ssim(shift_deblurred_img, shift_gt)
        psnr_results.append(psnr)
        ssim_results.append(ssim)

    print("Average PSNR: ", np.mean(psnr_results))
    print("Average SSIM: ", np.mean(ssim_results))
