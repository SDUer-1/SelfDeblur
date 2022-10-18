import os
import cv2
import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

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

    grount_truth_path = './datasets/lai/ground_truth'
    deblurred_results_x_path = './deblurred_results/lai/x'
    deblurred_results_k_path = './deblurred_results/lai/k'

    deblurred_imgs_list = os.listdir(deblurred_results_x_path)
    for deblurred_img_name in deblurred_imgs_list:
        print(deblurred_img_name)
        deblurred_img_path = os.path.join(deblurred_results_x_path, deblurred_img_name)
        gt_img_name = deblurred_img_name.split('_')[0] + '_' + deblurred_img_name.split('_')[1] + '.png'
        gt_img_path = os.path.join(grount_truth_path, gt_img_name)

        deblurred_img = cv2.imread(deblurred_img_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_img_path)
        gt_img_Y = cv2.cvtColor(gt_img, cv2.COLOR_BGR2YCrCb)[:,:,0]

        shift_deblurred_img, shift_gt = align_image(deblurred_img, gt_img_Y)

        psnr = compare_psnr(shift_deblurred_img, shift_gt, data_range=255)
        ssim = compare_ssim(shift_deblurred_img, shift_gt)
        print(psnr)
        print(ssim)

