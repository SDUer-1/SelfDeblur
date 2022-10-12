import os
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

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

        psnr = compare_psnr(deblurred_img, gt_img_Y, data_range=255)
        ssim = compare_ssim(deblurred_img, gt_img_Y)
        print(psnr)
        print(ssim)

