import cv2
import torch

def change_dimension(tensor_1, tensor_2):
  '''

  :param tensor_1: [B, C, H_1, W_1]
  :param tensor_2: [B, C, H_2, W_2]
  :return: tensor_1 tensor_2 in shape [B, C, min(H_1,H_2), min(W_1,W_2)]
  '''

  min_H = min(tensor_1.shape[2],tensor_2.shape[2])
  min_W = min(tensor_1.shape[3],tensor_2.shape[3])

  diff_tensor_1_H = (tensor_1.shape[2] - min_H) // 2
  diff_tensor_2_H = (tensor_2.shape[2] - min_H) // 2
  diff_tensor_1_W = (tensor_1.shape[3] - min_W) // 2
  diff_tensor_2_W = (tensor_2.shape[3] - min_W) // 2

  return_tensor_1 = tensor_1[:,:,diff_tensor_1_H:(diff_tensor_1_H + min_H),diff_tensor_1_W:(diff_tensor_1_W + min_W)]
  return_tensor_2 = tensor_2[:,:,diff_tensor_2_H:(diff_tensor_2_H + min_H),diff_tensor_2_W:(diff_tensor_2_W + min_W)]

  return return_tensor_1, return_tensor_2

def read_image_to_torch(img_path, gray_scale=True):
  '''
  read image and turn it to tensor
  :param img_path: the path of image that need to read
  :param gray_scale: whether read image in grayscale
  :return: a tensor of image in shape [B, C, H, W] with range [0,1]
  '''
  if gray_scale:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  else:
    img = cv2.imread(img_path)
  img_shape_0 = img.shape[0]
  img_shape_1 = img.shape[1]

  if gray_scale:
    torch_img = torch.from_numpy(img).view(1, img_shape_0, img_shape_1).float() / 255
  else:
    torch_img = torch.from_numpy(img).permute(2,0,1).float() / 255

  return torch_img.unsqueeze(0)

def torch_to_np_save_image(save_path, img):
  '''
  turn the image to np and save
  :param save_path: the path of saved image
  :param img: image to save (torch) in shape [C, H, W]
  :return: None
  '''

  # change dimension
  img = img.permute(0,2,3,1)
  img_np = img.squeeze().detach().cpu().numpy() * 255
  cv2.imwrite(save_path, img_np)

def sample_from_distribution(channels, size, var=0.1, distribution='uniform'):
  '''
  sample tensors in given shape from given distribution
  :param channels: channels in sampling tensor: C
  :param size: spatial size of the sampling tensor: [H, W]
  :param distribution: 'uniform' or 'normal'
  :param var: std scaler
  :return: sampling tensor in shape [1, C, H, W]
  '''
  if distribution == 'uniform':
    sampling_tensor = torch.rand((channels, size[0], size[1]))
  elif distribution == 'normal':
    sampling_tensor = torch.randn((channels, size[0], size[1]))

  sampling_tensor = sampling_tensor * var
  return sampling_tensor.unsqueeze(0)