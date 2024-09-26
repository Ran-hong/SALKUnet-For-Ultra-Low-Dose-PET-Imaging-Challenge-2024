import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def restore_checkpoint(ckpt_dir, state, device):
  if not os.path.exists(ckpt_dir):
    return state
  else:
    # loaded_state = torch.load(ckpt_dir, map_location=device)
    loaded_state = torch.load(ckpt_dir)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['epoch'] = loaded_state['epoch']
    print(f"resume on epoch: {state['epoch']}")
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'epoch': state['epoch']
  }
  torch.save(saved_state, ckpt_dir)

def save_image_npy(tensor_map, path_dir, name):
  tensor_map = tensor_map.detach()
  np.save("{}/{}_row_data.npy".format(path_dir, name), tensor_map.permute(0, 2, 3, 1).cpu().numpy()[0, ..., 0])
  # tensor_map = np.clip(tensor_map.permute(0, 2, 3, 1).cpu().numpy()[0, ..., 0] * 255, 0, 255).astype(np.float32)
  # cv2.imwrite("{}/{}_img.png".format(path_dir, name), tensor_map)
  tensor_map = tensor_map.permute(0, 2, 3, 1).cpu().numpy()[0, ..., 0]
  fig = plt.figure()
  plt.axis("off")
  plt.imshow(tensor_map, cmap='gray')
  plt.savefig("{}/{}_img.png".format(path_dir, name), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
  plt.close()
  fig.clear()

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr,structural_similarity as ssim,mean_squared_error as mse

def indicate(img1,img2):
    if len(img1.shape) == 3:
        batch = img1.shape[0]
        psnr0 = np.zeros(batch)
        ssim0 = np.zeros(batch)
        mse0 = np.zeros(batch)
        for i in range(batch):
            t1= img1[i,...]/np.max(img1[i,...])
            t2= img2[i,...]/np.max(img2[i,...])
            psnr0[i,...] = psnr(t1,t2,data_range=1)
            ssim0[i,...] = ssim(t1,t2)
            mse0[i,...] = mse(t1,t2)
        return psnr0,ssim0,mse0
    else:
        img1 /= img1.max()
        img2 /= img2.max()
        psnr0 = psnr(img1,img2,data_range=1)
        # ssim0 = ssim(img1,img2, multichannel=False)
        ssim0 = 0
        mse0 = mse(img1,img2)
        return {
            "psnr": psnr0,
            "ssim": ssim0,
            "mse": mse0,
        }
