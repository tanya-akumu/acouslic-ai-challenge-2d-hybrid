import SimpleITK as sitk
import numpy as np
import scipy
import torch
from nnunetv2.postprocessing.remove_connected_components import (remove_all_but_largest_component_from_segmentation)


def pad_and_downsample(image, target_shape, order=3):
    '''
    Function to downsample and pad an image
    '''
    original_time, original_height , original_width = image.shape
    target_time,  target_height, target_width  = target_shape
    
    # Calculate the scaling factors for height and width
    height_scale = target_height / original_height
    width_scale = target_width / original_width

    # Determine the dominant scale to maintain aspect ratio for spatial dimensions
    scale = min(height_scale, width_scale)
    
    # Calculate new dimensions while maintaining aspect ratio
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)
    new_time = original_time  # No scaling for the time dimension
    
    # Calculate padding amounts for spatial dimensions
    pad_height_before = (target_height - new_height) // 2
    pad_height_after = target_height - new_height - pad_height_before
    
    pad_width_before = (target_width - new_width) // 2
    pad_width_after = target_width - new_width - pad_width_before

    # Resize the image (only height and width)
    resized_image = scipy.ndimage.zoom(image, ( 1, new_height / original_height, new_width / original_width ), order=order)

    # Pad the resized image
    padded_image = np.pad(resized_image,
                          ((0, 0),
                           (pad_height_before, pad_height_after),
                           (pad_width_before, pad_width_after),
                           ),
                          mode='constant', constant_values=0)

    return padded_image, (pad_height_before,pad_height_after, pad_width_before, pad_width_after)

def unpad_and_upsample(image, original_shape, padding_info, order=0, mode='nearest'):
    '''
    Function to remove padding and upsample an image to its original size
    '''
    original_time, original_height, original_width  = original_shape
    pad_height_before, pad_height_after, pad_width_before, pad_width_after = padding_info

    # Remove padding from the image
    unpadded_image = image[:,
                           pad_height_before:image.shape[1]-pad_height_after,
                           pad_width_before:image.shape[2]-pad_width_after,
                           ]
    resized_image = scipy.ndimage.zoom(unpadded_image, ( 1,original_height / unpadded_image.shape[1], original_width/ unpadded_image.shape[2], ), order=order, mode=mode)
    return resized_image

def normalize_tensor(tensor):
    """
    function to normalize tensor image to values between -1 and 1
    """
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    tensor = (((tensor - min_val) / (max_val - min_val))-0.5)*2

    return tensor







