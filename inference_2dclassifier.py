# """
# The following is a the inference code for running the hybrid algorithm.

# It is meant to run within a container.

# To run it locally, you can call the following bash script:

#   ./test_run.sh

# This will start the inference and reads from ./test/input and outputs to ./test/output

# """

import json
import joblib
import numpy as np
import os
# from tqdm import tqdm
from pathlib import Path
import SimpleITK as sitk
import torch.nn.functional as F
from glob import glob
from postprocess import * # keep_largest_component, select_fetal_abdomen_mask_and_frame, pad_and_downsample, unpad_and_upsample
from inference_utils import get_model_checkpoints, get_ensemble_logits_mean, get_best_models, get_top_k_labels
from model_2d_nnunet import FetalAbdomenSegmentation
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("new_resources")
# checkpoint_path = "acoustic_2d_classif"

config = {
    'device': DEVICE,
    'num_classes': 3,
    'model_dir': RESOURCE_PATH,
}

def run():

    image_file_path = INPUT_PATH / "images/stacked-fetal-ultrasound"
    img_arr, image_information, uuid = read_and_preprocess_image(image_file_path)
    img_arr =(img_arr- torch.min(img_arr))/(torch.max(img_arr)-torch.min(img_arr))

    # preprocessing step to downsample image
    img_arr_down, padding_info = pad_and_downsample(img_arr.squeeze(0), (840,128,128))
    
    # define classifer model
    checkpoint_path = "new_resources/halfres_cv"
    classifier_models = get_best_models(checkpoint_path, config['device'])
    ensemble_probs, all_probs = get_ensemble_logits_mean(classifier_models,img_arr[:,:,::2,::2], config['device'])

    # smooth out ensemble probabilites
    N=5
    ensemble_probs = np.convolve(ensemble_probs, np.ones(N)/N, mode='same')

    # select best frame from ensemble logits
    fetal_abdomen_frame_number = ensemble_probs.argmax().item()

    fetal_abdomen_frame_image = img_arr_down[fetal_abdomen_frame_number]

    # get segmentation of best frame image
    # Instantiate the 2d nnunet
    algorithm = FetalAbdomenSegmentation()

    # Forward pass
    fetal_abdomen_probability_map = algorithm.predict(
        fetal_abdomen_frame_image,image_information, save_probabilities=True)

    # Postprocess the output
    fetal_abdomen_segmentation = algorithm.postprocess(
        fetal_abdomen_probability_map, padding_info)

    # write out array as image file
    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/fetal-abdomen-segmentation",
        array=fetal_abdomen_segmentation.squeeze(0),
        frame_number=fetal_abdomen_frame_number,
        uuid=uuid
    )

    # write json file of frame number and file location of binary mask 
    write_json_file(
        location=OUTPUT_PATH / "fetal-abdomen-frame-number.json",
        content=fetal_abdomen_frame_number
    )

    # Print shape and type of the output
    print("\printing output shape and type:")
    print(f"shape: {fetal_abdomen_segmentation.shape}")
    print(f"type: {type(fetal_abdomen_segmentation)}")
    print(f"dtype: {fetal_abdomen_segmentation.dtype}")
    print(f"unique values: {np.unique(fetal_abdomen_segmentation)}")
    print(f"frame number: {fetal_abdomen_frame_number}")
    print(type(fetal_abdomen_frame_number))

    return 0

def read_and_preprocess_image(file_path):
    input_files = glob(str(file_path / "*.tiff")) + \
        glob(str(file_path / "*.mha"))

    uuid = Path(input_files[0]).stem
    image_arr, properties = SimpleITKIO().read_images([input_files[0]])
        
    # Convert to tensors
    image_arr = torch.tensor(image_arr, dtype=torch.float32) 
      
    return image_arr, properties, uuid
    
def write_json_file(*, location, content):
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))

def write_array_as_image_file(*, location, array, uuid, frame_number=None):
    location.mkdir(parents=True, exist_ok=True)
    suffix = ".mha"
    # Assert that the array is 2D
    assert array.ndim == 2, f"Expected a 2D array, got {array.ndim}D."
    
    # Convert the 2D mask to a 3D mask (this is solely for visualization purposes)
    array = convert_2d_mask_to_3d(
        mask_2d=array,
        frame_number=frame_number,
        number_of_frames=840,
    )

    image = sitk.GetImageFromArray(array)
    # Set the spacing to 0.28mm in all directions
    image.SetSpacing([0.28, 0.28, 0.28])
    sitk.WriteImage(
        image,
        location / f"{uuid}{suffix}",
        useCompression=True,
    )


def convert_2d_mask_to_3d(*, mask_2d, frame_number, number_of_frames):
    # Convert a 2D mask to a 3D mask
    mask_3d = np.zeros((number_of_frames, *mask_2d.shape), dtype=np.uint8)
    # If frame_number == -1, return a 3D mask with all zeros
    if frame_number == -1:
        return mask_3d
    # If frame_number is within the valid range, set the corresponding frame to the 2D mask
    if frame_number is not None and 0 <= frame_number < number_of_frames:
        mask_3d[frame_number, :, :] = mask_2d
        return mask_3d
    # If frame_number is None or out of bounds, raise a ValueError
    else:
        raise ValueError(
            f"frame_number must be between -1 and {number_of_frames - 1}, got {frame_number}."
        )
    

if __name__ == "__main__":
    raise SystemExit(run())
