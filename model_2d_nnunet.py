from pathlib import Path
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniqueImagesValidator,
    UniquePathIndicesValidator,
)
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from postprocess import *

RESOURCE_PATH = Path("new_resources")

''' 
code adapted from https://github.com/DIAGNijmegen/ACOUSLIC-AI-baseline/blob/main 
'''

class FetalAbdomenSegmentation(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        # Path to nnUNet model directory
        self.nnunet_model_dir = RESOURCE_PATH / "nnUNet_results"

        # Initialize the predictor
        self.predictor = self.initialize_predictor()

    def initialize_predictor(self, task="Dataset840_FetalAbdomen",
                             network="2d", checkpoint="checkpoint_best.pth", folds=(0,1,2,3,4)):
        """
        Initializes the nnUNet predictor
        """
        # instantiates the predictor
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            device=torch.device('cuda', 0),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )

        # initializes the network architecture, loads the checkpoint
        predictor.initialize_from_trained_model_folder(
            join(self.nnunet_model_dir,
                 f'{task}/nnUNetTrainer__nnUNetPlans__{network}'),
            use_folds=folds,
            checkpoint_name=checkpoint,
        )
        predictor.dataset_json['file_ending'] = '.nii.gz'

        return predictor

    def predict(self, input_img,properties, save_probabilities=True):
        """
        Use trained nnUNet network to generate segmentation masks
        """
        # ideally we would like to use predictor.predict_from_files but this docker container will be called
        # for each individual test case so that this doesn't make sense
        # image_np, properties = SimpleITKIO().read_images([input_img_path])
        # image_np, padding_info = pad_and_downsample(image_np, (840,128,128))
        
        input_img = input_img[np.newaxis,np.newaxis,...]
        _, probabilities = self.predictor.predict_single_npy_array(
            input_img, properties, None, None, save_probabilities)
        return probabilities

    def postprocess(self, probability_map, padding_info):
        """
        Postprocess the nnUNet output to generate the final AC segmentation mask
        """
        # Define the postprocessing configurations
        configs = {
            "soft_threshold": 0.9,
        }

        # Postprocess the probability map
        mask_postprocessed = postprocess_single_probability_map(
            probability_map, configs)
        print('Postprocessing done')
        mask_postprocessed = unpad_and_upsample(mask_postprocessed, (840,562,744), padding_info)
        return mask_postprocessed


def postprocess_single_probability_map(softmax_prob_map, thresh=0.9):

    # Copy the input probability map
    softmax_maps = softmax_prob_map.copy()

    # Find the class with the maximum probability at each pixel across all channels
    # This will have shape [n_frames, H, W]
    masks = np.argmax(softmax_maps, axis=0)
    masks = masks.astype(np.uint8)
    masks [ masks > 1 ] = 1

    return masks
