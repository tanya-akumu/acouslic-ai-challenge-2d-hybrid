# Fetal Abdominal Detection and Segmentation in 3D blind sweep Ultrasound

This repository contains code for an advanced fetal abdominal segmentation system designed to automatically identify and delineate the fetal abdomen in 3D ultrasound volumes.

## Features

- Two-stage pipeline: frame selection followed by segmentation
- Ensemble of modified ResNet50 models for optimal frame selection
- 2D nnUNet for precise segmentation
- Custom data processing and augmentation for ultrasound imaging
- Specialized post-processing steps

## Installation

1. Clone this repository:
```
git clone https://github.com/tanya-akumu/acouslic-ai-challenge-2d-hybrid/tree/main
cd acouslic-ai-challenge-2d-hybrid
```

2. Create a new environment
   ```
   conda create --name fetal_seg 
   conda activate fetal_seg
   ```
   
4. Install dependencies:

   Using pip:
   `pip install -r requirements.txt`

## Usage

1. Training:
To train the models, run:
`sh run_inference.sh`

2. Inference:
To perform inference on new data, run:
`sh run_inference.sh`

## Model Architecture

Our approach utilizes a two-stage architecture:

1. Frame Selection: A modified ResNet50 identifies the optimal frame containing the fetal abdomen. The network is adapted to process single-channel ultrasound images and output binary classifications for class 0 (background) or 1 (optimal/suboptimal).

2. Segmentation: A 2D nnUNet performs precise segmentation on the selected frame. The nnUNet dynamically adapts its architecture based on the input data characteristics, optimizing performance for our specific task.


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project uses a modified version of nnUNet. For the original implementation, please see [nnUNet GitHub Repository](https://github.com/MIC-DKFZ/nnUNet).

## nnUNet Modifications

This project uses a modified version of nnUNet. The modifications are located in the `nnunet_modifications/` folder. To use this project:

1. Install nnUNet as per their instructions: [nnUNet Installation Guide](https://github.com/MIC-DKFZ/nnUNet#installation)
2. Replace the original dataloader file with our modified version:
   `cp nnunet_modifications/data_loader_2d_binary.py /path/to/nnUNet/installation/nnunet/training/dataloading/dataloader_2d.py`

3. The main modifications are in the sampling strategy to only include class 1 (optimal) and class 2 (suboptimal) frames in the training.

For the original nnUNet implementation, please refer to the [nnUNet GitHub Repository](https://github.com/MIC-DKFZ/nnUNet).
