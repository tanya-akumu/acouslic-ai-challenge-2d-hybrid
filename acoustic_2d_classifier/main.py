# """
# The following is a the training code for running the hybrid algorithm.

# It is meant to run within a container.

# To run it locally, you can call the following bash script:

#   ./run_train.sh

# """

import os
import argparse
import traceback
import shutil
import torch
import numpy as np
from data_utils import AcousticDataset, create_dataloaders
from model import initialize_model
from train import train_and_evaluate, generate_classification_report
from utils import create_output_folder, save_checkpoint
import sys
import json
from torchvision.models import ResNet50_Weights
import torch.nn.functional as F


if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description="Process an integer argument.")
    parser.add_argument('fold', type=int, help='Fold')
    parser.add_argument('name', type=str, help='expname')

    args = parser.parse_args()
    fold = args.fold
    expname= args.name

    print("FOLD: ", fold)
    
    print(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES")}', flush=True)
    print(f'CUDA available: {torch.cuda.is_available()}')
   print("CUDA is not available. Check your PyTorch installation and CUDA setup.")

    output_folder,timestamp, logger = create_output_folder(expname,fold)
    logger.info(f'Output folder created at {output_folder}')
    # with open(os.path.join(output_folder,'logs_'+str(fold),'logs.txt'), "w") as f:
    try:

        # Redirect standard output to the file
        # sys.stdout = f
        # sys.stderr = f

        logger.info(f'Run timestamp: {timestamp}')

        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # Define the paths to your dataset
        images_dir = os.path.join('/workspace/raw_data','images','stacked_fetal_ultrasound')
        labels_dir = os.path.join('/workspace/raw_data','masks','stacked_fetal_abdomen')

        splits=os.path.join(os.path.dirname(os.path.abspath(__file__)),'splits_final.json')
        splits = json.load(open(splits))
      
        # Create dataset and dataloaders
        acoustic_dataset = AcousticDataset(images_dir, labels_dir,sublen=80)
        train_loader, _,_ = create_dataloaders(acoustic_dataset,
                                                    splits = splits,
                                                    fold=fold, 
                                                    batch_size=8,
                                                    logger=logger,
                                                    suffix = '.mha')
        acoustic_hard_dataset = AcousticDataset(images_dir, labels_dir,sublen=0)

        _, val_loader,test_loader = create_dataloaders(acoustic_hard_dataset,
                                                        splits = splits,
                                                        fold=fold, 
                                                        batch_size=8,
                                                        collate_fn=None,
                                                        logger=logger,
                                                        suffix = '.mha')
    
        # Initialize the model
        model, optimizer, scheduler, criterion = initialize_model(device,
                                                                warmup=5,
                                                                weights=None
                                                                )
                                                                
        model.eval()

        train_and_evaluate(model, 
                        train_loader, 
                        test_loader, 
                        optimizer, 
                        scheduler, 
                        criterion, 
                        device, 
                        total_epochs=301, 
                        output_folder=output_folder,
                        epoch_save = 25,
                        fold=fold,
                        logger=logger
                        )
        # Save the final model
        save_checkpoint(model, os.path.join(output_folder,'logs_'+str(fold),'checkpoints_'+str(fold), 'final_model.pth'))

        # evaluate on test
        test_report,ts_print = generate_classification_report(model, test_loader, device)
        logger.info(ts_print)

    except Exception as e:
        logger.info(f"An error occurred:{e}")
        logger.info("Stack trace:")
        logger.info(traceback.print_exc())


    
