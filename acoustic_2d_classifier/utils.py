import shutil
import datetime
from datetime import datetime as dt

import torch
import os
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as patches
import numpy as np
import logging

def create_output_folder(exp,fold):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    project_root = os.path.dirname(os.path.abspath(__file__))
    experiments_folder = os.path.join(project_root, 'experiments')
    output_folder = os.path.join(experiments_folder, f'exp_{exp}')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder,'logs_'+str(fold)), exist_ok=True)
    os.makedirs(os.path.join(output_folder,'logs_'+str(fold),'checkpoints_'+str(fold)), exist_ok=True)

    logger = setup_logger(fold, os.path.join(output_folder,'logs_'+str(fold)))
        
    
    # Copy main.py to the output folder
    # shutil.copy(os.path.join(project_root, 'main.py'      ), os.path.join(output_folder,'main_'+exp+'.py'      ))
    # shutil.copy(os.path.join(project_root, 'data_utils.py'), os.path.join(output_folder,'data_utils_'+exp+'.py'))
    # shutil.copy(os.path.join(project_root, 'train.py'     ), os.path.join(output_folder,'train_'+exp+'.py'     ))
    # shutil.copy(os.path.join(project_root, 'utils.py'), os.path.join(output_folder,'utils_'+exp+'.py'))
    # shutil.copy(os.path.join(project_root, 'model.py'     ), os.path.join(output_folder,'model_'+exp+'.py'     ))

    return output_folder,timestamp, logger

def save_checkpoint(model, filepath):
    torch.save(model.state_dict(), filepath)



def update_plot(train_f1_class_0, train_f1_class_1, val_f1_class_0, val_f1_class_1, output_folder, epoch,fold):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(0,len(train_f1_class_0)*5,5), train_f1_class_0, label='Train F1-score Class 0')
    plt.plot(range(0,len(val_f1_class_0)*5,5), val_f1_class_0, label='Val F1-score Class 0')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.title('Class 0')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0,len(train_f1_class_1)*5,5), train_f1_class_1, label='Train F1-score Class 1')
    plt.plot(range(0,len(val_f1_class_1)*5,5), val_f1_class_1, label='Val F1-score Class 1')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.title('Class 1')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder,'logs_'+str(fold), f'classif_plot.png'))
    plt.close()

def create_visualization(all_labels, all_preds, save_path, box_width=6, box_height=24,fold=0, candidates=None):
    n_examples=int(len(all_labels)/840)
    label_image = np.ones((n_examples * box_height, 840 * box_width, 3), dtype=np.uint8)*255
    """
    for i in range(n_examples):
        for j in range(840):
            label_value = all_labels[i * 840 + j]
            
            # Determine the color for each box based on label
            if label_value == 0:
                label_image[i * box_height:(i + 1) * box_height, j * box_width:(j + 1) * box_width] = [0, 0, 0]  # Black for label 0
            else:
                label_image[i * box_height:(i + 1) * box_height, j * box_width:(j + 1) * box_width] = [0, 255, 0]  # Green for label 1
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(label_image, interpolation='nearest')
    
    # Add blue squares around predictions
    for i in range(n_examples):
        for j in range(840):

            if j >836:
                rect = patches.Rectangle((j * box_width, i * box_height), box_width, box_height, linewidth=0.3, edgecolor='white', facecolor='white')
                continue
            else:
                if all_labels[i * 840 + j] == 1:
                    rect = patches.Rectangle((j * box_width, i * box_height), box_width, box_height, linewidth=0.3, edgecolor='white', facecolor='green')
                else:
                    rect = patches.Rectangle((j * box_width, i * box_height), box_width, box_height, linewidth=0.3, edgecolor='white', facecolor='black')
                
                ax.add_patch(rect)

                if all_preds[i * 840 + j] == 1:
                    rect = patches.Rectangle((j * box_width+0.5, i * box_height+0.5), box_width-1, box_height-1, linewidth=0.3, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)

        if candidates is not None:
            rect = patches.Rectangle((candidates[i] * box_width+0.5, i * box_height+0.5), box_width-1, box_height-1, linewidth=0.3, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
            

    ax.set_title(f'Key Frames and Predictions (Fold {fold})')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path,dpi=500)
    plt.close()
    

def setup_logger(fold, log_dir):
    # Create a custom logger
    logger = logging.getLogger(f'fold_{fold}')
    logger.setLevel(logging.INFO)

    # Create handlers
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'fold_{fold}_{dt.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
