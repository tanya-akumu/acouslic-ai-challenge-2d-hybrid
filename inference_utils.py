from acoustic_2d_classifier.model import  initialize_model
from acoustic_2d_classifier.train import preprocess_batch_inf, augment_batch, preprocess_batch
import numpy as np
from pathlib import Path
import torch
import os


def create_model(device):
  ''' 
  intialize model 
  '''
    model, _, _, _ = initialize_model(device,
                                    warmup=10,
                                    weights=None)
    
    return model


def get_best_models(model_dir, device):
  '''
  Load model checkpoints
  '''
    models = []
    model_files = os.listdir(model_dir)
    for file in model_files:
        model = create_model(device)
        checkpoint = torch.load(os.path.join(model_dir, file))
        model.load_state_dict(checkpoint)
        model.eval()
        models.append(model)
        

    return models


def _return_logits(model, image,device):
  '''
  Run forward pass 
  '''
    image = image.unsqueeze(0)

    all_preds = []
    data_list, target_list = preprocess_batch(image, image)
    for inputs in data_list:
        inputs= inputs.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            augment = augment_batch(inputs,inf=True).to(device)
            outputs = model(augment) #.swapaxes(-1,-2))
            preds = torch.nn.Softmax(dim=1)(outputs).cpu().numpy()
            all_preds.extend([preds[i,1].squeeze() for i in range(preds.shape[0])])

    return torch.tensor(all_preds)

def get_ensemble_logits_mean(models,image, device):
    '''
    Gets average logits from all models
    '''
    all_probs = []
    for model in models:
        probs = _return_logits(model, image,device)
        all_probs.append(probs)

    ensemble_probs = torch.stack(all_probs).mean(dim=0)

    return ensemble_probs, all_probs

def get_ensemble_logits_multiplicative(models,image, device):
    '''
    Gets geometric mean of logits from all models
    '''
    all_probs = []
    for model in models:
        probs = _return_logits(model, image,device)
        all_probs.append(probs)
        
    ensemble_probs = torch.prod(torch.stack(all_probs),dim=0)

    return ensemble_probs, all_probs


def get_top_k_labels(prob_list, label_list=None, k=7):
    '''
    Gets the top k labels from the model probabilities
    '''
    # Get the indexes of the top k probabilities
    top_k_indexes = list(np.argsort(prob_list)[-k:])
    

    # Count occurrences of labels 1 and 2
    count_label_1 = 0
    count_label_2 = 0
    
    return top_k_indexes[::-1] 
