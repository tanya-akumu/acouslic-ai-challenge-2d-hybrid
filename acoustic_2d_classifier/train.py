import os
import tqdm
import torch
import numpy as np
from sklearn.metrics import classification_report
from utils import update_plot, save_checkpoint
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import random

transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),        # Random horizontal flip
    T.RandomVerticalFlip(p=0.5),          # Random vertical flip
    T.RandomApply([
    T.RandomAffine(degrees=60, 
                   translate=(0.3, 0.3), 
                   scale=(0.7, 1.3), 
                   shear=(-10, 10))], p=0.8), 
   T.RandomApply([
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
    ], p=0.8),
    T.Normalize(mean=[0.5], std=[0.25])     # Normalize for grayscale
])


transform_inf= T.Compose([
    T.Normalize(mean=[0.5], std=[0.25])
])


def augment_batch(batch, p=0,inf=False):
    #probabilistic 2d augmentation

    b, c, h, w = batch.size()
    nb = []
    for i in range(b):
        image = batch[i]
        if inf:
            image = transform_inf(image)
        else:
        
            if np.random.random()> p:
                image = transform(image)
            else:
                image = transform_inf(image)

        nb.append(image)
    augmented_batch = torch.stack(nb,dim =0)
    return augmented_batch



def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, criterion, device, total_epochs, output_folder,epoch_save,fold, logger):
    train_f1_class_0 = []
    train_f1_class_1 = []
    val_f1_class_0 = []
    val_f1_class_1 = []

    for epoch in tqdm.tqdm(range(total_epochs),desc='Epoch'):


        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch}: Learning Rate = {current_lr:.6f}")
    
        for data, target in tqdm.tqdm(train_loader,desc='Train epoch'):

            data_list, target_list = preprocess_batch(data, target, undersample= None)
            dataloader_2d_chain = create_balanced_dataloader(data_list, target_list, device)

            for inputs, labels in dataloader_2d_chain:

                optimizer.zero_grad()
                outputs = model(augment_batch(inputs).to(device))

                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

        scheduler.step()

        if epoch%epoch_save == 0 :
            # not updating the loss
            criterion = evaluate(model, train_loader, val_loader, train_f1_class_0, train_f1_class_1, val_f1_class_0, val_f1_class_1, device, output_folder, epoch,fold, logger)
            update_plot(train_f1_class_0, train_f1_class_1, val_f1_class_0, val_f1_class_1, output_folder, epoch,fold)




def preprocess_batch(data, target, undersample= None):
    
    # this is to do a balanced sampling among all the subjects in the batch,
    # we convert the 3Ds to 2D slices for the custom dataset
    # also, we include probabilistic undersample for background
    data_list = []
    target_list = []
    for b in range(data.shape[0]):
        for t in range(data.shape[2]):
            label = torch.clamp(torch.max(target[b, 0, t, ...]), 0, 1)
            if undersample is not None:
                if np.random.random() > undersample and int(label.cpu().numpy()) == 0:
                    continue 
                else:
                    data_list.append(data[b, 0, t, ...])
                    target_list.append(torch.clamp(torch.max(target[b, 0, t, ...]), 0, 1))

            else:
                data_list.append(data[b, 0, t, ...])
                target_list.append(torch.clamp(torch.max(target[b, 0, t, ...]), 0, 1))
    return data_list, target_list


def preprocess_batch_inf(data, target=None):
    
    data_list = []
    target_list = []
    for b in range(data.shape[0]):
        batched_data_list = []
        batched_target_list = []
        for t in range(data.shape[1]):
            batched_data_list.append(data[b, t, ...].unsqueeze(0))
            if target is not None:
                batched_target_list.append(torch.clamp(torch.max(target[b, t, ...]), 0, 1).unsqueeze(0))
                target_list.append(torch.stack(batched_target_list))
        data_list.append(torch.stack(batched_data_list))
        

    return data_list, target_list


def create_balanced_dataloader(data_list, target_list, device):
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

    class CustomDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx].unsqueeze(0).to(device), self.labels[idx].to(device)

    dataset = CustomDataset(data_list, target_list)
    class_counts = np.bincount([l.cpu().detach().numpy() for l in target_list])
    sample_weights = [1. / class_counts[int(label.cpu().detach().numpy())] for label in target_list]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(dataset, batch_size=8, sampler=sampler)

def evaluate(model, train_loader, val_loader, train_f1_class_0, train_f1_class_1, val_f1_class_0, val_f1_class_1, device, output_folder, epoch,fold, logger):

    model.eval()
    train_report,tr_print = generate_classification_report(model, train_loader, device)
    val_report,vl_print = generate_classification_report(model, val_loader, device)
    train_f1_class_0.append(train_report['0']['f1-score'])
    train_f1_class_1.append(train_report['1']['f1-score'])
    val_f1_class_0.append(val_report['0']['f1-score'])
    val_f1_class_1.append(val_report['1']['f1-score'])

    criterion = torch.nn.CrossEntropyLoss()

    
    if val_report['0']['f1-score'] == 0 or val_report['1']['f1-score'] == 0:
        criterion = torch.nn.CrossEntropyLoss()

    else:
        weights = torch.tensor([1 / (val_report['0']['f1-score']), 1 / (val_report['1']['f1-score'])])
        criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    
    logger.info(f'New weights for CE {criterion.weight}')
    logger.info(f'Epoch {epoch} Train report')
    logger.info(tr_print)
    logger.info(f'Epoch {epoch} Validation report')
    logger.info(vl_print)
    
    val_class1_precision = val_report['1']['precision']
    macro_avg_f1 = val_report['macro avg']['f1-score']

    checkpoint_name = f'_{val_class1_precision:.3f}_{macro_avg_f1:.3f}.pth'
    checkpoint_path = os.path.join(output_folder, f'logs_{fold}', f'checkpoints_{fold}', f'checkpoint_epoch_{epoch}_{checkpoint_name}')
    save_checkpoint(model, checkpoint_path)
    return criterion

def generate_classification_report(model, data_loader, device):
    all_preds = []
    all_labels = []

    for data, target in tqdm.tqdm(data_loader,desc="Infering..."):
        data_list, target_list = preprocess_batch(data, target,undersample=0.1)
        for inputs, labels in zip(data_list, target_list):
            with torch.no_grad():
                inputs= inputs.unsqueeze(0).unsqueeze(0)
                augmented = augment_batch(inputs,inf=True).to(device)
                outputs = model(augmented)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.append(labels.numpy())
    return classification_report(all_labels, all_preds, output_dict=True),classification_report(all_labels, all_preds, output_dict=False)


def return_logits(model, data_loader, device,tta=False):
    model.eval()
    all_preds = []
    all_labels = []
    for data, target in tqdm.tqdm(data_loader):
        data_list, target_list = preprocess_batch(data, target)
        for inputs, labels in zip(data_list, target_list):
            with torch.no_grad():
                inputs= inputs.unsqueeze(0).unsqueeze(0)
                if not tta:
                    
                    augmented = augment_batch(inputs,inf=True).to(device)
                    outputs = model(augmented)
                    all_preds.extend(outputs[:,1].cpu().numpy())
                    
                else:
                    preds = []
                    for x in range(5):
                        augmented = augment_batch(inputs,inf=False).to(device)
                        outputs = model(augmented)
                        preds.extend(torch.nn.Softmax()(outputs)[:,1].cpu().numpy())
                    all_preds.append(np.mean(preds))
                
                
                all_labels.append(labels.numpy())
    return all_labels, all_preds










