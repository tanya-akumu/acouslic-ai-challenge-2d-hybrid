import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchvision.transforms as T
import SimpleITK as sitk
import random

from multiprocessing import Pool

class AcousticDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, n=0,sublen=80):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.max = []
        self.sublen = sublen
        if not n:
            self.image_files = sorted(os.listdir(images_dir))
            self.label_files = sorted(os.listdir(labels_dir))
        else:
            self.image_files = sorted(os.listdir(images_dir))[:n]
            self.label_files = sorted(os.listdir(labels_dir))[:n]

    def __len__(self):
        return len(self.image_files)

    def _read_image(self, image_path):
        image = sitk.ReadImage(image_path)
        return sitk.GetArrayFromImage(image)

    def _read_label(self, label_path):
        label = sitk.ReadImage(label_path)
        return sitk.GetArrayFromImage(label)

    def __getitem__(self, idx):



        image_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])


        image = self._read_image(image_path)
        label_full = self._read_image(label_path)


        if not self.sublen:
            if image.shape[0] > 840:
                label = label_full.swapaxes(-1,-3).swapaxes(-1,-2).max(axis=(1, 2))
                image = image.swapaxes(-1,-3).swapaxes(-1,-2)
                image_half = image[:,::2,::2]
            else:
                
                label = label_full.max(axis=(1, 2))
                image_half = image[:,::2,::2]
                

            
            image_half = torch.tensor(image_half, dtype=torch.float32).unsqueeze(0)
            image_half = (image_half - torch.min(image_half)) / (torch.max(image_half) - torch.min(image_half))
            label = torch.tensor(label).unsqueeze(0) 
            return image_half,label

        label = label_full.max(axis=(1, 2))


        new_i_1 = []
        new_i_0 = []
        if False:
            t = 0
            while t < len(label):
                l = label[t]
                if l > 0:
                    start = t
                    while t < len(label) and label[t] != 0:
                        t += 1
                    end = t

                    n_sel = int((end-start)*0.3)
                    central =int((end-start)//2)
                    new_i_1.append(image[start+central])  # Appending a slice of the image list
                    for x in range(1,n_sel):
                        new_i_1.append(image[start+central-x])
                        new_i_1.append(image[start+central+x])


                else:
                    new_i_0.append(image[t])
                    t += 1
        else:
            for t, l in enumerate(label):
                if l > 0 :
                    new_i_1.append(image[t])
                else:
                    new_i_0.append(image[t])
        del image

        new_len = self.sublen-len(new_i_1)
        new_i_0 = random.sample(new_i_0, new_len)
        
        new_l_1 = [1]*len(new_i_1)
        new_l_0 = [0]*len(new_i_0)

        new_i_0.extend(new_i_1)
        new_l_0.extend(new_l_1)

        new_i = np.stack(new_i_0)[:,::2,::2]
        del new_i_0

        # Count the number of items in the list that are not 0
        non_zero_count = np.count_nonzero(new_l_0)
        #print('undersampled',len(new_l_0),new_i.shape)
        self.max.append(non_zero_count)
        #print(f'Non-zero count: {non_zero_count}, Max: {np.max(self.max)}')

        image = torch.tensor(new_i, dtype=torch.float32).unsqueeze(0) 
        label = torch.tensor(new_l_0).unsqueeze(0) 
        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))


        return image, label

        

def create_dataloaders(dataset,splits,fold,batch_size=32, collate_fn=None, suffix='', logger=None):
   
    train_names = [n.replace('_down','')+suffix for n in splits[fold]['train']]
    val_names = [n.replace('_down','')+suffix for n in splits[fold]['val']]

    train_indices = [idx for idx,name in enumerate(dataset.image_files) if name in train_names]
    val_indices = [idx for idx,name in enumerate(dataset.image_files) if name in val_names]
    test_indices = [idx for idx,name in enumerate(dataset.image_files) if ((name not in val_names) and (name not in train_names))]

    if logger:
        print(f'Fold {fold} Train examples: {len(train_indices)}')
        print(f'Fold {fold} Val examples: {len(val_indices)}')
        print(f'Fold {fold} Test examples: {len(test_indices)}')

    train_sampler = SubsetRandomSampler(train_indices, generator=torch.Generator().manual_seed(fold))
    val_sampler = SubsetRandomSampler(val_indices, generator=torch.Generator().manual_seed(42))
    test_sampler = SubsetRandomSampler(test_indices, generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader,test_loader





