#%%
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from PIL import Image
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F

from model import SDP_CrossView_model, SDPL_ClsLoss

#%%
SUES_PATH_drone = 'E:\Datasets\SUES-200-512x512-V2\SUES-200-512x512\drone_view_512'
SUES_PATH_sattelite = 'E:\Datasets\SUES-200-512x512-V2\SUES-200-512x512\satellite-view'

#%%

class SUESDataset(Dataset):
    def __init__(self,
                 root_dir,
                 height,
                 num_classes,
                 transform):
        
        self.root = root_dir
        self.height = height
        self.num_classes = num_classes
        self.transform = transform
        self.dataframe = self.get_dataframe()

    def get_dataframe(self):
        fname_id = {'filename': [], 'id': []}

        for id in os.listdir(self.root):
            subdir = os.path.join(self.root, id)
            int_id = int(id)
            for hs in os.listdir(subdir):
                if ".png" in hs:
                    img_dir = subdir
                else:
                    img_dir = os.path.join(subdir, hs)
                if hs != self.height and self.height != 'all':
                    continue
                else:
                    for name in os.listdir(img_dir):
                        file_p = os.path.join(img_dir, name)
                        fname_id['filename'].append(file_p)
                        fname_id['id'].append(int_id)
        
        DF = pd.DataFrame.from_dict(fname_id)
        return DF

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        fname = row['filename']
        id = row['id']

        image = Image.open(fname).convert('RGB')
        image_tensor = self.transform(image)
        image.close()
       
        onehot_id = torch.zeros(self.num_classes, dtype=torch.float)
        onehot_id[id-1] = 1.
        
        return image_tensor, onehot_id, id

image_size = 512

data_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#%%
import copy
import random

drone_dataset = SUESDataset(SUES_PATH_drone, height='300', num_classes=200, transform=data_transforms)
sattelite_gallery = SUESDataset(SUES_PATH_sattelite, height='all', num_classes=200, transform=data_transforms)
sattelite_dataset = copy.deepcopy(sattelite_gallery)
sattelite_dataset.dataframe = pd.concat([sattelite_dataset.dataframe]*50, ignore_index=True)

eval_indexes = random.choices(range(0,10000), k=10)
drone_val_dict = {'filename':[], 'id':[]}
for i in eval_indexes:
    drone_val_dict['filename'].append(drone_dataset.dataframe.iloc[i]['filename'])
    drone_val_dict['id'].append(drone_dataset.dataframe.iloc[i]['id']) 
drone_val = copy.deepcopy(drone_dataset)
drone_val.dataframe = pd.DataFrame.from_dict(drone_val_dict)

#%%

drone_loader = DataLoader(drone_dataset, batch_size = 8, shuffle=True, drop_last=True)
sattelite_loader = DataLoader(sattelite_dataset, batch_size = 8, shuffle=True, drop_last=True)

#%%
from tqdm import tqdm
import time
import math

def train_epoch(model, dataloader1, dataloader2, device, optimizer, criterion, scheduler):
    
    mean_loss = []
    cur_tqdm = tqdm(zip(dataloader1, dataloader2))
    for drone_data, sattelite_data in cur_tqdm:

        drone_image, drone_label, _ = drone_data
        sattelite_image, sattelite_label, _ = sattelite_data

        drone_image = drone_image.to(device, non_blocking=True)
        drone_label = drone_label.to(device, non_blocking=True)
        sattelite_image = sattelite_image.to(device, non_blocking=True)
        sattelite_label = sattelite_label.to(device, non_blocking=True)

        drone_out = model(drone_image, mode='cls')
        sattelite_out = model(sattelite_image, mode='cls')

        loss = criterion(drone_out,
                         drone_label,
                         sattelite_out,
                         sattelite_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        mean_loss.append(loss.item())
        show_dict = {'Loss': f'{loss.item():.6f}'}
        # WANDB HERE LOSS ITARATIONS
        #wandb.log({"train/iteration_loss": loss.item()})
        cur_tqdm.set_postfix(show_dict)

    if not scheduler is None:
        scheduler.step()
    print("Calculating average loss...")
    epoch_loss = sum(mean_loss) / len(mean_loss)
    # WANDB EPOCH LOSS
    return {"epoch_loss": epoch_loss}

def run_model_training(model, 
                       dataloader1, 
                       dataloader2,
                       val_set,
                       gallery, 
                       n_epochs,
                       device, 
                       optimizer, 
                       criterion,
                       scheduler,
                       save_weights_path = 'model_weights.ckpt'):
    
    phases = ['test', 'train']
    save_metrics = {'loss':[], 'R1':[], 'R5':[], 'AP':[]}
    start_epoch = 1
    for epoch in range(start_epoch, n_epochs+1):
        start_time = time.time()
        print("=" * 100)
        print(f'Epoch {epoch}/{start_epoch+n_epochs}')
        print('-' * 10)
        
        for phase in phases:
            print(f"Current phase: {phase}")
            if phase == 'train':
                model.train()
                epoch_loss = train_epoch(model, dataloader1, dataloader2, device, optimizer, criterion, scheduler)
            elif epoch % 1 == 0:
                with torch.inference_mode():
                    model.eval()
                    average_r1 = 0
                    average_r5 = 0
                    mean_ap = 0
                    set_length = len(val_set)
                    for i in tqdm(range(set_length)):
                        rank = dict()
                        q_img, _, q_id = val_set[i]
                        q_img = q_img.to(device, non_blocking=True)
                        q_embedding = model(q_img.unsqueeze(0), mode='retrieve')
                        q_embedding = q_embedding.reshape(1, -1).squeeze(0)
                        for j in range(len(gallery)):
                            r_img, _, r_id = gallery[j]
                            r_img = r_img.to(device, non_blocking=True)
                            r_embedding = model(r_img.unsqueeze(0), mode='retrieve')
                            r_embedding = r_embedding.reshape(1, -1).squeeze(0)
                            rank[r_id] = F.mse_loss(r_embedding, q_embedding)

                        sorted_rank = {k: v for k, v in sorted(rank.items(), key=lambda item: item[1])}
                        list_rank = list(sorted_rank.items())
                        print(f"{q_id} \n{list_rank[0:10]}")

                        r1 = int(list_rank[0][0]==q_id)
                        average_r1 += r1

                        r5 = 0
                        for k in range(5): 
                            r5 += int(list_rank[k][0]==q_id)
                        average_r5 += r5

                        ap = 0
                        cnt = 0
                        for k in range(len(list_rank)):
                            rel = int(list_rank[k][0]==q_id)
                            cnt += rel
                            ap += (cnt*rel/(k+1))
                        mean_ap += ap
                    average_r1 /= set_length
                    average_r5 /= set_length
                    mean_ap /= set_length
                    #wandb.log({'recall@1': average_r1, 'recall@5': average_r5, 'AP': mean_ap})

        
        if epoch % 1 == 0:
            print(f'Epoch mean loss: {epoch_loss}')
            print(f'Average Recall@1: {average_r1}')
            print(f'Average Recall@1: {average_r5}')           
            print(f'Mean AP: {mean_ap}')
            #wandb.log({'recall@1': average_r1, 'recall@5': average_r5, 'AP': mean_ap})
            save_metrics['R1'].append(average_r1)
            save_metrics['R5'].append(average_r5)
            save_metrics['AP'].append(mean_ap)
            
        save_metrics['loss'].append(epoch_loss)    
        #wandb.log({'mean_loss': epoch_loss})
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch Time: {math.floor(epoch_time // 60)}:{math.floor(epoch_time % 60):02d}")
        print("-" * 10)

        if epoch % 2 == 0:
            torch.save(model.state_dict(), save_weights_path)

        
    print("*** Training Completed ***")
    return save_metrics


# %%
DEVICE = 'cpu'
model = SDP_CrossView_model().to(DEVICE)
criterion = SDPL_ClsLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
schedul = torch.optim.lr_scheduler.StepLR(optim,step_size=1,gamma=0.997, verbose=True)

#%%

saved = run_model_training(model=model,
                           dataloader1=drone_loader,
                           dataloader2=sattelite_loader,
                           val_set=drone_val,
                           gallery=sattelite_gallery,
                           n_epochs=2,
                           device=DEVICE,
                           optimizer=optim,
                           criterion=criterion,
                           scheduler=schedul)

# %%
