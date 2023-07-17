 #### Specify all the paths here #####

val_imgs = r'C:\My_Data\train\imgs'
val_masks = r'C:\My_Data\train\gts'

train_imgs = r'C:\My_Data\KeenAI\train\imgs'
train_masks = r'C:\My_Data\KeenAI\train\gts'

path_to_save_check_points='/data/scratch/acw676/'+'/DeepLabV3Plus_1'
path_to_save_Learning_Curve='/data/scratch/acw676/'+'/DeepLabV3Plus_1'


        #### Specify all the Hyperparameters\image dimenssions here #####

batch_size = 5
Max_Epochs = 200

        #### Import All libraies used for training  #####
import torch    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import cv2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import albumentations as A

            ### Data_Generators ########
            
            #### The first one will agument and Normalize data and used for training ###
            #### The second will not apply Data augmentaations and only prcocess the data during validation ###
NUM_WORKERS=0
PIN_MEMORY=True

# transform = A.Compose([
#     A.AdvancedBlur(blur_limit=(3, 7)),
#     A.HorizontalFlip(p=0.2),
#     A.VerticalFlip(p=0.2)
# ])

class Dataset_train(Dataset):
    def __init__(self, image_dir, mask_dir,transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        image=cv2.imread(img_path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.moveaxis(image,-1,0) 
        image = image/255
        
        mask=cv2.imread(mask_path,0)
        mask[np.where(mask!=0)]=1
        mask=np.expand_dims(mask, axis=0)

        if self.transform is not None:
            augmentations = self.transform(image=image,mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            
            
        #mask = np.stack((1-mask,mask), 0)    

        return image,mask,self.images[index][:-4]
    
def Data_Loader_train( image_dir,mask_dir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_train( image_dir=image_dir, mask_dir=mask_dir)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

class Dataset_val(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        image=cv2.imread(img_path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.moveaxis(image,-1,0) 
        image = image/255
        
        mask=cv2.imread(mask_path,0)
        mask[np.where(mask!=0)]=1
        mask=np.expand_dims(mask, axis=0)
        
        #mask = np.stack((1-mask,mask), 0)       
        return image,mask,self.images[index][:-4]
    
def Data_Loader_val( image_dir,mask_dir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_val( image_dir=image_dir, mask_dir=mask_dir)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

# UnetPlusPlus   DeepLabV3Plus 
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
NUM_CLASSES = 1
ACTIVATION = 'sigmoid'
Model = smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=NUM_CLASSES, 
    activation=ACTIVATION,
)


   ## Load the Data using Data generators and paths specified #####
   ######################################
train_loader=Data_Loader_train(train_imgs,train_masks,batch_size)
val_loader=Data_Loader_val(val_imgs,val_masks,batch_size)
print(len(train_loader)) ### this shoud be = Total_images/ batch size
print(len(val_loader))   ### same here


### Specify all the Losses (Train+ Validation), and Validation Dice score to plot on learing-curve
avg_train_losses = []   # losses of all training epochs
avg_valid_losses = []  #losses of all training epochs
avg_valid_DS = []  # all training epochs

### Next we have all the funcitons which will be called in the main for training ####


###  1- To stop the training before model overfits 
class EarlyStopping:
    def __init__(self, patience=None, verbose=True, delta=0,  trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score1 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.max_score = 0
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss,val_metric):

        score = -val_loss
        score1=-val_metric

        if (self.best_score is None) and (self.best_score1 is None):
            self.best_score = score
            self.best_score1 = score1
            self.verbose_(val_loss,val_metric)
        elif (score < self.best_score + self.delta) or (score1 > self.best_score1 + self.delta):
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score1 = score1
            self.verbose_(val_loss,val_metric)
            self.counter = 0

    def verbose_(self, val_loss,val_metric):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.trace_func(f'Validation metric increased ({self.max_score:.6f} --> {val_metric:.6f}).')
        self.val_loss_min = val_loss
        self.max_score = val_metric
        
### 2- the main training fucntion to update the weights....
def train_fn(loader_train,loader_valid, model, optimizer,loss_fn1, scaler):
    train_losses = [] # loss of each batch
    valid_losses = []  # loss of each batch
    loop = tqdm(loader_train)
    model.train()
    for batch_idx, (img1,gt1,label) in enumerate(loop):
        img1 = img1.to(device=DEVICE,dtype=torch.float)  
        gt1 = gt1.to(device=DEVICE,dtype=torch.float)
       
        # forward
        with torch.cuda.amp.autocast():
            out1= model(img1)   
            loss = loss_fn1(out1, gt1)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        train_losses.append(loss.item())
    
    loop_v = tqdm(loader_valid)
    model.eval()
    for batch_idx, (img1,gt1,label) in enumerate(loop_v):
        img1 = img1.to(device=DEVICE,dtype=torch.float)
        gt1 = gt1.to(device=DEVICE,dtype=torch.float)
        # forward
        with torch.no_grad():
            out1 = model(img1,)   
            loss = loss_fn1(out1, gt1)
        loop_v.set_postfix(loss=loss.item())
        valid_losses.append(loss.item())    
    train_loss_per_epoch = np.average(train_losses)
    valid_loss_per_epoch = np.average(valid_losses)
    ## all epochs
    avg_train_losses.append(train_loss_per_epoch)
    avg_valid_losses.append(valid_loss_per_epoch)
    
    return train_loss_per_epoch,valid_loss_per_epoch
    
### 3 - this function will save the check-points 
def save_checkpoint(state, filename=path_to_save_check_points+".pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

    ### 4 - It will check the Dice-Score on each epoch for validation data 
    
def check_Dice_Score(loader, model, device=DEVICE):
    dice_score1=0
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (img1,gt1,label) in enumerate(loop):
            img1 = img1.to(device=DEVICE,dtype=torch.float)
          
            gt1 = gt1.to(device=DEVICE,dtype=torch.float)
            p1 = model(img1)  
            
            p1 = (p1 > 0.5)* 1
            
            #p1 = torch.argmax(p1,axis=1)
            #gt1 = torch.argmax(gt1,axis=1)
            
            dice_score1 += (2 * (p1 * gt1).sum()) / (
                (p1 + gt1).sum() + 1e-8)
            
    print(f"Dice score: {dice_score1/len(loader)}")
    return dice_score1/len(loader)
    

### 5 - This is IoU loss function ###
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection   
        IoU = (intersection + smooth)/(union + smooth)          
        return 1 - IoU


### 6 - This is Focal Tversky Loss loss function ### 

ALPHA = 0.5
BETA = 0.5
GAMMA = 1

class FocalTverskyLoss(nn.Module):
    def _init_(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self)._init_()

    def forward(self, inputs, targets, smooth=.0001, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky
 

## 7- This is the main Training function, where we will call all previous functions
       
epoch_len = len(str(Max_Epochs))
early_stopping = EarlyStopping(patience=10, verbose=True)


def main():
    model = Model.to(device=DEVICE,dtype=torch.float)
    loss_fn1 =IoULoss()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(Max_Epochs):
        if epoch<10:
          LEARNING_RATE = 0.001
        if epoch>10:
          LEARNING_RATE = 0.0005
          
        optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.9),lr=LEARNING_RATE)
        train_loss,valid_loss=train_fn(train_loader,val_loader, model, optimizer, loss_fn1,scaler)
        
        print_msg = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)

        dice_score = check_Dice_Score(val_loader, model, device=DEVICE)
        
        
        avg_valid_DS.append(dice_score.detach().cpu().numpy())
        
        early_stopping(valid_loss, dice_score)
        if early_stopping.early_stop:
            print("Early stopping Reached at  :",epoch)
            
            ### save model    ######
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

            break
            
if __name__ == "__main__":
    main()


### This part of the code will generate the learning curve ......

avg_train_losses=avg_train_losses
avg_train_losses=avg_train_losses
avg_valid_DS=avg_valid_DS
# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(avg_train_losses)+1),avg_train_losses, label='Training Loss')
plt.plot(range(1,len(avg_valid_losses)+1),avg_valid_losses,label='Validation Loss')
plt.plot(range(1,len(avg_valid_DS)+1),avg_valid_DS,label='Validation DS')

# find position of lowest validation loss
minposs = avg_valid_losses.index(min(avg_valid_losses))+1 
plt.axvline(minposs,linestyle='--', color='r',label='Early Stopping Checkpoint')

font1 = {'size':20}

plt.title("Learning Curve Graph",fontdict = font1)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 1) # consistent scale
plt.xlim(0, len(avg_train_losses)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig(path_to_save_Learning_Curve+'.png', bbox_inches='tight')  
