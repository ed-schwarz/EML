#library imports
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import torch.nn as nn
from torch import optim
from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import json
import random



class Generator(nn.Module):
    def __init__(self, latent_dim,channel_num):
        super(Generator,self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim,    channel_num,     4, 1, 0),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(),
            nn.ConvTranspose2d(channel_num,   channel_num//2,  4, 2, 1),
            nn.BatchNorm2d(channel_num//2),
            nn.ReLU(),
            nn.ConvTranspose2d(channel_num//2, channel_num//4, 4, 2, 1),
            nn.BatchNorm2d(channel_num//4),
            nn.ReLU(),
            nn.ConvTranspose2d(channel_num//4, channel_num//8, 4, 2, 1),
            nn.BatchNorm2d(channel_num//8),
            nn.ReLU(),
            nn.ConvTranspose2d(channel_num//8, channel_num//16, 4, 2, 1),
            nn.BatchNorm2d(channel_num//16),
            nn.ReLU(),
            nn.ConvTranspose2d(channel_num//16, channel_num//32, 4, 2, 1),
            nn.BatchNorm2d(channel_num//32),
            nn.ReLU(),
            nn.ConvTranspose2d(channel_num//32, 3,              4, 2, 1),
            nn.Tanh())

    def forward(self,x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, channel_num):
        super(Discriminator,self).__init__()
        self.net = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(3,              channel_num//16, 4, 2, 0)),
            nn.LeakyReLU(0.2),
            torch.nn.utils.spectral_norm(nn.Conv2d(channel_num//16, channel_num//8, 4, 2, 0)),
            nn.BatchNorm2d(channel_num//8),
            nn.LeakyReLU(0.2),
            torch.nn.utils.spectral_norm(nn.Conv2d(channel_num//8, channel_num//4, 4, 2, 0)),
            nn.BatchNorm2d(channel_num//4),
            nn.LeakyReLU(0.2),
            torch.nn.utils.spectral_norm(nn.Conv2d(channel_num//4, channel_num//2, 4, 2, 0)),
            nn.BatchNorm2d(channel_num//2),
            nn.LeakyReLU(0.2),
            torch.nn.utils.spectral_norm(nn.Conv2d(channel_num//2, channel_num,    4, 2, 0)),
            nn.BatchNorm2d(channel_num),
            nn.LeakyReLU(0.2),
            torch.nn.utils.spectral_norm(nn.Conv2d(channel_num,    1,              4, 3, 0)),
            nn.Sigmoid())
    def forward(self, input):
        return self.net(input)


def select_subset(choosen_categories, train_dataset):
    train_subset = []

    for LABEL_IDX in choosen_categories:
        if LABEL_IDX != 102:
            train_indices_i = [i for i, (e, c) in enumerate(train_dataset) if c in [LABEL_IDX]]
            train_subset_i = torch.utils.data.Subset(train_dataset, train_indices_i)
        else:
            train_subset_i = train_dataset

        train_subset += train_subset_i

    return train_subset

def flower_name_to_class(flower_names):
    classlabel = []
    for flowername in flower_names:
        classlabel_i = int(flowername_to_label[flowername])-1
        classlabel += [classlabel_i]

    return classlabel

def plot_grid(data, n_row):
    grid_img = make_grid(data, nrow=n_row)
    plt.imshow(grid_img.permute(1,2,0))
    plt.axis('off')
    plt.show()


def get_one_of_label(data, label_goal):
    train_indices = [i for i, (e, c) in enumerate(data) if c in [label_goal]]
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    return train_subset[torch.randint(len(train_subset), (1,))]

def get_plot_data(data, labels):
    data_plot = []
    data_plot_out = []
    for label in labels:
        data_i = get_one_of_label(data, label)
        data_plot += data_i

        train_dataloader_i = DataLoader(data_i)
        data_plot_i = next(iter(train_dataloader_i))
        data_plot_out += data_plot_i

    return data_plot_out




# the size of the images is (500 x (something > 500)) or ((something > 500) x 500)
# we choose the maximum crop to obtain squared images
IMAGE_CROP = 500
# fix the the image size according to the problem description
IMAGE_SIZE_TRANSFORMER = 256

# specify data transformations
transform = transforms.Compose([
    # crop the images to be squared
    transforms.CenterCrop(IMAGE_CROP),
    # resize the images to the desired resolution
    transforms.Resize(IMAGE_SIZE_TRANSFORMER),
    # convert images to tensors and scale 
    transforms.ToTensor(),
    # normalize images to have values in [-1,1] in each channel
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# read flower training dataset 
train_dataset = datasets.Flowers102(
    root="data", 
    split="train",
    download=True,
    transform=transform
)

# read flower validation dataset 
val_dataset = datasets.Flowers102(
    root="data",
    split="val",
    download=True,
    transform=transform
)

# read flower test dataset 
test_dataset = datasets.Flowers102(
    root="data",
    split="test",
    download=True,
    transform=transform
)


# join all datasets into one as we want to select images from the whole dataset
flower_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset, test_dataset])

#print('Total number of image samples in dataset:',len(flower_dataset))


# read 'label : flower category' dictionary
# note: class labels are in [0,1,...,101], in the the dictionary labels are in [1,2,...,102]
with open('./Homework3/flower-categories.json', 'r') as f:
    label_to_flowername = json.load(f)








# reversed dictionary: switch label and flower name
flowername_to_label= dict((v, k) for k, v in label_to_flowername.items()) 

flower_names = ['lotus', 'magnolia', 'passion flower', 'water lily', 'anthurium', 'foxglove', 'geranium', 'clematis', 'hibiscus', 'sword lily', 'rose', 'snapdragon']


choosen_categories = flower_name_to_class(flower_names)

#choosen_categories = [102]
train_subset = select_subset(choosen_categories, flower_dataset)


indices = torch.arange(1600)
train_subset_cap = torch.utils.data.Subset(train_subset, indices)

BATCH_SIZE = 50
train_dataloader = DataLoader(train_subset_cap,batch_size=BATCH_SIZE,shuffle=True)
print(len(train_dataloader.sampler))


data_batch, labels_batch = next(iter(train_dataloader))

#plot_grid(get_plot_data(train_dataset, choosen_categories), 10)









device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)


# dimension of latent space
LATENT_DIM = 128
# number of channels in first convolutional layer of generator
CHANNEL_NUM = 512

modelGen = Generator(LATENT_DIM, CHANNEL_NUM).to(device)    
modelDis = Discriminator(CHANNEL_NUM).to(device)   

print(modelGen)
print(modelDis)


# use weight initialization as suggested in the DCGAN paper: N(0,0.02)
# initialization in batch norm layer: N(1,0.02) -> see meaning of learnable parameters in batch norm layer
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)
        
modelDis.apply(weights_init);
modelGen.apply(weights_init);



# sample BATCH_SIZE vectors of dimension LATENT_DIM from standard normal distribution
z = torch.randn((BATCH_SIZE, LATENT_DIM)).view(-1,LATENT_DIM,1,1).to(device)

# generate batch of images with generator, renormalize output from range [-1,1] to [0,1]
gen_output_batch = (modelGen(z) + 1) / 2

gen_images_batch = gen_output_batch.to('cpu').detach()
#plot_grid(gen_images_batch, 10)

#grid_img = make_grid(gen_images_batch, nrow=10)
#plt.imshow(grid_img.permute(1,2,0))
#plt.axis('off')
#plt.show()




#count and print number of parameters: total/trainable 
num_params_modelGen = sum(param.numel() for param in modelGen.parameters())
num_train_params_modelGen = sum(p.numel() for p in modelGen.parameters() if p.requires_grad)

num_params_modelDis = sum(param.numel() for param in modelDis.parameters())
num_train_params_modelDis = sum(p.numel() for p in modelDis.parameters() if p.requires_grad)

print('Generator:')
print('Number parameters:', num_params_modelGen)
print('Trainable parameters:', num_train_params_modelGen,'\n')
print('Discriminator:')
print('Number parameters:', num_params_modelDis)
print('Trainable parameters:', num_train_params_modelDis,'\n')
print('Total:')
print('Number parameters:', num_params_modelGen+num_params_modelDis)
print('Trainable parameters:', num_train_params_modelGen+num_train_params_modelDis)


# binary entropy loss: used in the right way it provides us both: log(D(x)) and log(1−D(G(z)))
criterion = nn.BCELoss()

# optimizers as suggested in DCGAN paper with similar parameters

# adam optimizer for generator
optimizerGen = optim.Adam(modelGen.parameters(),
                       lr = 0.0001, #0.0001
                       betas=(0.5,0.999))

# adam optimizer for discriminator
optimizerDis = optim.Adam(modelDis.parameters(),
                      lr=0.0001, #0.0001
                      betas=(0.5,0.999))

# these probability labels are used during training to label training images and generated images 
# as 'real' and 'fake'; allows to use the BCELoss() function appropriately
real_labels = torch.full((BATCH_SIZE,),
                       0.95,
                       dtype=torch.float,
                       device=device)

fake_labels = torch.full((BATCH_SIZE,),
                        0.05,
                        dtype=torch.float,
                        device=device)

# sample BATCH_SIZE vectors of dimension LATENT_DIM from standard normal distribution
# to test and visualize image generation during training
z_val = torch.randn((BATCH_SIZE, LATENT_DIM)).view(-1,LATENT_DIM,1,1).to(device)


# fix number of epochs
EPOCHS = 2000

# initialize loss trackers
av_batch_loss_gen = []
av_batch_loss_dis = []
av_batch_loss_dis_real = []
av_batch_loss_dis_fake = []

# initialize discriminator output trackers
D_real = []
D_fake = []

# training loop
for epoch in range(EPOCHS):
    
    batch_loss_gen = []
    batch_loss_dis = []
    batch_loss_dis_real = []
    batch_loss_dis_fake = []
    
    print(f'Epoch {epoch+1}')
    p_bar = tqdm(train_dataloader, total=(len(train_dataloader)), desc="Training...")
    
    for batch in p_bar:
        
        ################################################
        ## UPDATE DISCRIMINATOR                       ##
        ## maximize  log(D(x)) + log(1 - D(G(z))) via ##
        ## minimize -log(D(x)) - log(1 - D(G(z)))     ##
        ################################################

        # train discriminator with batch of real images
        modelDis.zero_grad()
        # rescale image values from [0,1] to [-1,1]
        real_images = batch[0].to(device) *2. - 1.
        # feed real images into discriminator
        output_dis = modelDis(real_images).view(-1)
        
        # calculate loss on batch of real images, labels are used to select -log(D(x)) part of BCELoss()
        lossD_real = criterion(output_dis,real_labels)        
        # calculate gradients for discriminator in backward pass
        lossD_real.backward()
            
        # averaged discriminator output for real images            
        D_x = output_dis.mean().item()
        
        # train discriminator with batch of fake images
        # generate batch of latent vectors
        z = torch.randn((BATCH_SIZE, LATENT_DIM)).view(-1,LATENT_DIM,1,1).to(device)        
        # generate fake image batch with generator
        fake_images = modelGen(z)
        #print(fake_images.size())
        # feed fake images into discriminator, .detach() required here since the generator gradients 
        # should not be calculated/updated at this point
        output_dis = modelDis(fake_images.detach()).view(-1)
        #print(modelDis(fake_images).size())
        # calculate loss on batch of fake images, labels are used to select -log(1-D(G(z))) part of BCELoss()
        lossD_fake = criterion(output_dis,fake_labels)
        # calculate gradients for discriminator in backward pass, accumulated with previous gradients
        lossD_fake.backward()
        
        # averaged discriminator output for fake images
        D_G_z1 = output_dis.mean().item()
        
        # compute loss of dicrimintor as sum over fake and real images
        lossD = lossD_real + lossD_fake
        
        # update discriminator
        optimizerDis.step()
        
        ################################################################
        ## UPDATE GENERATOR                                           ##
        ## maximize log(D(G(z))) INSTEAD of minimize log(1 - D(G(z))) ##
        ## via minimize -log(D(G(z)))                                 ##
        ################################################################
        
        # train generator with fake images
        modelGen.zero_grad()
        # feed fake images into the just updated discriminator
        output_dis = modelDis(fake_images).view(-1)
        # calculate loss on batch of fake images, labels are used to select -log(D(G(z))) part of BCELoss()
        lossG = criterion(output_dis, real_labels)
        # calculate gradients for generator in backward pass (also for discriminator, but not used)
        lossG.backward()
        
        # averaged output of updated descriminator for fake images
        D_G_z2 = output_dis.mean().item()
        
        # update generator
        optimizerGen.step()
        
        # track batch losses
        batch_loss_dis.append(lossD.item())
        batch_loss_dis_real.append(lossD_real.item())
        batch_loss_dis_fake.append(lossD_fake.item())
        batch_loss_gen.append(lossG.item())
        
        # track discriminator output for real and fake images (averaged per batch)
        D_real.append(D_x)
        D_fake.append(D_G_z2)
      
        p_bar.set_postfix({'Generator loss': f'{lossG.item():.5f}','Discriminator loss': f'{lossD.item():.5f}'})
        p_bar.update()

    # track average batch losses
    av_batch_loss_gen.append(np.mean(batch_loss_gen))
    av_batch_loss_dis.append(np.mean(batch_loss_dis))
    av_batch_loss_dis_real.append(np.mean(batch_loss_dis_real))
    av_batch_loss_dis_fake.append(np.mean(batch_loss_dis_fake))
    
    print(f'Avg batch loss for epoch {epoch+1}: Generator: {np.mean(batch_loss_gen):.5f} Discriminator: {np.mean(batch_loss_dis):.5f} (Real: {np.mean(batch_loss_dis_real):.5f} / Fake: {np.mean(batch_loss_dis_fake):.5f})')
    
    # generate batch of images with generator, renormalize output from range [-1,1] to [0,1]
fake_images_val = (modelGen(z_val) + 1) / 2
fake_images_val = fake_images_val.to('cpu').detach()
grid_img = make_grid(fake_images_val, nrow=10)
plot_grid(grid_img, 10)
    #plt.imshow(grid_img.permute(1,2,0))
    #plt.axis('off')
    #plt.show()  
''' 
'''