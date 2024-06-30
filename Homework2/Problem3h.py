# imports for data handling, network definition, and training
import numpy as np

import torch
from torch import nn

from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# imports for illustrations
from tqdm import tqdm
import matplotlib.pyplot as plt

# read MNIST training data
train_data = datasets.EMNIST(
    root="data",
    train=True,
    download=True,
    split='letters',
    transform=ToTensor(),
)

# read MNIST test/validation data
val_data = datasets.EMNIST(
    root="data",
    train=False,
    download=True,
    split='letters',
    transform=ToTensor(),
)
class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.clayer_1 = nn.Sequential( # convolution -> relu
            nn.Conv2d(1,8,5,1,2),         # 1 input channel, 8 output channels, 5x5 kernel, 1 stride, 4 padding
            nn.ReLU())           
        
        self.clayer_2 = nn.Sequential( # convolution -> relu -> max pooling
            nn.Conv2d(8,16,5,1),         # 8 input channels, 16 output channels, 5x5 kernel, 1 stride
            nn.ReLU(),
            nn.MaxPool2d(2,2))          # 2x2 max pooling
        
        self.clayer_3 = nn.Sequential( # convolution -> relu -> max pooling
            nn.Conv2d(16,32,5,1),         # 16 input channels, 32 output channels, 5x5 kernel, 1 stride
            nn.ReLU(),
            nn.MaxPool2d(2,2))          # 2x2 max pooling
        
        self.llayer_1 = nn.Sequential( # fully connected linear layer -> relu
            nn.Linear(512,128),        # input size 512 , output size 128
            nn.ReLU())
        
        self.llayer_2 = nn.Sequential( # fully connected linear layer -> relu
            nn.Linear(128,64),         # input size 128, output size 64
            nn.ReLU())
        
        self.llayer_3 = nn.Sequential( # fully connected linear layer -> softmax
            nn.Linear(64,27),         # input size 64, output size 64
            nn.Softmax(dim=1))
        
    def forward(self,x):
        x = self.clayer_1(x)
        x = self.clayer_2(x)
        x = self.clayer_3(x)
        x = x.view(-1, 512) # flatten the tensor for the fully connected layers
        x = self.llayer_1(x)
        x = self.llayer_2(x)
        x = self.llayer_3(x)
        return x
    
class CNNh(nn.Module):
    
    def __init__(self):
        super(CNNh, self).__init__()
        self.clayer_1 = nn.Sequential( # convolution -> relu
            nn.Conv2d(1,16,7,1),         # 1 input channel, 8 output channels, 5x5 kernel, 1 stride, 4 padding
            nn.LeakyReLU())           
        
        
        self.clayer_4 = nn.Sequential( # convolution -> relu -> max pooling
            nn.Conv2d(16,32,5,1),         # 16 input channels, 32 output channels, 5x5 kernel, 1 stride
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2))          # 2x2 max pooling  
        
        self.clayer_5 = nn.Sequential( # convolution -> relu -> max pooling
            nn.Conv2d(32,64,5,1),         # 16 input channels, 32 output channels, 5x5 kernel, 1 stride
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2))          # 2x2 max pooling  
                    
        self.llayer_1 = nn.Sequential( # fully connected linear layer -> relu
            nn.Linear(256,30),        # input size 512 , output size 128
            nn.LeakyReLU())
        
        
        self.llayer_2 = nn.Sequential( # fully connected linear layer -> softmax
            nn.Linear(30,27),         # input size 64, output size 64
            nn.Softmax(dim=1))
        
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self,x):
        x = self.clayer_1(x)
        x = self.clayer_4(x)
        x = self.clayer_5(x)
        x = x.view(-1, 256) # flatten the tensor for the fully connected layers
        x = self.dropout(self.llayer_1(x))
        x = self.llayer_2(x)
        return x
    
# define training loop over batches
def train_loop(dataloader, model, loss_fn, optimizer,device):
    size_train_set = len(dataloader.dataset)
    num_train_batches = len(dataloader)
    av_train_loss, train_accuracy = 0, 0
    p_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training...")
    model = model.to(device)
    for batch, (X, y) in p_bar:
        X = X.to(device)
        y = y.to(device)
        # compute prediction and loss
        yhat = model(X)

        loss = loss_fn(yhat, y)
        
        av_train_loss += loss.item()
        train_accuracy += (yhat.argmax(1) == y).type(torch.float).sum().item()

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        p_bar.set_postfix({'Avg training loss (batch)': f'{loss.item():.4f}'})
    
    # average prediction and loss (over training dataset)
    av_train_loss /= num_train_batches
    train_accuracy /= size_train_set
    print(f"Train Result: \n Accuracy: {(100*train_accuracy):>0.1f}%, Avg training loss (dataset): {av_train_loss:>8f} \n") 
    
    return av_train_loss, train_accuracy*100
    
# define validation loop
def val_loop(dataloader, model, loss_fn,device):
    size_val_set = len(dataloader.dataset)
    num_val_batches = len(dataloader)
    av_val_loss, val_accuracy = 0, 0
    model = model.to(device)    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            yhat = model(X)
            av_val_loss += loss_fn(yhat, y).item()
            val_accuracy += (yhat.argmax(1) == y).type(torch.float).sum().item()

    # average prediction and loss (over validation dataset)        
    av_val_loss /= num_val_batches
    val_accuracy /= size_val_set
    print(f"Validation Result: \n Accuracy: {(100*val_accuracy):>0.1f}%, Avg validation loss (dataset): {av_val_loss:>8f} \n")
    return av_val_loss, val_accuracy*100

# define loop over epochs
def loop_over_epochs(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs,device,target_acc):
    # prepare training / validation accuracy recording
    train_accuracy_log = []
    val_accuracy_log = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        _ , train_accuracy = train_loop(train_dataloader, model, loss_fn, optimizer,device)
        _, val_accuracy = val_loop(val_dataloader, model, loss_fn,device)
        train_accuracy_log.append(train_accuracy)
        val_accuracy_log.append(val_accuracy) 
        if val_accuracy > target_acc*100: 
            print('break')
            break         
    return train_accuracy_log, val_accuracy_log

# define plot function
def plot_accuracy(train_accuracy_log, val_accuracy_log):
    axes = plt.figure().gca() 
    axes.xaxis.get_major_locator().set_params(integer=True) 
    plt.xlabel("epochs")
    plt.ylabel("accuracy (%)")
    plt.plot(np.arange(len(train_accuracy_log))+1, train_accuracy_log, color='blue')
    plt.plot(np.arange(len(val_accuracy_log))+1, val_accuracy_log, color='red')
    plt.legend(['train accuracy', 'validation accuracy'], loc='lower right')
    plt.show()
    plt.clf()  



#change to True to train net and store it, to false to load already trained net
train = False

classes = val_data.classes
# choose gpu device if available, otherwise cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
model = CNNh().to(device)
# print network architecture
print(model)
model_pre_trained = CNN().to(device)

if(train):

    
    #count and print number of parameters: total/trainable
    num_total_params = sum(param.numel() for param in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_trainable_params_prev = sum(p.numel() for p in model_pre_trained.parameters() if p.requires_grad)
    print('Total number of network parameters:', num_total_params)
    print('Trainable number of network parameters:', num_trainable_params)
    param_reduc = 100 * (num_trainable_params_prev - num_trainable_params)/num_trainable_params_prev
    print('Reduction in the number of treinable parameters:', param_reduc)



    # network training parameter specifications
    batch_size = 32
    epochs = 1000

    target_acc = 0.92
    lr = 1e-3
    wd = 1e-4

    # loss function specification
    loss_fn = nn.CrossEntropyLoss()

    # optimizer specification (with default parameter setting, esp. learning rate of 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, amsgrad=True)

    # prepare data for training (partion data into minibatches)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)
    classes = train_data.classes
    print(classes)

    # training and testing 
    train_accuracy_log, val_accuracy_log = loop_over_epochs(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs, device, target_acc)

    # plot training and testing accuracy over the number of epochs
    plot_accuracy(train_accuracy_log, val_accuracy_log)  

    torch.save(model.state_dict(), "Homework2/problem3h_model")

else:
    model.load_state_dict(torch.load("Homework2/problem3h_model"))
    model.eval()


param_reduc = 100 * (num_trainable_params_prev - num_trainable_params)/num_trainable_params_prev
print('Reduction in the number of treinable parameters:', param_reduc)