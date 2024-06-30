import numpy as np
import matplotlib.pyplot as plt
import math
from   matplotlib import cm
import random
import json 

'''
This File contains all of the answers of the Problem 2 of the homework 1, and the diferent questions are marked in with their corresponding letters.
'''

'''
b), c), d), e)

Equations:
z1 = w1 * x + b1
y1 = tanh(z1)
z2 = w2 * y1 + b2
y_hat = exp(z2) / sum(exp(z2))

Update Equations:
w2 = w2 - y1 * (y_hat - y)
b2 = b2 - (y_hat - y)
w1 = w1 - w2 * (1 - y1 **2) * (y_hat - y)
b1 = b1 - (1 - y1 **2) * (y_hat - y)
'''

###############################################################################
# class / function definitions
###############################################################################

def y_from_x(x1, x2):
    return np.sin(np.power((np.power(x1, 2) + np.power(x2, 2)), (7/12)))

#------------------------------------------------------------------------------
#Class that generates input data, as well as true values
class GenerateData:
    def __init__(self, range_data, size):
        self.range_data = range_data
        self.size_data_set = size
        sqrt_8 = np.sqrt(8)
        self.N_cl = 12
        x_c = np.asarray([[1, 1], [1, -1], [-1, 1], [-1, -1], [2, 2], [2, -2], [-2, 2], [-2, -2], [0, sqrt_8], [0, -sqrt_8], [-sqrt_8, 0], [sqrt_8, 0]])
        self.x_c = x_c.T
        self.init_data()
        

    def init_data(self):

        self.step = (2*self.range_data) / (np.sqrt(self.size_data_set))

        xr1 = np.random.default_rng().uniform(-self.range_data, self.range_data,self.size_data_set)
        xr2 = np.random.default_rng().uniform(-self.range_data, self.range_data,self.size_data_set)
        x = np.vstack((xr1, xr2)).T

        self.x = x.T
        self.labels = np.zeros(self.size_data_set)
        y = np.zeros((self.size_data_set,self.N_cl))       

        for t in range(self.size_data_set):
            dist_temp = np.zeros(self.N_cl)
            for u in range(self.N_cl):
                dist_temp[u] = np.linalg.norm(self.x[:, t] - self.x_c[:, u])

            y[t, np.argmin(dist_temp)] = 1
            self.labels[t] = np.argmin(dist_temp)

        
        self.y=y.T



        


#------------------------------------------------------------------------------
#Class containing the weights. Also used to "use" the network and calculate the loss
class NeuralNetwork:
    def __init__(self, N1, N_cl):
        self.N1 = N1
        self.N_cl = N_cl
        self.init_weights_biases()

    def init_weights_biases(self):
        # initialize weights / biases randomly
        # hidden layer weights
        self.w1 = np.random.randn(self.N1,2)
        self.b1 = np.random.randn(self.N1,1)
        self.w2 = np.random.randn(self.N_cl,self.N1)
        self.b2 = np.random.randn(self.N_cl,1)

    def forward_pass(self, x):
        #hidden layer with tanh activation function

        z1 = np.matmul(self.w1,x) + self.b1
        y1 = np.tanh(z1)
        #output layer with softmax activation function
        z2 = np.matmul(self.w2,y1) + self.b2
        y_hat = np.exp(z2) / np.sum(np.exp(z2), axis=0)
        
        return y1, y_hat

    def calculate_loss_accuracy(self, y_hat, y, labels, size_data_set):
        # determine labels from network output
        labels_hat = np.argmax(y_hat, axis = 0)
        # mean cross entropy loss for data set
        emp_loss = (-y * np.log(y_hat)).sum() / size_data_set
        # classification accuracy
        acc = (labels_hat == labels).sum() / size_data_set
        
        return acc, emp_loss

    def get_dict_json(self):
        dict_obj = {
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2" : self.w2.tolist(),
            "b2" : self.b1.tolist()
            }

        return json.dumps(dict_obj)
    
#------------------------------------------------------------------------------
#Class used to update the weights usiing gradient descent
class Backpropagation:
    def __init__(self, learning_rate, N1, N_cl):
        self.learning_rate = learning_rate
        self.N1 = N1
        self.N_cl = N_cl

    def update_weights_biases(self, nn, y1, y_hat, x, y, size_data_set):
        # updating weights according to the gradient decent method
        delta2 = y_hat - y
        delta2_rs = delta2.reshape(self.N_cl,1,size_data_set)    

        grad_w2 = y1.reshape(1,self.N1,size_data_set) * delta2_rs
        grad_b2 = delta2

        grad_w2_mean = grad_w2.sum(axis=2) / size_data_set
        grad_b2_mean = grad_b2.sum(axis=1).reshape(self.N_cl,1) / size_data_set


        delta1 = (nn.w2.reshape((self.N_cl,self.N1,1)) * delta2_rs).sum(axis=0) * (1 - y1 **2)

        grad_w1 = x.reshape(1, 2, size_data_set) * delta1.reshape(self.N1, 1, size_data_set)
        grad_b1 = delta1

        grad_w1_mean = grad_w1.sum(axis=2) / size_data_set
        grad_b1_mean = grad_b1.sum(axis=1).reshape(self.N1,1) / size_data_set

        nn.w2 -= self.learning_rate * grad_w2_mean
        nn.b2 -= self.learning_rate * grad_b2_mean    
        nn.w1 -= self.learning_rate * grad_w1_mean
        nn.b1 -= self.learning_rate * grad_b1_mean


#------------------------------------------------------------------------------
def train(nn, backpropagation,data_train,data_val,epochs, print_loss_every, size_batch):
                       
    # divide training data set into mini-batches for training
    num_mb = data_train.size_data_set // size_batch
    x_train_mb = np.array_split(data_train.x, num_mb, axis=1)
    y_train_mb = np.array_split(data_train.y, num_mb, axis=1)


    acc_val = 0
    for t in range(epochs):
        for mb in range(num_mb):
            # perform forward and backward pass on each minibatch
            y1, y_hat = nn.forward_pass(x_train_mb[mb])

            backpropagation.update_weights_biases(nn, y1, y_hat, x_train_mb[mb], y_train_mb[mb], size_batch)

            # determine/print classification accuraccy/empirical loss every ... epochs
        if (t+1) % print_loss_every == 0:
            # calculate/print loss and accuracy
            # training data set #
            _, y_hat = nn.forward_pass(data_train.x)
            acc_train, emp_loss_train = nn.calculate_loss_accuracy(y_hat, data_train.y, data_train.labels, data_train.size_data_set)            
            # validation data set #
            _, y_hat = nn.forward_pass(data_val.x)
            acc_val, emp_loss_val = nn.calculate_loss_accuracy(y_hat, data_val.y, data_val.labels, data_val.size_data_set)  
            
            print("---------------------------------")
            print("Epoch: ",t+1)
            print("Training dataset:")
            print("Model accuracy: ", acc_train*100,"%", " Emp. loss: ", emp_loss_train)        
            print("Validation dataset:")
            print("Model accuracy: ", acc_val*100,"%", " Emp. loss: ", emp_loss_val)                    

        if acc_val >= 0.99:
           break 
    # final accuracy / loss after training
    # training data set
    _, y_hat = nn.forward_pass(data_train.x)
    acc_train, emp_loss_train = nn.calculate_loss_accuracy(y_hat, data_train.y, data_train.labels, data_train.size_data_set)            
    # validation data set
    _, y_hat = nn.forward_pass(data_val.x)
    acc_val, emp_loss_val = nn.calculate_loss_accuracy(y_hat, data_val.y, data_val.labels, data_val.size_data_set)   
    print("---------------------------------")
    print("Final Model Performance after ",t+1, "epochs.")
    print("Training dataset:")
    print("Model accuracy: ", acc_train*100,"%", " Emp. loss: ", emp_loss_train)        
    print("Validation dataset:")
    print("Model accuracy: ", acc_val*100,"%", " Emp. loss: ", emp_loss_val)
    


#plot the original function
def plot_original_function(data_train):

    data_colors = ["tab:green","tab:orange","tab:blue","tab:red","tab:purple","tab:cyan","tab:gray"]
    plt.figure(1)


    for cl in range(data_train.N_cl):
        xcl_train_np = data_train.x.T[data_train.labels == cl]
        plt.scatter(xcl_train_np[:,0],xcl_train_np[:,1])

    x_c_train_np = data_train.x_c.T
    plt.scatter(x_c_train_np[:,0],x_c_train_np[:,1], c="black")
    plt.show()




#plot the output from the neural network
def plot_nn(model):
    x1_test_np = np.linspace(-5, 5, 600,dtype = np.float32) 
    x2_test_np = np.linspace(-5, 5, 600,dtype = np.float32)
    x_test_np = np.array(np.meshgrid(x1_test_np,x2_test_np,indexing="ij")).T.reshape(-1,2)


    _, y_hat_learned = model.forward_pass(x_test_np.T)

    data_colors = ["tab:green","tab:orange","tab:blue","tab:red","tab:purple","tab:cyan","tab:gray"]
    plt.figure(1)
    print("Number of training data points:")
    labels_hat_learned = np.argmax(y_hat_learned, axis = 0)

    for cl in range(data_train.N_cl):

        xcl_train_np = x_test_np[labels_hat_learned == cl]
        plt.scatter(xcl_train_np[:,0],xcl_train_np[:,1])

    x_c_train_np = data_train.x_c.T
    plt.scatter(x_c_train_np[:,0],x_c_train_np[:,1], c="black")
    plt.show()

###############################################################################





#range of input data
range_data = 4

#steps between elements
size = 100000

# number of hidden nodes
N1 = 8  # 

# max number of epochs for training
epochs = 100000

# the period of printing loss / classification accuracy (epochs)
print_loss_every = 500

# learning rate for training
learning_rate = 5*1e-3

# size of the minibatches
# note that the total size of the training dataset should be divisible by the 
# minibatch size for this code to work correctly
# training dataset of size ((range_data * step_data) ** 2), we have: 
size_batch = 50

#training and test data
data_train = GenerateData(range_data, size)
data_val = GenerateData(range_data, size)

# -----------------------------------------------------------------------------
model = NeuralNetwork(N1, data_train.N_cl)
backpropagation = Backpropagation(learning_rate, N1, data_train.N_cl)

'''
a)
'''
print("Plot original function")
plot_original_function(data_train)

#print("Plot from before training")
#plot_nn(model)

###############################################################################
# network training
###############################################################################
train(model, backpropagation, data_train, data_val, epochs, print_loss_every, size_batch)

'''
The weights are saved in a json file, that can be use to set the values of the net to the trained ones
'''
# Writing to problem2.json
with open("problem2.json", "w") as outfile:
    outfile.write(model.get_dict_json())

###############################################################################
# model output illustration after training
###############################################################################
'''
f)
'''
print("Plot from after training")
plot_nn(model)

