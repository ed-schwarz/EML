import numpy as np
import matplotlib.pyplot as plt
import math
from   matplotlib import cm
import random
import json 

'''
This File contains all of the answers of the Problem 1 of the homework 1, and the diferent questions are marked in with their corresponding letters.
'''

'''
a) Equations
z1 = w1 * x + b1 
y1 = 1/(1 + exp(-z1))
y_hat = w2 * y1 + b2

Update equations:
w2 = w2 - y1 * 2 * (y_hat - y)
b2 = b2 - 2 * (y_hat - y)
w1 = w1 - x * (1 - y1) * y1 * 2 * (y_hat - y)
b1 = b1 - (1 - y1) * y1 * 2 * (y_hat - y)
'''

'''
b)
'''
###############################################################################
# class / function definitions
###############################################################################

def y_from_x(x1, x2):
    return np.sin(np.power((np.power(x1, 2) + np.power(x2, 2)), (7/12)))

#------------------------------------------------------------------------------
#Class that generates input data, as well as true values
class GenerateData:
    def __init__(self, range_data, step):
        self.range_data = range_data
        self.step = step
        self.init_data()

    def init_data(self):

        self.size_data_set = (int((2*self.range_data) / self.step) ** 2)
        x1 = np.arange(-self.range_data, self.range_data, self.step)
        x2 = np.arange(-self.range_data, self.range_data, self.step)

        xe1, xe2 = np.meshgrid(x1, x2)

        y = y_from_x(xe1, xe2)

        xt1 = []
        xt2 = []

        elements = int(np.sqrt(self.size_data_set))

        for t in range((elements)):
            for u in range((elements)):
                xt1 = np.append(xt1, (t * self.step) - self.range_data)
                xt2 = np.append(xt2, (u * self.step) - self.range_data)

        xr1 = np.random.random(self.size_data_set)
        xr2 = np.random.random(self.size_data_set)
        xr1 = np.around((xr1/self.step) - self.range_data, decimals=1)
        xr2 = np.around((xr2/self.step) - self.range_data, decimals=1)
        

        x = np.vstack((xr1, xr2)).T
        self.x = x.T
        self.labels = 1   
        
        # determine the number of classes and the x-data range
        self.x_min = x.min()
        self.x_max = x.max()
        
        y = np.zeros((self.size_data_set))
        y = np.sin(np.power((np.power(xr1, 2) + np.power(xr2, 2)), (7/12)))
        self.y=y.T


#------------------------------------------------------------------------------
#Class containing the weights. Also used to "use" the network and calculate the loss
class NeuralNetwork:
    def __init__(self, N1):
        self.N1 = N1
        self.init_weights_biases()

    def init_weights_biases(self):
        self.w1 = (np.random.rand(self.N1,2) * 10) - 5
        self.b1 = (np.random.rand(self.N1,1) * 10) - 5
        self.w2 = np.random.rand(self.N1)
        self.b2 = (np.random.random(1) * 10) - 5

    def forward_pass(self, x, size_data_set):
        #hidden layer with sigm activation function
        z1 = np.matmul(self.w1,x) + self.b1
 
        y1 = 1/(1 + np.exp(-z1))

        #output layer 
        z2 = np.matmul(self.w2,y1) + self.b2
        y_hat = z2
        
        return y1, y_hat

    def calculate_loss_accuracy(self, y_hat, y, size_data_set):

        emp_loss = np.sum(np.power((y - y_hat), 2)) / size_data_set
        # classification accuracy
        acc = (y == y_hat).sum() / size_data_set
        
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
    def __init__(self, learning_rate, N1):
        self.learning_rate = learning_rate
        self.N1 = N1


    def update_weights_biases(self, nn, y1, y_hat, x, y, size_data_set):
        # updating weights according to the gradient decent method
        delta2 = 2 * (y_hat-y)
        delta2_rs = delta2.reshape(1,size_data_set)    

        grad_w2 = y1.reshape(1,self.N1,size_data_set) * delta2_rs
        grad_b2 = delta2

        grad_w2_mean = (grad_w2.sum(axis=2)).reshape(self.N1) / size_data_set
        grad_b2_mean = grad_b2.sum() / size_data_set

        delta1 = (nn.w2.reshape((self.N1,1)) * delta2_rs).sum(axis=0) * y1 * (1 - y1)

        grad_w1 = x.reshape(1, 2, size_data_set) * delta1.reshape(self.N1, 1, size_data_set)
        grad_b1 = delta1

        grad_w1_mean = grad_w1.sum(axis=2) / size_data_set
        grad_b1_mean = grad_b1.sum(axis=1).reshape(self.N1,1) / size_data_set



        nn.w2 -= (self.learning_rate * grad_w2_mean)
        nn.b2 -= (self.learning_rate * grad_b2_mean)
        nn.w1 -= (self.learning_rate * grad_w1_mean)
        nn.b1 -= (self.learning_rate * grad_b1_mean)


#------------------------------------------------------------------------------
def train(nn, backpropagation,data_train,data_val,epochs, print_loss_every, size_batch):
    print("---------------------------------")
    print("Start training netwrok")
    print("Number of hidden layers in hidden network: ",backpropagation.N1)
    print("Number of max epochs: ",epochs)
    print("Size of mini batch: ",size_batch)
    print("Learning rate: ",backpropagation.learning_rate)
    print("---------------------------------")
    # divide training data set into mini-batches for training
    num_mb = (data_train.size_data_set) // size_batch

    x_train_mb = np.array_split(data_train.x, num_mb, axis=1)
    y_train_mb = np.array_split(data_train.y, num_mb)

    acc_val = 0
    emp_loss_train = 1
    emp_loss_val = 1


    for t in range(epochs):  
        for mb in range(num_mb):
            # perform forward and backward pass on each minibatch
  
            y1, y_hat = nn.forward_pass(x_train_mb[mb], size_batch)

            backpropagation.update_weights_biases(nn, y1, y_hat, x_train_mb[mb], y_train_mb[mb], size_batch)

            # determine/print classification accuraccy/empirical loss every ... epochs
        if (t+1) % print_loss_every == 0:
            # calculate/print loss and accuracy
            # training data set #
            _, y_hat = nn.forward_pass(data_train.x, data_val.size_data_set)
            acc_train, emp_loss_train = nn.calculate_loss_accuracy(y_hat, data_train.y, data_train.size_data_set)            
            # validation data set #
            _, y_hat = nn.forward_pass(data_val.x, data_val.size_data_set)
            _, emp_loss_val = nn.calculate_loss_accuracy(y_hat, data_val.y, data_val.size_data_set)   
            print("---------------------------------")
            print("Epoch: ",t+1)
            print("Training dataset:")
            print(" Emp. loss: ", emp_loss_train)        
            print("Validation dataset:")
            print(" Emp. loss: ", emp_loss_val)  
        '''
        c)
        '''                  
        if emp_loss_train <= 0.002 and emp_loss_val <= 0.002:
            print("Achieved loss smaller than 0.002!")
            break 
    # final accuracy / loss after training
    # training data set
    _, y_hat = nn.forward_pass(data_train.x, data_val.size_data_set)
    _, emp_loss_train = nn.calculate_loss_accuracy(y_hat, data_train.y, data_train.size_data_set)            
    # validation data set
    _, y_hat = nn.forward_pass(data_val.x, data_val.size_data_set)
    _, emp_loss_val = nn.calculate_loss_accuracy(y_hat, data_val.y, data_val.size_data_set)   
    print("---------------------------------")
    print("Final Model Performance after ",t+1, "epochs.")
    print("Training dataset:")
    print(" Emp. loss: ", emp_loss_train)        
    print("Validation dataset:")
    print(" Emp. loss: ", emp_loss_val) 
    


#plot the original function
def plot_original_function():

    size_input = 100
    step = 0.1
    range_data = 6

    x1 = np.arange(-range_data, range_data, step)
    x2 = np.arange(-range_data, range_data, step)

    xe1, xe2 = np.meshgrid(x1, x2)

    xt1 = []
    xt2 = []

    elements = (int((2*range_data) // step))


    for t in range((elements + 1)):
        for u in range((elements + 1)):
            xt1 = np.append(xt1, (t * step) - range_data)
            xt2 = np.append(xt2, (u * step) - range_data)



    y = y_from_x(xt1, xt2)

    y_plot = np.array(y).reshape((len(x2), len(x1)))



    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(xe1, xe2, y_plot, cmap=cm.coolwarm)


    ax.set_xlabel('$x1$')
    ax.set_ylabel('$x2$') 
    ax.set_zlabel('$y$')

    # show plot
    plt.show()


#plot the output from the neural network
def plot_nn(model):
    x1_test_np = np.linspace(-6, 6, 600,dtype = np.float32) 
    x2_test_np = np.linspace(-6, 6, 600,dtype = np.float32)
    x_test_np = np.array(np.meshgrid(x1_test_np,x2_test_np,indexing="ij")).T.reshape(-1,2)

    #generate network output for test data grid
    #y = np.sin(np.power((np.power(x1_test_np, 2) + np.power(x2_test_np, 2)), (7/12)))
    _, y_hat_learned = model.forward_pass(x_test_np.T, 50)
    #_, emp_loss_train = model.calculate_loss_accuracy(y_hat_learned, y, 3600)  
    #print("Plot from data ")
    # illustrate the network output class data
    xe1, xe2 = np.meshgrid(x1_test_np, x2_test_np)
    y_plot = np.array(y_hat_learned).reshape((len(x2_test_np), len(x1_test_np)))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(xe1, xe2, y_plot, cmap=cm.coolwarm)
    # set axes labels
    ax.set_xlabel('$x1$')
    ax.set_ylabel('$x2$') 
    ax.set_zlabel('$\\hat{y}$')
    # show plot
    plt.show()

###############################################################################





#range of input data
range_data = 5

#steps between elements
step_data = 0.1

# number of hidden nodes
N1 = 75  # 

# max number of epochs for training
epochs = 100000

# the period of printing loss / classification accuracy (epochs)
print_loss_every = 1000

# learning rate for training
learning_rate = 1e-2

# size of the minibatches
# note that the total size of the training dataset should be divisible by the 
# minibatch size for this code to work correctly
# training dataset of size ((range_data * step_data) ** 2), we have: 
size_batch = 100

#training and test data
data_train = GenerateData(range_data, step_data)
data_val = GenerateData(range_data, step_data)

# -----------------------------------------------------------------------------
model = NeuralNetwork(N1)
backpropagation = Backpropagation(learning_rate, N1)

#print("Plot original function")
#plot_original_function()

#print("Plot from before training")
#plot_nn(model)

###############################################################################
# network training
###############################################################################
train(model, backpropagation, data_train, data_val, epochs, print_loss_every, size_batch)

'''
The weights are saved in a json file, that can be use to set the values of the net to the trained ones
'''
# Writing to problem1.json
with open("problem1.json", "w") as outfile:
    outfile.write(model.get_dict_json())


###############################################################################
# model output illustration after training
###############################################################################
'''
d)
'''
print("Plot from after training")
plot_nn(model)

