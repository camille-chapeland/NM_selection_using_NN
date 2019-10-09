#########################################################################################################
######################################### LOAD PYTHON PACKAGES ##########################################
#########################################################################################################

# Pytorch can be installed using the following link:
# https://pytorch.org/ 
# Make sure to select python 2.7 if you don't have python 3 but note that this code is
# in python3 and might need to be adjusted accordingly.

# A very good resource to understand how to use of Pytorch in NN:
# https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e

#If everything works as intended, the program should run smoothly and output prediction probabilities are 
# 0.8 - 1

#########################################################################################################
#########################################################################################################
#########################################################################################################

import torch
import numpy as np
import scipy
import json
import os

#########################################################################################################
################################### DATA UPLOAD AND PRE-PROCESSING ######################################
#########################################################################################################

# The files must be in the format and file arrangement zipped with this Notebook, with the 'archive' 
# folder standing in the same directory ('valid' and 'invalid' nested in an 'archive' folder).  
#       > This is the format of the zip file sent

# The Dataset package from torch.utils.data helps create a dataset compatable with Pytorch such that the 
# data is arrange in torch tensors. The tensor contains the 1D image vector and a label where 1 
# represents a valid normal mode pick and 0 an invalid pick. Before being stored in a tensor, each data 
# vector (or image) is transformed to the absolute value of the imaginary number it represents and 
# resampled to have homogeneous image length in the tensor. 

# The __init__ function loads the data, the __len__ function returns the size of the tensor and the 
# __get_item__ function returns an indexed image from the tensor. Notice that normalisation of the image 
# is applied in the __get_item__ function, this is to help prevent vanishing gradients in the gradient 
# descent loss process applied later. 

#########################################################################################################
#########################################################################################################
#########################################################################################################

from torch.utils.data import Dataset
from torchvision import transforms
from scipy.signal import resample

print('------------------------------------------------------------')
print('----------------------start file upload---------------------')
print('------------------------------------------------------------')
class ImageDataset(Dataset):
    
    def __init__(self, data_folder):
    
        self.data_folder = data_folder
        self.images = []
        self.label = []
                
        counter=0
        
        for root, dirs, files in os.walk(data_folder):
            for filename in files:
                
                image = []
                
                # This line is for Mac.OS operating systems which 
                # create a .DS_store file when decoding 
                if filename.endswith('Store'):
                    continue
                    
                elif root.endswith('invalid'):
                    with open(data_folder + '/invalid/' + filename) as f:
                        
                        data = json.load(f)
                        label = 'invalid'
                        
                        for l in range(0,len(data)):
                            x = [complex(y[0], y[1])for y in data[l]] 
                            d = [abs(y) for y in x]*10
                            d = resample(d,20)
                            
                            self.images.append(d)
                            self.label.append(0)
                               
                        counter+=1
                        print(counter,',','filename = ', filename, ',' , 'label = invalid')
                        
                else:
                     with open(data_folder + '/valid/' + filename) as f:
                        
                        data = json.load(f)
                        label = 'valid'
                        
                        for l in range(0,len(data)):
                            x = [complex(y[0], y[1])for y in data[l]] 
                            d = [abs(y) for y in x] 
                            d = resample(d,20)
                            
                            self.images.append(d)
                            self.label.append(1)
                            
                        counter += 1
                        print(counter,',','filename = ', filename, ',' , 'label = valid')
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        data = torch.FloatTensor(self.images[idx])
        data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
        label = torch.FloatTensor(self.label[idx])
        
        return (data, self.label[idx])

if __name__ == '__main__':
    dataset = ImageDataset(os.getcwd() + '/archiveZ')

print('------------------------------------------------------------')
print('-------------------file upload complete---------------------')
print('------------------------------------------------------------')

#########################################################################################################
###################################### SOME INFO ABOUT THE DATA #########################################
#########################################################################################################

# It is important to have some knowledge about the dimension of the tensor but also to ensure that the 
# training and testing data have balanced amount of valid and invalid picks. This is done by checking the 
# ratio of valid to invalid images (aim for 0.8 - 1.2). It is also possible to train with an imbalanced 
# dataset but this might however lead to underfitting of some data.  

# The size of the resampled image vector in the tensor (printed below) will be used in the first layer 
# of the neural network.

#########################################################################################################
#########################################################################################################
#########################################################################################################  

print('------------------------------------------------------------')
print('--------------------info about the data---------------------')
print('------------------------------------------------------------')
valid=0
invalid=0
for i in range(0,len(dataset)):
    img , label = ImageDataset.__getitem__(dataset,i)
    if label == 1 :
        valid+=1
    else:
        invalid+=1
print('Full dataset ratio of valid/invalid data is :', (valid/invalid))
print('Number of invalid data is :', invalid)
print('Number of valid data is :', valid)

img , label = ImageDataset.__getitem__(dataset,10)
print('...')
print('Example of a image tensor:')
print('1D vector' , img , 'Respective label:' , label)
print('Confirming the type of the output output:' , type(img))
image_vector = img.shape 
print('Length of a image vector:' , image_vector)

#########################################################################################################
################################# SPLITTING DATA (TRAIN AND TEST) #######################################
#########################################################################################################

# To ensure the training is successful, the original data tensor is randomly separated into a training 
# set and a testing set. The testing set is not shown to the netwrok until the training is completed. The 
# amount of testing data chosen is 20% (validation_split) to ensure that the testing is as broad as 
# possible without too much compromise on the size of the training dataset. 

# The PyTorch SubsetRandomSampler helps with specialised functions to this end. Not only does it help 
# split the full dataset, it also allows subdivision of the training set into mini-batches of specific 
# length (batch_size = 16). The batches are used to compute the loss using all points in the traning set 
# (batch gradient descent). If we were to use a single point at each time, it would be a stochastic 
# gradient descent. Because the data set is very large, the batch is divided into the mini-batches 
# described above. Selecting shuffle_dataset = True ensures that the batches are made up of valid and 
# invalid picks.

# The random seed selected (42 being the 'least' random) ensures reproducibility of the results.

# The dataloader (train and test loaders) will behave like an iterator, so we can loop over it and fetch 
# a different mini-batch during training.

#########################################################################################################
#########################################################################################################
#########################################################################################################

print('------------------------------------------------------------')
print('----------------------start splitting-----------------------')
print('------------------------------------------------------------')

from torch.utils.data.sampler import SubsetRandomSampler

batch_size = 16

#20% of the dataset will go to test dataset
validation_split = .2
shuffle_dataset = True
random_seed= 42

print('The split rate is:',validation_split)

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating data samplers and loaders:
train_data = SubsetRandomSampler(train_indices)
test_data = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_data)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_data)

print('The full dataset has: %2d elements' %(len(dataset))) 
print('Of these, %2d will be used for training the network and %2d will be used to validate the training' %(len(train_data), len(test_data))) 
print('The number of the batches are %2d and %2d respectively of size %2d' %(len(train_loader), len(test_loader), batch_size))

print('------------------------------------------------------------')
print('---------------------splitting complete---------------------')
print('------------------------------------------------------------')

#########################################################################################################
########################################### NETWORK TRAINING ############################################
#########################################################################################################

# The data is trained using the gradient descent method. This method works as follow:
#       > (Create a random model with a given number of nodes, weights and layers)
#       > Use the model to predict the class of an image (forward-pass)
#       > Compute the error between the prediction and the labels using an appropriate loss-function
#       > Compute the gradient of the error for every parameter
#       > Adjust the parameters using the gradients (backward-pass)
#       > Use the updated parameters to perform a new forward-pass and repeat

# This workflow represents  the work done in a full epoch (taking into account that the first step only
# applies to the first layer).

# We use the nn.Sequential function to create a forward-pass model with an arbitrary amount of hidden 
# layers (1 hidden layer in this example). This function is the most spatially efficient way to create 
# the model where the layers are specified sequentially together with the wanted activation function. In 
# each layer, a linear system is solved given by nn.linear(weights, nodes):
#                                 node_1 x weight_1 + node_2 x weight_2 = bias  
# where the weights and biases are first chosen randomly and then adjusted. In the linear layer the 
# number of nodes (and thus weights) are given by the the first argument and the biases by the second 
# argument. Note that for the initial layer, the number of nodes must be equal to the length of an image 
# vector (which is 20 as defined by the resampling). The input nodes (weights) of the second layer must 
# be equal to the number of output nodes (biases) of the first layer. In the final layer, the output nodes 
# must be equal to the number of classes wanted. Finally, the more layers and nodes are used, the more
# accurate each epoch will be but the more time and memory expensive the program becomes. When adjusting these 
# parameters, it is important to take into account the effects of over- and underfitting. I noticed that the 
# network wasn't training properly for valid picks until I lowered the epochs from 5 to 2 and the learning rate 
# from 0.0001 to 0.01 (relatively high compared to some example for 2D or 3D data training); I suspect that this 
# was caused by overfitting of the training dataset.  
 
# After each linear system forward-pass, the activation function is used to represent linear boundary 
# solution of the system as a probability space. Here we use the choose the ReLU (Rectified Linear Unit) 
# activation function defined as y = max(0, x) because it is cheap, converges faster and prevents the issue 
# of vanishing gradients.        

# At the end of a full forward pass, the softmax function  assigns a score for each class (z1,..,z_n) then 
# probability that an object belongs to a certain class is givem using : 
#                                    P(class i) = e^z_i/(e^z_n)!
# where the use of exponentials helps deal with negative values in the probability space.  Here we use a 
# LogSoftmax function (which is only compatible with the nn.NLLLoss() loss function). The criterion parameter
# describes is the loss function used to compute the error, and the optimizer is used to updated the many
# parameters automatically using the computed gradients.  

#########################################################################################################
#########################################################################################################
#########################################################################################################
print('------------------------------------------------------------')
print('-----------------------start training-----------------------')
print('------------------------------------------------------------')

from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim

# The feed-forward model
model = nn.Sequential(nn.Linear(20, 20),        # input layer
                      nn.ReLU(),        
                      nn.Linear(20, 10),        # hidden layer 
                      nn.ReLU(),
                      nn.Linear(10, 2),         # output layer
                      nn.LogSoftmax(dim=0))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 2
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print('Epoch:' , e+1 , '/' , epochs)
        print(f'Training loss: {running_loss/len(train_loader)}')
        

print('------------------------------------------------------------')
print('---------------------training complete----------------------')
print('------------------------------------------------------------')

#########################################################################################################
#################################### COMPUTE CLASS-PROBABILITIES ########################################
#########################################################################################################

# Here are the results of using the trained network on unseen testing data. The output is given as a 
# probability from 0 to 1

#########################################################################################################
#########################################################################################################
#########################################################################################################
invalid_prob = []
valid_prob = []

for i in range(len(test_loader)):
    
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    
    for unit in range(0,batch_size-1):
        ps = torch.exp(model(images[unit]))
        if labels[unit] == 1:
            valid_prob.append(ps[1])
        else:
            invalid_prob.append(ps[0])

print('Probability that an invalid pick is predicted correctly:' , sum(invalid_prob)/len(invalid_prob))
print('Probability that a valid pick is predicted correctly:' , sum(valid_prob)/len(valid_prob))


