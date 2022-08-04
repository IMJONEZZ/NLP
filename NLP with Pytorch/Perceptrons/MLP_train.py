from MLP import MultilayerPerceptron
from random_forest import get_header_rm

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


block_list = ['Year','Date','Round','Region','Winning Seed','Winner','Winning Score','Losing Seed','Loser','Losing Score','Overtime','Rand','_name','_w','_l','_opp_points','_ft%','_orb','_position']

header = get_header_rm("./train_matrix/matrix_2002-2019.csv", block_list)
df = pd.read_csv("./train_matrix/matrix_2002-2019.csv")
X = df[header]
Y = df['Rand']


input_size = len(header)
output_size = 1
num_hidden_layers = 2
hidden_size = 220
seed = 8855


torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
continuing = False

if continuing == True:
    model = torch.load("./model/MLP/MLP_model.pt")
    model.train()
else:
    model = MultilayerPerceptron(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_hidden_layers=num_hidden_layers, 
                            output_size=output_size,
                            hidden_activation=nn.Sigmoid)
    model.train()
#print(model)

losses = []
batch_size = 32
n_batches = 10
max_epochs = 100

loss_change = 1.0
last_loss = 10.0
change_threshold = 1e-3
epoch = 0

lr = 0.01
optimizer = optim.Adam(params=model.parameters())
cross_ent_loss = nn.BCEWithLogitsLoss()

def early_termination(loss_change, change_threshold, epoch, max_epochs):
    terminate_for_loss_change = loss_change < change_threshold
    terminate_for_epochs = epoch > max_epochs
    
    #return terminate_for_loss_change
    return terminate_for_epochs

x_data, y_target = torch.tensor(X.values).float(), torch.tensor(Y.values).float()
train_dataset = torch.utils.data.TensorDataset(x_data, y_target)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

while not early_termination(loss_change, change_threshold, epoch, max_epochs):
    for x_data, y_target in train_loader:
 
        # step 1: zero the gradients
        model.zero_grad()
        
        # step 2: run the forward pass
        y_pred = model(x_data).squeeze()
        
        # step 3: compute the loss and accuracy
        loss = cross_ent_loss(y_pred, y_target)
        squished = torch.sigmoid(y_pred)
        accuracy = (squished > 0.5).float() == y_target
        accuracy = accuracy.sum() / accuracy.size()[0] * 100

        # step 4: compute the backward pass
        loss.backward()
        
        # step 5: have the optimizer take an optimization step
        optimizer.step()
        
        # auxillary: bookkeeping
        loss_value = loss.item()
        losses.append(loss_value)
        loss_change = abs(last_loss - loss_value)
        last_loss = loss_value
                
    epoch += 1
    print(f"Epoch: {epoch} | Loss: {last_loss:.5f} | Acc: {accuracy}")

torch.save(model, './model/MLP/MLP_model.pt')

testing = True
def test(test_file, model_file, header, prediction_file='predictions.txt'):

    # with open(pickle_name, 'rb') as f:
    #     classifier = pickle.load(f)

    with open(model_file, 'rb') as f:
        best_model = torch.load(f)
        best_model.eval()
        # df = pd.read_csv('test_csv_final_four.csv')
        df = pd.read_csv(test_file)
        X = df[header]
        Y = df['Rand']
        X, Y = torch.tensor(X.values).float(), torch.tensor(Y.values).float()
        test_dataset = torch.utils.data.TensorDataset(X, Y)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=67, shuffle=False)
        with open(prediction_file, 'w') as f:
            for x_data, y_target in test_loader:
                y_pred = best_model(x_data).squeeze()
                y_pred = torch.sigmoid(y_pred)
                squished = torch.sigmoid(y_pred)
                accuracy = (squished > 0.5).float() == y_target
                accuracy = accuracy.sum() / accuracy.size()[0] * 100
                for i in range(len(y_pred)):
                    if y_pred[i] > 0.5:
                        y_pred[i] = 1
                    else:
                        y_pred[i] = 0
                    f.write(f"Pred: {str(y_pred[i].item())} | Target: {y_target[i].item()}\n")
                f.write(f"Accuracy: {accuracy}\n")

if testing == True:
    test('./test_matrix/matrix_2021-2021.csv', './model/MLP/MLP_model.pt', header)