import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from random import choice

# REMEMBER TO IMPORT PERCEPTRON!
from Perceptron import Perceptron

df = pd.read_csv("Colors.csv")

#print(df)

# ALL DATA MUST BE CONVERTED TO NUMBERS!!!

d = {"A": 0, "B": 1, "C": 2}
df["Grade"] = df["Grade"].map(d)
d = {"RED": 0, "BLUE": 1, "GREEN": 2}
df["Color"] = df["Color"].map(d)
d = {"YES": 1, "NO": 0}
df["Pass"] = df["Pass"].map(d)

#print(df)

def get_real_data(idx):
    x = [df["Grade"][idx], df["Color"][idx]]
    y = df["Pass"][idx]
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

lr = 0.01
input_dim = 2

batch_size = len(df["Grade"])
n_epochs = 3
n_batches = 1

seed = 1337

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

perceptron = Perceptron(input_dim=input_dim)
perceptron.load_state_dict(torch.load("perceptron_trained_model.pth"))
perceptron.train()

optimizer = optim.Adam(params=perceptron.parameters(), lr=lr)
bce_loss = nn.BCELoss()

losses = []

change = 1.0
last = 10.0
epsilon = 1e-3
epoch = 0
while change > epsilon or epoch < n_epochs or last > 0.3:
#for epoch in range(n_epochs):
    for idx in range(n_batches):

        optimizer.zero_grad()
        x_data, y_target = get_real_data(idx)
        y_pred = perceptron(x_data).squeeze()
        loss = bce_loss(y_pred, y_target)
        loss.backward()
        optimizer.step()
        
        
        loss_value = loss.item()
        losses.append(loss_value)

        change = abs(last - loss_value)
        last = loss_value
               
    epoch += 1

perceptron.eval()


def predict_MIT_or_Cornell(student, target, classifier, decision_threshold=0.4):

    result = classifier(student)
    print(f"Result: {result}")
    
    index = 1
    if result[0] < decision_threshold:
        index = 0

    if index == 1 and target == 1:
        return "It guessed MIT and it was right!"
    elif index == 0 and target == 0:
        return "It guessed Cornell and it was right!"
    elif index == 1 and target == 0:
        return "It guessed MIT and it was wrong."
    elif index == 0 and target == 1:
        return "It guessed Cornell and it was wrong!"

def get_toy_data():
    x = [choice([0, 1, 2]) for i in range(2)]
    print(x)
    if x == [0,0]:
        y = 1
    else:
        y = 0
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

data, target = get_toy_data()
guess = predict_MIT_or_Cornell(data, target, perceptron)
print(guess)