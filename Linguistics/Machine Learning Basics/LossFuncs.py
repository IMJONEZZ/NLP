import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# REMEMBER TO IMPORT PERCEPTRON!
from Perceptron import Perceptron

LEFT_CENTER = (3, 3)
RIGHT_CENTER = (3, -2)

class ShowcaseML():
    def __init__(self):
        pass

    def show_sigmoid(self):
        x = torch.arange(-5., 5., 0.1)
        y = torch.sigmoid(x)
        print("Sigmoids look like this mathematically: 1/1+e^-x")
        plt.plot(x.numpy(), y.detach().numpy())
        plt.show()

    def show_tanh(self):
        x = torch.arange(-5., 5., 0.1)
        y = torch.tanh(x)
        print("tanh is similar to the sigmoid, but cosmetically different.")
        print("It is expressed mathematically like this: (e^x - e^-x)/(e^x + e^-x")
        plt.plot(x.numpy(), y.detach().numpy())
        plt.show()

    def show_relu(self):
        relu = torch.nn.ReLU()
        x = torch.arange(-5., 5., 0.1)
        y = relu(x)
        print("Rectified Linear Unit (ReLU)")
        print("Looks like this: max(0,x)")
        plt.plot(x.numpy(), y.detach().numpy())
        plt.show()

    def show_prelu(self):
        prelu = nn.PReLU(num_parameters=1)
        x = torch.arange(-5., 5., 0.1)
        y = prelu(x)
        print("Parametric ReLU")
        print("max(x, ax)")
        plt.plot(x.numpy(), y.detach().numpy())
        plt.show()

    def show_softmax(self):
        softmax = nn.Softmax(dim=1)
        x_input = torch.randn(1, 3)
        y_output = softmax(x_input)
        print("Softmax is useful for giving us a list of probabilities that all add up to 1.")
        print(f"X input: {x_input}")
        print(f"Y output: {y_output}")
        print(f"Sum: {torch.sum(y_output, dim=1)}")

    def show_MSE(self):
        mse_loss = nn.MSELoss()
        print("Mean squared error allows us to assign a real value to the distance between target and prediction")
        print("(sum(y-ŷ)^2)/n")
        outputs = torch.randn(3, 5, requires_grad=True)
        targets = torch.randn(3, 5)
        loss = mse_loss(outputs, targets)
        loss.backward()
        print(loss)

    def show_BCE(self):
        bce_loss = nn.BCELoss()
        print("We use CE loss when we want to know HOW different two things are, not just whether they are different.")
        print("The correct class will be close to 1, and the incorrect class will be close to 0.")
        print("–sum(y) * log(ŷ)")
        sigmoid = nn.Sigmoid()
        probabilities = sigmoid(torch.randn(4, 1, requires_grad=True))
        print(probabilities)

        targets = torch.tensor([1, 0, 1, 0], dtype=torch.float32).view(4, 1)
        loss = bce_loss(probabilities, targets)
        loss.backward()
        print(loss)

def get_toy_data(batch_size, left_center=LEFT_CENTER, right_center=RIGHT_CENTER):
    x_data = []
    y_targets = np.zeros(batch_size)
    for batch_i in range(batch_size):
        if np.random.random() > 0.5:
            x_data.append(np.random.normal(loc=left_center))
        else:
            x_data.append(np.random.normal(loc=right_center))
            y_targets[batch_i] = 1
    return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_targets, dtype=torch.float32)

# SHOWCASE ACTIVATION
showstuff = ShowcaseML()

showstuff.show_sigmoid()
showstuff.show_tanh()
showstuff.show_relu()
showstuff.show_prelu()
showstuff.show_softmax()

# SHOWCASE LOSS
showstuff.show_MSE()
showstuff.show_BCE()

# Load the Trained Perceptron!
perceptron = Perceptron(input_dim=2)
perceptron.load_state_dict(torch.load("perceptron_trained_model.pth"))


perceptron.eval()
print(f"\n\n\nThis is our Perceptron: {perceptron}\n")
data, target = get_toy_data(batch_size=1)
print(f"Here's our data: {data}")
print(f"And here's the shape: {data.shape}")
print(f"And this is the number we're trying to get the Perceptron to guess: {target}\n")
y_pred = perceptron(data).squeeze()
print(f"This is the guess: {y_pred}")

def vectorize(animal):
        one_hot = np.zeros([1, 2], dtype=np.float32)
        
        idx = 0
        for token in animal:
            if token[0] < 0.5:
                one_hot[idx][0] = 1
            idx += 1

        return one_hot

def predict_cat_or_dog(animal, target, classifier, decision_threshold=0.5):
    vectorized_animal= torch.tensor(vectorize(animal))
    print(f"vectorized: {vectorized_animal.shape}")

    result = classifier(vectorized_animal.view(1, -1))
    print(f"result: {result}")
    
    probability_value = torch.sigmoid(result).item()
    index = 1
    if probability_value < decision_threshold:
        index = 0

    target_prob = torch.sigmoid(target).item()
    target_index = 1
    if target_prob < decision_threshold:
        target_index = 0

    if index == 1 and target_index == 1:
        return "It guessed Cat and it was right!"
    elif index == 0 and target_index == 0:
        return "It guessed Dog and it was right!"
    elif index == 1 and target_index == 0:
        return "It guessed Cat and it was wrong."
    elif index == 0 and target_index == 1:
        return "It guessed Dog and it was wrong!"

guess = predict_cat_or_dog(data, target, perceptron)
print(guess)