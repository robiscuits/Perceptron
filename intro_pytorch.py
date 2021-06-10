import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set = datasets.MNIST('./data', train=True, download=True,
                       transform=custom_transform)
    test_set = datasets.MNIST('./data', train=False,
                       transform=custom_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 50)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 50, shuffle = False)
    if training:
        return train_loader
    else:
        return test_loader


def build_model():
    model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64,10)
    )
    return model


def train_model(model, train_loader, criterion, T):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(0,T):
        r_l = 0.0
        set_size = 0.0
        total = 0
        correct = 0
        for image, labels in train_loader:
            opt.zero_grad()
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = correct/total
            pct = 100* acc
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            r_l += loss.item()
            set_size += 1
        print("Train Epoch: {} Accuracy: {}/{}({}%) Loss: {}".format(epoch, correct, total, round(pct,2), round(r_l/set_size,3)))
            
    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    model.eval()
    with torch.no_grad():
        r_l = 0.0
        set_size = 0.0
        total = 0
        correct = 0
        for image, labels in test_loader:
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = correct/total
            pct = 100* acc
            loss = criterion(outputs, labels)
            r_l += loss.item()
            set_size += 1
        if show_loss:
            print("Average loss: {}\nAccuracy: {}%".format(round(r_l/set_size,4), round(pct,2)))
        else:
            print("Accuracy: {}%".format(round(pct,2)))
    


def predict_label(model, test_images, index):
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    model.eval()
    l = []
    with torch.no_grad():
        for image in test_images:
            outputs = model(image)
            prob = F.softmax(outputs, dim=1)
            top3 = torch.topk(prob, 3, 1)
            l.append(top3)
    idx_vals = l[index]
    for i in range(len(idx_vals.values[0])):
        name = class_names[int(idx_vals.indices[0][i])]
        print("{}: {}%".format(name, round(float(idx_vals.values[0][i])*100, 2)))
        

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
