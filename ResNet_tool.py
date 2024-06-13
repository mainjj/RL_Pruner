# 
import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from torch.utils.data import Subset
import torch.optim as optim

import torchvision.transforms as transforms

from tqdm import tqdm
import random

# test
def resnet_test(net, testloader, device):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

    
            outputs = net(images)
    
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    return correct / total


def test_for_each_classes(testloader, net, device):
    classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    print("="*70, end="\n\n")


class ResNet:
    def __init__(self, model_path, batch_size=128):
        self.batch_size = batch_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=False, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                shuffle=False, num_workers=2)

        
        self.load(model_path=model_path)
        self.orig_test_acc = self.test()

    def load(self, model_path):
        # model setting
        net = resnet50(weights=ResNet50_Weights.DEFAULT)
        net.fc = nn.Linear(net.fc.in_features, 10)

        if model_path:
            net.load_state_dict(torch.load(model_path))

        self.net = net.to(self.device)
    
    def get_module_names(self):
        # layer 이름 얻기
        module_names = []
        for name , module in self.net.named_modules():
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
                module_names.append(name)

        # print(f"\nFind {len(module_names)} layers\n")
        return module_names

    def train(self, epochs):
        
        # dataloader
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=self.transform)
        
        num_samples = 1000
        indices = random.sample(range(len(trainset)), num_samples)
        # random sampling (1000 images)
        subset_trainset = Subset(trainset, indices)


        trainloader = torch.utils.data.DataLoader(subset_trainset, batch_size=self.batch_size,
                                                shuffle=True, num_workers=2)

        # crit, optim
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        # train
        # print(f"\nStart training Resnet , epochs : {epochs}\n")
        for epoch in range(epochs):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()


                running_loss += loss.item() 
            # print(f'\tTrain Loss : {round(loss.item(),4)} | Test Acc : {round(100 * test_acc, 2)} %\n')
            running_loss = 0.0
        # print('\nFinished Training')


    def test(self):
        test_acc = resnet_test(self.net, self.testloader, self.device)
        return test_acc
    
    def save(self, name):
        PATH = f'./checkpoints/{name}.pth'
        torch.save(self.net.state_dict(), PATH)
        print(f"Checkpoints saved at {PATH}")



    def retrain(self, epochs):
        
        # dataloader
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=self.transform)
        

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                shuffle=True, num_workers=2)

        # crit, optim
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        # train
        print(f"\nStart re-training ResNet , epochs : {epochs}\n")
        for epoch in range(epochs):

            running_loss = 0.0
            for i, data in tqdm(enumerate(trainloader, 0), desc=f"{epoch}/{epochs} "):
                
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()


                running_loss += loss.item()

            test_acc = resnet_test(self.net, self.testloader, self.device)
            print(f'\tTrain Loss : {round(loss.item(),4)} | Test Acc : {round(100 * test_acc, 2)} %\n')
            running_loss = 0.0
        print('\nFinished Training')
