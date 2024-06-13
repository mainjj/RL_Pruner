import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

from ResNet_tool import resnet_test, test_for_each_classes

from tqdm import tqdm


def train(epochs, batch_size, model_path, name, lr, is_pruned):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    
    # transforms
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # dataloader
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    

    # model setting
    net = resnet50(weights=ResNet50_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, 10)

    if model_path:
        net.load_state_dict(torch.load(model_path))

    net.to(device)

    # crit, optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)


    if is_pruned:
        masks = []
        for _p in net.parameters():
            masks.append((_p == 0))

    # train
    print(f"\nStart training Resnet , epochs : {epochs}\n")
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0), desc=f"[{epoch+1}/{epochs}]\t"):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if is_pruned:
                for i, p in enumerate(net.parameters()):
                    p.grad.data[masks[i]] = 0

            optimizer.step()

            # print statistics
            running_loss += loss.item()
        
        
        test_acc = resnet_test(net, testloader, device)
        print(f'\tTrain Loss : {round(loss.item(),4)} | Test Acc : {round(100 * test_acc, 2)} %')

        running_loss = 0.0
        
    test_for_each_classes(testloader, net, device)
    
    if name:
        PATH = f'./checkpoints/{name}.pth'
        torch.save(net.state_dict(), PATH)
        print(f"Checkpoints saved at {PATH}")

    print('\nFinished Training')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Resnet Training')
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')  
    parser.add_argument('--batch', default=32, type=int, help='batch size')  
    parser.add_argument('--lr', default=0.001, type=float, help='SGD learning rate')  
    parser.add_argument('--model_path', default='', help='model path for continue training')
    parser.add_argument('--name', help='save model name, save like $name$.pth')
    parser.add_argument('--is_pruned', action="store_true",help='checkpointed is already pruned?')


    args = parser.parse_args()

    train(
        epochs = args.epochs,
        batch_size= args.batch, 
        model_path = args.model_path,
        name = args.name,
        lr = args.lr,
        is_pruned = args.is_pruned
    )