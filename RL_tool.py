import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

import random
import numpy as np

from ResNet_tool import ResNet

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=100):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)  
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))  
        x = torch.relu(self.ln2(self.fc2(x)))  
        x = self.softmax(self.fc3(x))
        return x

        
class Env:
    def __init__(self, model_path="./checkpoints/Resnet_CIFAR10_200epochs.pth"):
        self.model_path = model_path
        
        

    def reset(self):
        self.resnet = ResNet(model_path=self.model_path)
        self.module_names =self.resnet.get_module_names()

        # layer별 sparsity 정보 저장용
        self.sparsity = {name : 0 for name in self.module_names}
        self.global_sparsity = 0
        
        self.acc = {name : self.resnet.orig_test_acc for name in self.module_names}
        self.new_acc = self.resnet.orig_test_acc
        
        self.order_to_prune = [random.choice(self.module_names) for _ in range(len(self.module_names) * 5)] # layer수의 2배 만큼의 수를 random sampling
        
        module_name = self.order_to_prune[0]
        self.state = np.array(
            [
            self.sparsity[module_name], # layer_i의 sparsity ratio
            self.global_sparsity, # model 전체의 sparsity ratio
            0,
            self.resnet.orig_test_acc # test acc
        ]
        )
         
        return self.state


    def step(self, index, action):
        module_name = self.order_to_prune[index]

        self.pruning(action, module_name)

        self.get_sparsity(module_name)
        self.get_sparsity("global")

        new_acc = self.resnet.test()
        
        reward =  5*(self.new_acc/self.resnet.orig_test_acc + self.global_sparsity/0.8)
        
        self.new_acc = new_acc

        self.acc[module_name] = self.new_acc

        if index+1 == len(self.order_to_prune):
            return [self.global_sparsity,self.new_acc], reward
        
        next_module_name = self.order_to_prune[index+1]

        
        self.state = np.array(
            [
            self.sparsity[next_module_name], # layer_i의 sparsity ratio
            self.global_sparsity, # model 전체의 sparsity ratio
            self.acc[next_module_name],
            self.new_acc # test acc
        ]
        )
        
        


        return self.state, reward

    def _onehot(self):
        # One-hot 인코딩 수행
        num_classes = 54 
        onehot = F.one_hot(torch.tensor(self.state[0],dtype=int), num_classes=num_classes).numpy()
        self.state = np.concatenate((
            onehot,
            self.state[1:]
        ))
        
        
    def pruning(self, action, module_name):            
        
        module = self.resnet.net.get_submodule(module_name)

        prune.l1_unstructured(module, name='weight', amount=action)
        prune.remove(module,'weight')


    def get_sparsity(self, module_name):
        if module_name == "global":

            self.global_sparsity = 0
            for k in self.sparsity.keys():
                self.global_sparsity += self.sparsity[k]
            
            self.global_sparsity /= len(self.sparsity.keys())

        else:
            m = self.resnet.net.get_submodule(module_name)
            self.sparsity[module_name] = (torch.sum(m.weight == 0) / m.weight.nelement()).cpu().numpy()

