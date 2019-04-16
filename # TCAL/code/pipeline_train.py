# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 10:44:59 2018

@author: Franc
"""
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from steppy.base import BaseTransformer
from tqdm import tqdm

from pipeline_config import weights
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NetClassifier(BaseTransformer):
    
    def __init__(self):
        super(NetClassifier, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        #self.criterion = nn.MultiMarginLoss(weight=weights)
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def fit(self, loader, valid_loader):
        self.model = self.train_model(self.model, loader, valid_loader, num_epochs=10, Logits = None)
        return self.model
            
    def train_model(self, model, data_loader, valid_loader, num_epochs=10, Logits = None):
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = WeightedCrossEntropyLoss()
        if Logits:
            criterion = WeightedCrossEntropyLossWithLogits()
        for epoch in tqdm(range(num_epochs)):
            model.train()
            print(f'\nEpoch {epoch}/{num_epochs}')
            print('-'*10)
            running_loss = 0.0
            running_acc = 0.0
            for train_image, train_label in data_loader['train']:
                inputs, labels = train_image.to(self.device), train_label.to(self.device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    if Logits:
                        loss = criterion(outputs, labels)
                        _, labels = torch.max(labels, 1)
                    else:
                        _, labels = torch.max(labels, 1)
                        loss = criterion(outputs, labels)
                    acc = (predictions==labels).sum().item()
                    print('\t batch loss',round(float(loss),4), 
                          '\t batch accuracy', round(float(acc),4))
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()*inputs.size(0)
                    running_acc += acc
            with torch.no_grad():
                model.eval()                
                correct = 0
                for images, labels in valid_loader['valid']:
                    images, labels = images.to(self.device), labels.to(self.device)
                    _, labels = torch.max(labels, 1)
                    outputs = model(images)    
                    _, predictions = torch.max(outputs, 1)
                    correct += (predictions==labels).sum().item()
                
            print('epoch loss', round(float(running_loss),4), '\t', 
                  'epoch accuracy', running_acc,
                  'epoch valid accuracy', correct)
        return model
    
    def eval_model(self, valid_loader):
        with torch.no_grad():
            self.model.eval()
            test_loss = 0
            correct = 0
            for images, labels in tqdm(valid_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                _, labels = torch.max(labels, 1)
                outputs = self.model(images)    
                _, predictions = torch.max(outputs, 1)
                correct += (predictions==labels).sum().item()
            test_loss /= len(valid_loader.dataset)
            print('\nNums of Test Images:', len(valid_loader.dataset)*images.size(0),
                  '\nNums of Corrects:', correct,
                  '\nCrossEntropyLoss:', test_loss)
    
    def predict(self, test_loader):
        with torch.no_grad():
            self.model.eval()
            results = []
            for images in tqdm(test_loader):
                images = images.to(self.device)
                outputs = self.model(images)
                results.append(outputs)
        return results
    
    def model_save(self, model):
        torch.save(self.model, f'cache/model_{datetime.datetime.now().strftime("%Y%m%d")}')
    
    def load_model(self, path):
        return torch.load(path)
    
    
class WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    
    def __init__(self, weight = torch.FloatTensor([4.5,1.0,1.5,2.5])):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
    def forward(self, logits, target):
        criterion = nn.CrossEntropyLoss(weight = self.weight, size_average=False)
        return criterion(logits, target)
        
    
class WeightedMulLabelLoss(nn.MultiMarginLoss):
    
    def __init__(self, weight = torch.FloatTensor([4.5,1.0,1.5,2.5])):
        super(WeightedMulLabelLoss, self).__init__()
        self.weight = weight
    def forward(self, logits, target):
        criterion = nn.MultiMarginLoss(p=2, margin=0, 
                                       weight = self.weight, size_average=False)
        return criterion(logits, target)    
    
    
class WeightedCrossEntropyLossWithLogits(nn.Module):
    
    def __init__(self, weight = torch.FloatTensor([4.5,1.0,1.5,2.5])):
        super(WeightedCrossEntropyLossWithLogits, self).__init__()
        self.weight = weight
    
    def forward(self, logits, target):
        logits = F.softmax(logits, 1)
        #loss = F.binary_cross_entropy_with_logits(logit, target*weight)
        loss = self.weight*torch.pow(logits-target,2).sum()
        return loss

    
def train_model(model, criterion, optimizer, data_loader, num_epochs):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}/{num_epochs}')
        print('-'*10)
        running_loss = 0.0
        running_acc = 0.0
        for train_image, train_label in data_loader:
            inputs, labels = train_image.to(device), train_label.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, labels = torch.max(labels, 1)
                loss = criterion(outputs, labels)    
                _, predictions = torch.max(outputs, 1)
                acc = (predictions==labels).sum()/len(outputs)
                
                print('\t batch loss',round(float(loss),4), '\t batch accuracy', round(float(acc),4))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()*inputs.size(0)
                running_acc += acc
        print('epoch loss', round(float(running_loss),4), '\t', 
              'epoch accuracy', round(float(running_acc/len(data_loader)),4))
    return model

def eval_model(model, test_loader, criterion):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            print(labels, predictions)
            correct += (predictions==labels).sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nNums of Test Images:', len(test_loader.dataset)*images.size(0),
              '\nNums of Corrects:', correct,
              '\nCrossEntropyLoss:', test_loss) 
        


