import sys
import numpy as np
import random
import pandas as pd
import math, time
import itertools
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import datetime
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import gurobipy as gp
import logging
import copy
from collections import defaultdict
import joblib
import gurobipy as gp
from gurobipy import GRB

capacity = 200
penaltyTerm = 0

itemNum = 10
featureNum = 4096
trainSize = 700
targetNum = 2
meanPriceValue = 0
meanWeightValue = 0

def actual_obj(valueTemp, cap, weightTemp, n_instance):
    obj_list = []
    for num in range(n_instance):
        weight = np.zeros(itemNum)
        value = np.zeros(itemNum)
        cnt = num * itemNum
        for i in range(itemNum):
            weight[i] = weightTemp[cnt]
            value[i] = valueTemp[cnt]
            cnt = cnt + 1
        weight = weight.tolist()
        value = value.tolist()
        
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(itemNum, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
        m.setObjective(x.prod(value), GRB.MAXIMIZE)
        m.addConstr((x.prod(weight)) <= cap)
#        for i in range(itemNum):
#            m.addConstr((x.prod(weight[i])) <= cap)

        m.optimize()
#        sol = []
#        for i in range(itemNum):
#            sol.append(x[i].x)
#        print(sol)
        objective = m.objVal
        obj_list.append(objective)
#        print(objective)
        
    return np.array(obj_list)

def correction_single_obj(realPrice, predPrice, cap, realWeightTemp, predWeightTemp, penalty):
    realWeight = np.zeros(itemNum)
    predWeight = np.zeros(itemNum)
    realPriceNumpy = np.zeros(itemNum)
    for i in range(itemNum):
        realWeight[i] = realWeightTemp[i]
        predWeight[i] = predWeightTemp[i]
        realPriceNumpy[i] = realPrice[i]
        
    if min(predWeight) >= 0:
        predWeight = predWeight.tolist()
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(itemNum, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
        m.setObjective(x.prod(predPrice), GRB.MAXIMIZE)
        m.addConstr((x.prod(predWeight)) <= cap)

        m.optimize()
        sol = []
        for i in range(itemNum):
            sol.append(x[i].x)
#        print("realPrice: ", realPrice)
#        print("predPrice: ", predPrice)
#        print(sol)
#        objective = m.objVal
        objective = 0
        for i in range(itemNum):
            objective = objective + sol[i] * realPrice[i]
#        print(m.objVal, objective)
#        print(objective)

        #correction
        tau = 1
        selectedTotalWeight = np.dot(realWeight, sol)
        if selectedTotalWeight > cap:
            tau = cap / selectedTotalWeight
        objective = tau * objective - np.dot(np.multiply(realPriceNumpy, penalty), np.multiply(sol, 1-tau))
        sol = np.multiply(sol, tau)

    else:
        objective = 0
        
#    print(sol)
#    print(objective)
#        print(np.dot(G, sol), objective)
#        print("")
    return objective

    
# simply define a silu function
def silu(input):
    for i in range(itemNum):
        if input[i][0] < 0:
            input[i][0] = 0
        input[i][0] = input[i][0] + ReLUValue
        if input[i][1] < 0:
            input[i][1] = 0
        input[i][1] = input[i][1] + ReLUValue
    return input

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class SiLU(nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        return silu(input) # simply apply already implemented SiLU

# initialize activation function
activation_function = SiLU()
    
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def make_fc(num_layers, num_features, num_targets=targetNum,
            activation_fn = nn.ReLU,intermediate_size=512, regularizers = True):
    net_layers = [nn.Linear(num_features, intermediate_size),
        activation_function]
    for hidden in range(num_layers-2):
        net_layers.append(nn.Linear(intermediate_size, intermediate_size))
        net_layers.append(activation_function)
    net_layers.append(nn.Linear(intermediate_size, num_targets))
    net_layers.append(activation_function)
    return nn.Sequential(*net_layers)
        

class MyCustomDataset():
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value

    def __len__(self):
        return len(self.value)

    def __getitem__(self, idx):
        return self.feature[idx], self.value[idx]


import sys
import ip_model_whole as ip_model_wholeFile
from ip_model_whole import IPOfunc

class Intopt:
    def __init__(self, c, h, A, b, penalty, n_features, num_layers=5, smoothing=False, thr=0.1, max_iter=None, method=1, mu0=None,
                 damping=0.5, target_size=targetNum, epochs=8, optimizer=optim.Adam,
                 batch_size=itemNum, **hyperparams):
        self.c = c
        self.h = h
        self.A = A
        self.b = b
        self.penalty = penalty
        self.target_size = target_size
        self.n_features = n_features
        self.damping = damping
        self.num_layers = num_layers

        self.smoothing = smoothing
        self.thr = thr
        self.max_iter = max_iter
        self.method = method
        self.mu0 = mu0

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.hyperparams = hyperparams
        self.epochs = epochs
        # print("embedding size {} n_features {}".format(embedding_size, n_features))

#        self.model = Net(n_features=n_features, target_size=target_size)
        self.model = make_fc(num_layers=self.num_layers,num_features=n_features)
        #self.model.apply(weight_init)
#        w1 = self.model[0].weight
#        print(w1)

        self.optimizer = optimizer(self.model.parameters(), **hyperparams)

    def fit(self, feature, value):
        logging.info("Intopt")
        train_df = MyCustomDataset(feature, value)

        criterion = nn.L1Loss(reduction='mean')  # nn.MSELoss(reduction='mean')
        grad_list = np.zeros(self.epochs)
        for e in range(self.epochs):
          total_loss = 0
#          for parameters in self.model.parameters():
#            print(parameters)
          if e < 0:
            #print('stage 1')
            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size, shuffle=False)
            for feature, value in train_dl:
                self.optimizer.zero_grad()
                op = self.model(feature).squeeze()
#                print(feature, value, op)
#                print(feature.shape, value.shape, op.shape)
                # targetNum=1: torch.Size([10, 4096]) torch.Size([10]) torch.Size([10])
                # targetNum=2: torch.Size([10, 4096]) torch.Size([10, 2]) torch.Size([10, 2])
#                print(value, op)
                
                loss = criterion(op, value)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            #print("Epoch{} ::loss {} ->".format(e,total_loss))
                
          else:
#            if e > 4:
#                for param_group in self.optimizer.param_groups:
#                    param_group['lr'] = 1e-10
            #print('stage 2')
            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size, shuffle=False)
            
            num = 0
            batchCnt = 0
            loss = Variable(torch.tensor(0.0, dtype=torch.double), requires_grad=True)
            for feature, value in train_dl:
                self.optimizer.zero_grad()
                op = self.model(feature).squeeze()
                while torch.min(op) <= 0 or torch.isnan(op).any() or torch.isinf(op).any():
                    self.optimizer.zero_grad()
#                    self.model.__init__(self.n_features, self.target_size)
                    self.model = make_fc(num_layers=self.num_layers,num_features=self.n_features)
                    op = self.model(feature).squeeze()
  
                price = np.zeros(itemNum)
                penaltyVector = np.zeros(itemNum)
                for i in range(itemNum):
                    price[i] = self.c[i+num*itemNum]
                    penaltyVector[i] = self.penalty[i+num*itemNum]
                    op[i] = op[i]
                    
                c_torch = torch.from_numpy(price).float()
                h_torch = torch.from_numpy(self.h).float()
                A_torch = torch.from_numpy(self.A).float()
                b_torch = torch.from_numpy(self.b).float()
                penalty_torch = torch.from_numpy(penaltyVector).float()
                
                G_torch = torch.zeros((itemNum+1, itemNum))
                for i in range(itemNum):
                    G_torch[i][i] = 1
                G_torch[itemNum] = value[:, 1]
                
#                op_torch = torch.zeros((itemNum+1, itemNum))
#                for i in range(itemNum):
#                    op_torch[i][i] = 1
#                op_torch[itemNum] = op
                
#                print(G_torch)
#                print(op_torch)
                x = IPOfunc(A=A_torch, b=b_torch, h=h_torch, cTrue=-c_torch, GTrue=G_torch, penalty=penalty_torch, max_iter=self.max_iter, thr=self.thr, damping=self.damping,
                            smoothing=self.smoothing)(op)
                #print(c_torch.shape, G_torch.shape, x.shape)    # torch.Size([242]) torch.Size([43, 242]) torch.Size([242])
                x_org = x / ip_model_wholeFile.violateFactor
#                print(x, c_torch)
#                newLoss = (x * c_torch).sum() + torch.dot(torch.mul(c_torch, penalty), torch.mul(x, 1-1/ip_model_wholeFile.violateFactor))
                newLoss = (x_org * -c_torch).sum() + torch.dot(torch.mul(-c_torch, 1+penalty_torch), torch.mul(x_org, ip_model_wholeFile.violateFactor - 1))
#                print(newLoss)
#                newLoss = - (x * c_torch).sum()
#                loss = loss - (x * c_torch).sum()
                loss = loss + newLoss
                batchCnt = batchCnt + 1
#                print(loss)
#                loss = torch.dot(-c_torch, x)
#                print(loss.shape)
                  
#                print(x)
                #loss = -(x * value).mean()
                #loss = Variable(loss, requires_grad=True)
                total_loss += newLoss.item()
                # op.retain_grad()
                #print(loss)
                
                newLoss.backward()
                #print("backward1")
                self.optimizer.step()
                
                # when training size is large
#                if batchCnt % 2 == 0:
#                    newLoss.backward()
#                    #print("backward1")
#                    self.optimizer.step()
                num = num + 1
                
          
          logging.info("EPOCH Ends")
          #print("Epoch{}".format(e))
#          print("Epoch{} ::loss {} ->".format(e,total_loss/trainSize))
          grad_list[e] = total_loss
#          for param_group in self.optimizer.param_groups:
#            print(param_group['lr'])
#          if e > 1 and grad_list[e] == grad_list[e-1] and grad_list[e-1] == grad_list[e-2]:
#            break
          if e > 0 and grad_list[e] >= grad_list[e-1]:
            break
#          if total_loss > -200000:
#            break
#          else:
#            currentBestLoss = total_loss
#          if total_loss > -500:
#            break
#           print(self.val_loss(valid_econ, valid_prop))
          # print("______________")

    def val_loss(self, cap, feature, value):
        valueTemp = value.numpy()
#        test_instance = len(valueTemp) / self.batch_size
        test_instance = np.size(valueTemp, 0) / self.batch_size
#        itemVal = self.c.tolist()
        itemVal = self.c
        real_obj = actual_obj(itemVal, cap, value[:, 1], n_instance=int(test_instance))
#        print(np.sum(real_obj))

        self.model.eval()
        criterion = nn.L1Loss(reduction='mean')  # nn.MSELoss(reduction='sum')
        valid_df = MyCustomDataset(feature, value)
        valid_dl = data_utils.DataLoader(valid_df, batch_size=self.batch_size, shuffle=False)

        obj_list = []
        corr_obj_list = []
        len = np.size(valueTemp, 0)
        predVal = torch.zeros((len, 2))
        
        num = 0
        for feature, value in valid_dl:
            op = self.model(feature).squeeze()
#            print(op)
            loss = criterion(op, value)

            realWT = {}
            predWT = {}
            realPrice = {}
            predPrice = {}
            penaltyVector = np.zeros(itemNum)
            for i in range(itemNum):
                realWT[i] = value[i][1]
                predWT[i] = op[i][1]
                realPrice[i] = value[i][0]
                predPrice[i] = op[i][0]
                predVal[i+num*itemNum][0] = op[i][0]
                predVal[i+num*itemNum][1] = op[i][1]
                penaltyVector[i] = self.penalty[i+num*itemNum]

            corrrlst = correction_single_obj(realPrice, predPrice, cap, realWT, predWT, penaltyVector)
            corr_obj_list.append(corrrlst)
            num = num + 1
            

        self.model.train()
#        print(corr_obj_list-real_obj)
#        print(np.sum(corr_obj_list))
#        return prediction_loss, abs(np.array(obj_list) - real_obj)
        return abs(np.array(corr_obj_list) - real_obj), predVal


#c_dataTemp = np.loadtxt('KS_c.txt')
#c_data = c_dataTemp[:itemNum]

h_data = np.ones(itemNum+1)
h_data[itemNum] = capacity
A_data = np.zeros((2, itemNum))
b_data = np.zeros(2)


#startmark = int(sys.argv[1])
#endmark = startmark + 30

print("*** HSD ****")

#for testmark in range(startmark, endmark):
    #recordFile = open('record(' + str(testmark) + ').txt', 'a')
testTime = 10
recordBest = np.zeros((1, testTime))

ReLUValue = 0
if penaltyTerm == 0:
    ReLUValue = 15
elif penaltyTerm == 0.25 or penaltyTerm == 0.5:
    ReLUValue = 23
elif penaltyTerm == 1:
    ReLUValue = 24
elif penaltyTerm == 2:
    ReLUValue = 26


for testi in range(testTime):
    print(testi)
    
    c_train = np.loadtxt('./data/train_prices/train_prices(' + str(testi) + ').txt')
    x_train = np.loadtxt('./data/train_features/train_features(' + str(testi) + ').txt')
    y_train1 = np.loadtxt('./data/train_prices/train_prices(' + str(testi) + ').txt')
    y_train2 = np.loadtxt('./data/train_weights/train_weights(' + str(testi) + ').txt')
    penalty_train = np.loadtxt('./data/train_penalty' + str(penaltyTerm) + '/train_penalty(' + str(testi) + ').txt')
    meanPriceValue = np.mean(y_train1)
    meanWeightValue = np.mean(y_train2)

    y_train = np.zeros((y_train1.size, 2))
    for i in range(y_train1.size):
        y_train[i][0] = y_train1[i]
        y_train[i][1] = y_train2[i]
    feature_train = torch.from_numpy(x_train).float()
    value_train = torch.from_numpy(y_train).float()
    
    c_test = np.loadtxt('./data/test_prices/test_prices(' + str(testi) + ').txt')
    x_test = np.loadtxt('./data/test_features/test_features(' + str(testi) + ').txt')
    y_test1 = np.loadtxt('./data/test_prices/test_prices(' + str(testi) + ').txt')
    y_test2 = np.loadtxt('./data/test_weights/test_weights(' + str(testi) + ').txt')
    penalty_test = np.loadtxt('./data/test_penalty' + str(penaltyTerm) + '/test_penalty(' + str(testi) + ').txt')

    y_test = np.zeros((y_test1.size, 2))
    for i in range(y_test1.size):
        y_test[i][0] = y_test1[i]
        y_test[i][1] = y_test2[i]
    feature_test = torch.from_numpy(x_test).float()
    value_test = torch.from_numpy(y_test).float()
    
    start = time.time()
    damping = 1e-2
    thr = 1e-3
    lr = 1e-7
#    lr = 1e-5
    bestTrainCorrReg = float("inf")
    for j in range(1):
        clf = Intopt(c_train, h_data, A_data, b_data, penalty_train, damping=damping, lr=lr, n_features=featureNum, thr=thr, epochs=8)
        clf.fit(feature_train, value_train)
        train_rslt, predTrainVal = clf.val_loss(capacity, feature_train, value_train)
        avgTrainCorrReg = np.mean(train_rslt)
    #    trainHSD_rslt = str(testmark) + ' train: ' + str(np.sum(train_rslt[1])) + ' ' + str(np.mean(train_rslt[1]))
        trainHSD_rslt = 'train: ' + str(np.mean(train_rslt))

        if avgTrainCorrReg < bestTrainCorrReg:
            bestTrainCorrReg = avgTrainCorrReg
            torch.save(clf.model.state_dict(), 'model.pkl')
        print(trainHSD_rslt)
        
#        if avgTrainCorrReg < 50:
#            break

#        val_rslt = clf.val_loss(source, sink, arc, feature_test, value_test)
#        #HSD_rslt = str(testmark) + ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
#        HSD_rslt = ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
#        print(HSD_rslt)

#    val_rslt = clf.val_loss(source, sink, arc, feature_test, value_test)
##    HSD_rslt = str(testmark) + ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
#    HSD_rslt = ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
#    print(HSD_rslt)
#    print('\n')
#    recordBest[0][i] = np.sum(val_rslt[1])

    clfBest = Intopt(c_test, h_data, A_data, b_data, penalty_test, damping=damping, lr=lr, n_features=featureNum, thr=thr, epochs=8)
    clfBest.model.load_state_dict(torch.load('model.pkl'))

    val_rslt, predTestVal = clfBest.val_loss(capacity, feature_test, value_test)
    #HSD_rslt = str(testmark) + ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
#    print(predTestVal.shape)
    end = time.time()

    predTestVal = predTestVal.detach().numpy()
#    print(predTestVal.shape)
    predTestVal1 = predTestVal[:, 0]
    predTestVal2 = predTestVal[:, 1]
    predValuePrice = np.zeros((predTestVal1.size, 2))
    for i in range(predTestVal1.size):
#        predValue[i][0] = int(i/itemNum)
        predValuePrice[i][0] = y_test1[i]
        predValuePrice[i][1] = predTestVal1[i]
#    np.savetxt('./data/proposed_prices200/proposed_prices' + str(penaltyTerm) + '(' + str(testi) + ').txt', predValuePrice, fmt="%.2f")
    predValueWeight = np.zeros((predTestVal2.size, 2))
    for i in range(predTestVal2.size):
#        predValue[i][0] = int(i/itemNum)
        predValueWeight[i][0] = y_test2[i]
        predValueWeight[i][1] = predTestVal2[i]
#    np.savetxt('./data/proposed_weights200/proposed_weights' + str(penaltyTerm) + '(' + str(testi) + ').txt', predValueWeight, fmt="%.2f")
    
    HSD_rslt = 'test: ' + str(np.mean(val_rslt))
    print(HSD_rslt)
    print ('Elapsed time: ' + str(end-start))
    recordBest[0][testi] = np.sum(val_rslt)
    
#    value = clfBest.model(feature_test).squeeze()
#    value = value.detach().numpy()
#    predValue = np.zeros((value.size, 2))
#    for i in range(value.size):
##        predValue[i][0] = int(i/itemNum)
#        predValue[i][0] = value_test[i]
#        predValue[i][1] = value[i]
##    np.savetxt('./proposed_GEANT/proposed_GEANT(' + str(testi) + ').txt', predValue, fmt="%.2f")
##    np.savetxt('./proposed_100/proposed_100(' + str(testi) + ').txt', predValue, fmt="%.2f")
#    np.savetxt('./CombOptNet/proposed_weights.txt', predValue, fmt="%.2f")

print(recordBest)
