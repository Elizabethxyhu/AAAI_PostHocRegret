# use ip_model_whole(logKKT(Gh)_hUn).py

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
import gurobipy as gp
import logging
import copy
from collections import defaultdict
import joblib
import gurobipy as gp
from gurobipy import GRB

nodeNum = 12
edgeNum = 18
featureNum = 8

class Graph:

    def __init__(self, arc, realCap, preCap):
        self.arc = arc
        self.realCap = realCap
        self.preCap = preCap

        self.realGraph = np.zeros((nodeNum, nodeNum))
        for i in range(edgeNum):
            self.realGraph[arc[i][0], arc[i][1]] = realCap[i]

        self.preGraph = np.zeros((nodeNum, nodeNum))
        for i in range(edgeNum):
            self.preGraph[arc[i][0], arc[i][1]] = preCap[i]

        self.ROW = len(self.realGraph)

    # self.COL = len(gr[0])

    '''Returns true if there is a path from source 's' to sink 't' in
    residual graph. Also fills parent[] to store the path '''

    def BFS(self, s, t, parent, graph):

        # Mark all the vertices as not visited
        visited = [False] * (self.ROW)
        # Create a queue for BFS
        queue = []
        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True
        # Standard BFS Loop
        while queue:
            # Dequeue a vertex from queue and print it
            u = queue.pop(0)
            # Get all adjacent vertices of the dequeued vertex u
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            for ind, val in enumerate(graph[u]):
                if visited[ind] == False and val > 0:
                    # If we find a connection to the sink node,
                    # then there is no point in BFS anymore
                    # We just have to set its parent and can return true
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u
                    if ind == t:
                        return True
        # We didn't reach sink in BFS starting
        # from source, so return false
        return False

    # Returns tne maximum flow from s to t in the given graph
    def maxFlow(self, source, sink):
        # graphTemp = self.realGraph.copy()
        # print(self.realGraph)
        # This array is filled by BFS and to store path
        parent = [-1] * (self.ROW)
        max_flow = 0  # There is no flow initially
        # Augment the flow while there is path from source to sink
        while self.BFS(source, sink, parent, self.realGraph):
            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = float("Inf")
            s = sink
            while (s != source):
                path_flow = min(path_flow, self.realGraph[parent[s]][s])
                s = parent[s]
            # Add path flow to overall flow
            max_flow += path_flow
            # update residual capacities of the edges and reverse edges
            # along the path
            v = sink
            while (v != source):
                u = parent[v]
                self.realGraph[u][v] -= path_flow
                self.realGraph[v][u] += path_flow
                v = parent[v]

        # reset realGraph
        # for i in range(nodeNum):
        #     for j in range(nodeNum):
        #         self.realGraph[i][j] = graphTemp[i][j]
        # print(self.realGraph)
        return max_flow

    def maxFlow_model(self, source, sink):
        # graphTemp = self.preGraph.copy()
        # print(self.realGraph)
        # This array is filled by BFS and to store path
        parent = [-1] * (self.ROW)
        max_flow = 0  # There is no flow initially
        # Augment the flow while there is path from source to sink
        while self.BFS(source, sink, parent, self.preGraph):
            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = float("Inf")
            path_flow_index = sink
            s = sink
            while (s != source):
                # path_flow = min(path_flow, self.preGraph[parent[s]][s])
                if path_flow > self.preGraph[parent[s]][s]:
                    path_flow = self.preGraph[parent[s]][s]
                    path_flow_index = s
                s = parent[s]

            preDf = path_flow # path_flow = self.preGraph[parent[path_flow_index]][path_flow_index]
            df = self.realGraph[parent[path_flow_index]][path_flow_index]
            # Add path flow to overall flow
            max_flow += df
            # print(preDf, df)
            # update residual capacities of the edges and reverse edges
            # along the path
            v = sink
            while (v != source):
                u = parent[v]
                # self.preGraph[u][v] -= path_flow
                # self.preGraph[v][u] += path_flow
                self.preGraph[u][v] -= preDf
                self.realGraph[u][v] -= df
                self.preGraph[v][u] += preDf
                self.realGraph[v][u] += df
                v = parent[v]

        # reset realGraph
        # for i in range(nodeNum):
        #     for j in range(nodeNum):
        #         self.preGraph[i][j] = graphTemp[i][j]
        # print(self.realGraph)
        return max_flow
        
    def corr_maxFlow(self, source, sink):
        graphTemp = self.preGraph.copy()
        realGraphTemp = self.realGraph.copy()
        preFlow = np.zeros([nodeNum, nodeNum])
        # print(self.realGraph)
        # This array is filled by BFS and to store path
        parent = [-1] * (self.ROW)
        max_flow = 0  # There is no flow initially
        # Augment the flow while there is path from source to sink
        while self.BFS(source, sink, parent, self.preGraph):
            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = float("Inf")
            path_flow_index = sink
            s = sink
            while (s != source):
                # path_flow = min(path_flow, self.preGraph[parent[s]][s])
                # print(parent[s], "->", end="")
                if path_flow > self.preGraph[parent[s]][s]:
                    path_flow = self.preGraph[parent[s]][s]
                    path_flow_index = s
                s = parent[s]

            preDf = path_flow # path_flow = self.preGraph[parent[path_flow_index]][path_flow_index]
            df = self.realGraph[parent[path_flow_index]][path_flow_index]
            # Add path flow to overall flow
            max_flow += df
            # print(df)
            # print(preDf, df)
            # update residual capacities of the edges and reverse edges
            # along the path
            v = sink
            while (v != source):
                u = parent[v]
                # self.preGraph[u][v] -= path_flow
                # self.preGraph[v][u] += path_flow
                self.preGraph[u][v] -= preDf
                self.realGraph[u][v] -= df
                preFlow[u][v] = preFlow[u][v] + df
                self.preGraph[v][u] += preDf
                self.realGraph[v][u] += df
                v = parent[v]

        # print(realGraphTemp)
        # np.savetxt("realGraphTemp.txt", realGraphTemp, fmt="%.2f")
        # print(self.realGraph)
        # np.savetxt("realGraph.txt", self.realGraph, fmt="%.2f")
        # print(preFlow)
        # np.savetxt("preFlow.txt", preFlow, fmt="%.2f")
        # check whether violate constraint and find tau
        tauTemp = []
        for i in range(nodeNum):
            for j in range(nodeNum):
                if preFlow[i][j] > realGraphTemp[i][j]:
                    newT = realGraphTemp[i][j]/preFlow[i][j]
                    # print(preFlow[i][j], realGraphTemp[i][j], newT)
                    tauTemp.append(newT)
                # if self.realGraph[i][j] < 0 and realGraphTemp[i][j] > 0:
                #     newT = realGraphTemp[i][j]/(-self.realGraph[i][j]+realGraphTemp[i][j])
                #     tauTemp.append(newT)
        # print(tauTemp)
        if len(tauTemp) > 0:
            tau = min(tauTemp)
            max_flow = tau * max_flow

        # reset realGraph
        for i in range(nodeNum):
            for j in range(nodeNum):
                self.preGraph[i][j] = graphTemp[i][j]
        # print(self.realGraph)
        return max_flow


def actual_obj(source, sink, arc, capT, n_instance):
    obj_list = []
    for num in range(n_instance):
        cap = np.zeros(edgeNum)
        cnt = num * edgeNum
        for i in range(edgeNum):
            cap[i] = capT[cnt]
            cnt = cnt + 1
        g = Graph(arc, cap, cap)
        objective = g.maxFlow(source, sink)
        obj_list.append(objective)
    return np.array(obj_list)

def pred_single_obj(c, G, prehTemp):
    preh = np.zeros(edgeNum)
    for i in range(edgeNum):
        preh[i] = prehTemp[i]

    m = gp.Model()
    m.setParam('OutputFlag', 0)
    x = m.addVars(allPathNum, vtype=GRB.CONTINUOUS, name='x')
    m.setObjective(x.prod(c), GRB.MAXIMIZE)
#        m.addConstr((x.prod(G)) <= h)
    for i in range(edgeNum):
        m.addConstr((x.prod(G[i])) <= preh[i])

    m.optimize()
    sol = []
    for i in range(allPathNum):
        sol.append(x[i].x)
    objective = m.objVal
    
    return objective

def correction_single_obj(c, G, realhTemp, prehTemp):
    realh = np.zeros(edgeNum)
    preh = np.zeros(edgeNum)
    for i in range(edgeNum):
        realh[i] = realhTemp[i]
        preh[i] = prehTemp[i]
    if min(preh) >= 0:
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(allPathNum, vtype=GRB.CONTINUOUS, name='x')
        m.setObjective(x.prod(c), GRB.MAXIMIZE)
    #        m.addConstr((x.prod(G)) <= h)
        for i in range(edgeNum):
            m.addConstr((x.prod(G[i])) <= preh[i])

        m.optimize()
        sol = []
    #    print(x)
        for i in range(allPathNum):
            sol.append(x[i].x)
    #        for i in range(allPathNum):
    #            if sol[i] != 0:
    #                print(i, end=" ")
        objective = m.objVal
        
        #correction
        Gx = np.dot(G, sol)
    #        print(Gx, objective)
        tauTemp = []
        for i in range(edgeNum):
            if Gx[i] > realh[i]:
                newT = realh[i]/Gx[i]
                tauTemp.append(newT)
        if len(tauTemp) > 0:
            tau = min(tauTemp)
            objective = tau * objective
            sol = np.multiply(sol, tau)
            
    else:
        objective = 0

#        print(np.dot(G, sol), objective)
#        print("")
    return objective
    
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def make_fc(num_layers, num_features, num_targets=1,
            activation_fn = nn.ReLU,intermediate_size=2*featureNum, regularizers = True):
    net_layers = [nn.Linear(num_features, intermediate_size),
         activation_fn()]
    for hidden in range(num_layers-2):
        net_layers.append(nn.Linear(intermediate_size, intermediate_size))
        net_layers.append(activation_fn())
    net_layers.append(nn.Linear(intermediate_size, num_targets))
    net_layers.append(nn.ReLU())
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
from ip_model_whole import IPOfunc

class Intopt:
    def __init__(self, c, G, A, b, n_features, num_layers=3, smoothing=False, thr=0.1, max_iter=None, method=1, mu0=None,
                 damping=0.5, target_size=1, epochs=8, optimizer=optim.Adam,
                 batch_size=edgeNum, **hyperparams):
        self.c = c
        self.G = G
        self.A = A
        self.b = b
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

    def fit(self, src, dist, arc, feature, value):
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
                #print(op)
                
                loss = criterion(op, value)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            #print("Epoch{} ::loss {} ->".format(e,total_loss))
                
          else:
            #print('stage 2')
            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size, shuffle=False)
            
            for feature, value in train_dl:
                self.optimizer.zero_grad()
                op = self.model(feature).squeeze()
                while torch.min(op) <= 0 or torch.isnan(op).any() or torch.isinf(op).any():
                    self.optimizer.zero_grad()
#                    self.model.__init__(self.n_features, self.target_size)
                    self.model = make_fc(num_layers=self.num_layers,num_features=self.n_features)
                    op = self.model(feature).squeeze()
    
                c_torch = torch.from_numpy(self.c).float()
                G_torch = torch.from_numpy(self.G).float()
                A_torch = torch.from_numpy(self.A).float()
                b_torch = torch.from_numpy(self.b).float()
  
                x = IPOfunc(A=A_torch, b=b_torch, G=G_torch, c=-c_torch, hTrue=value, max_iter=self.max_iter, thr=self.thr, damping=self.damping,
                            smoothing=self.smoothing)(op)
                
                loss = -(x * c_torch).mean()
                total_loss += loss.item()
                # op.retain_grad()
                #print(loss)
                loss.backward()
                #print("backward1")
                self.optimizer.step()
                  
          logging.info("EPOCH Ends")
          #print("Epoch{}".format(e))
#          print("Epoch{} ::loss {} ->".format(e,total_loss))
          grad_list[e] = total_loss
          if e > 0 and grad_list[e] >= grad_list[e-1]:
            break
          # print(self.val_loss(valid_econ, valid_prop))
          # print("______________")

    def val_loss(self, src, dist, arc, feature, value):
        valueTemp = value.numpy()
        test_instance = len(valueTemp) / self.batch_size
        real_obj = actual_obj(src, dist, arc, value, n_instance=int(test_instance))
#        print(real_obj)

        self.model.eval()
        criterion = nn.L1Loss(reduction='mean')  # nn.MSELoss(reduction='sum')
        valid_df = MyCustomDataset(feature, value)
        valid_dl = data_utils.DataLoader(valid_df, batch_size=self.batch_size, shuffle=False)
        prediction_loss = 0
        obj_list = []
        corr_obj_list = []
        num = 0
        for feature, value in valid_dl:
            op = self.model(feature).squeeze()
#            print(op)
            loss = criterion(op, value)
            prediction_loss += loss.item()

            realCap = {}
            preCap = {}
            for i in range(edgeNum):
                realCap[i] = value[i]
                preCap[i] = op[i]
            g = Graph(arc, realCap, preCap)
            # predrlst, predSol = minCost_model(src, dist, arc, capacity, cost, realCost, True)
            predrlst = g.maxFlow_model(src, dist)
            c_list = self.c.tolist()
            G_list = self.G.tolist()
#            predrlst = pred_single_obj(c_list, G_list, preCap)
            obj_list.append(predrlst)
            
#            corrG = Graph(arc, realCap, preCap)
#            corrrlst = corrG.corr_maxFlow(src, dist)
            corrrlst = correction_single_obj(c_list, G_list, realCap, preCap)
            corr_obj_list.append(corrrlst)
            
            num = num + 1

        self.model.train()
#        print(corr_obj_list)
#        return prediction_loss, abs(np.array(obj_list) - real_obj)
        return abs(np.array(obj_list) - real_obj), abs(np.array(corr_obj_list) - real_obj)


arc = np.loadtxt('./data/graph_POLSKA.txt', dtype=int)

c_data = np.loadtxt('./data/POLSKA_C011.txt')
G_data = np.loadtxt('./data/POLSKA_G011.txt')
rowSize = edgeNum
colSize = 13
allPathNum = 13
A_data = np.zeros((2, colSize))
b_data = np.zeros(2)

print("*** HSD ****")

source = 0
sink = nodeNum - 1

testTime = 10
recordBest = np.zeros((1, testTime))

for testi in range(testTime):
    print(testi)
    trainData = np.loadtxt('./data/train_POLSKA/train_POLSKA(' + str(testi) + ').txt')
#    trainData = np.loadtxt('train_POLSKA.txt')
    x_train = trainData[:, 1:featureNum+1]
    y_train = trainData[:, featureNum+1]
    feature_train = torch.from_numpy(x_train).float()
    value_train = torch.from_numpy(y_train).float()

    testData = np.loadtxt('./data/test_POLSKA/test_POLSKA(' + str(testi) + ').txt')
#    testData = np.loadtxt('test_POLSKA.txt')
    x_test = testData[:, 1:featureNum+1]
    y_test = testData[:, featureNum+1]
    feature_test = torch.from_numpy(x_test).float()
    value_test = torch.from_numpy(y_test).float()

    damping = 1e-2
    thr = 1e-3
    lr = 1e-5
    #lr = 1e-2
    bestTrainCorrReg = float("inf")
    for j in range(3):
        clf = Intopt(c_data, G_data, A_data, b_data, damping=damping, lr=lr, n_features=featureNum, thr=thr, epochs=6)
        clf.fit(source, sink, arc, feature_train, value_train)
        train_rslt = clf.val_loss(source, sink, arc, feature_train, value_train)
        avgTrainCorrReg = np.mean(train_rslt[1])
    #    trainHSD_rslt = str(testmark) + ' train: ' + str(np.sum(train_rslt[1])) + ' ' + str(np.mean(train_rslt[1]))
        trainHSD_rslt = ' train: ' + str(np.mean(train_rslt[0])) + ' ' + str(np.mean(train_rslt[1]))
    
        if avgTrainCorrReg < bestTrainCorrReg:
            bestTrainCorrReg = avgTrainCorrReg
            torch.save(clf.model.state_dict(), 'model.pkl')
#        print(trainHSD_rslt)
        
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

    clfBest = Intopt(c_data, G_data, A_data, b_data, damping=damping, lr=lr, n_features=featureNum, thr=thr, epochs=6)
    clfBest.model.load_state_dict(torch.load('model.pkl'))
#
    value = clfBest.model(feature_test).squeeze()
    value = value.detach().numpy()
    predValue = np.zeros((value.size, 3))
    for i in range(value.size):
        predValue[i][0] = int(i/edgeNum)
        predValue[i][1] = value_test[i]
        predValue[i][2] = value[i]
#    np.savetxt('./data/proposed_POLSKA/proposed_POLSKA(' + str(testi) + ').txt', predValue, fmt="%.2f")

    val_rslt = clfBest.val_loss(source, sink, arc, feature_test, value_test)
    #HSD_rslt = str(testmark) + ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
    HSD_rslt = ' avgPReg: ' + str(np.mean(val_rslt[1]))
    print(HSD_rslt)
    recordBest[0][testi] = np.sum(val_rslt[1])

print(recordBest)
