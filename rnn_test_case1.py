#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:07:21 2020

@author: liuwenliang
"""

import numpy as np
import torch
import math
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch import nn
from scipy.stats.mstats import gmean
from scipy.optimize import minimize 
import random

h=0.5
scale = 2000
scale_w = 3
l = 0.01
alpha = 0.7
INPUT_SIZE = 3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
n=10              # Test the model for n times

def tansig(x,u_min,u_max):
    return u_min + (u_max-u_min)*(torch.tanh(x)+1)/2
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,         # rnn hidden unit
            num_layers=NUM_LAYERS,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            dropout = 0.5,
#            nonlinearity = 'relu'
        )
        
        self.out = nn.Linear(HIDDEN_SIZE, 2)

    def forward(self, x, h_state):    
        # x (batch, time_step, input_size)
        # r_out (batch, time_step, hidden_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(tansig(self.out(r_out[:, time_step, :]),torch.tensor([0.,-3.]),torch.tensor([2.,3.])))
        return torch.stack(outs, dim=1), h_state

def system(q0,u): 
    # system dynamic
    nb_u = u.shape[1]
    q = np.zeros((3,nb_u+1))
    q[:,0] = q0
    for i in range(nb_u):
        q[0,i+1]=q[0,i]+scale_w*u[0,i]/u[1,i]*(math.sin(q[2,i]+u[1,i]/scale_w*h)-math.sin(q[2,i]))
        q[1,i+1]=q[1,i]+scale_w*u[0,i]/u[1,i]*(math.cos(q[2,i])-math.cos(q[2,i]+u[1,i]/scale_w*h))
        q[2,i+1]=q[2,i]+u[1,i]/scale_w*h#*u[0,i]*h
    return q

def pos(r):
    nb_r = r.shape[0]
    r_p = np.zeros(nb_r)
    for i in range(nb_r):
        r_p[i] = max(0,r[i])
    return r_p

def eventually(eta): 
    if np.any(eta>0):
        return np.mean(pos(eta)) # satisfy
    else:
        #return -gmean(1-eta)+1 # not satisfy
        return -gmean(-eta) # not satisfy
    
def always(eta):
    if np.any(eta<=0):
        return np.mean(-pos(-eta)) # not satisfy
    else:
        #return gmean(1+eta)-1 # satisfy
        return gmean(eta) # satisfy
    
def conjunction(eta):
    eta_conj = np.zeros(eta.shape[1])
    for i in range(eta.shape[1]):
        if np.any(eta[:,i]<=0):
            eta_conj[i] =  np.mean(-pos(-eta[:,i])) # not satisfy
        else:
            #eta_conj[i] =  gmean(1+eta[:,i])-1 # satisfy
            eta_conj[i] =  gmean(eta[:,i]) # satisfy
    return eta_conj

def disjunction(eta):
    eta_disj = np.zeros(eta.shape[1])
    for i in range(eta.shape[1]):
        if np.any(eta[:,i]>0):
            eta_disj[i] =  np.mean(pos(eta[:,i])) # satisfy
        else:
            #eta_disj[i] =  -gmean(1-eta[:,i])+1 # not satisfy
            eta_disj[i] =  -gmean(-eta[:,i]) # not satisfy
    return eta_disj


def robustness(args):
    # assume that ||q - qgi|| <= R, then rg - ||q - qgi|| is in [rg - R, rg], normalize to [-1, rg/(rg-R)]
    
    u_h, q0, Rx, Ry, t1, t2, T, XA1, XA2, YA1, YA2, XB1, XB2, YB1, YB2, XC1, XC2, YC1, YC2, X_obs1, X_obs2, Y_obs1, Y_obs2, XM1, XM2, YM1, YM2 = args
    def v(u):
        q = system(q0, np.concatenate((u_h, u.reshape(2,-1)),axis=1))  
        MuA=np.zeros((4 ,int(t1/h)))
        MuB=np.zeros((4 ,int(t1/h)))
        MuC=np.zeros((4 ,int(t2/h)))

#        MuM=np.zeros((4 ,int(T/h)))
        MuObs=np.zeros((4 ,int(T/h)))
        
        MuA[:,0:int(t1/h)]=np.array([1/Rx*(XA1-q[0,1:int(t1/h)+1]),1/Rx*(q[0,1:int(t1/h)+1]-XA2),1/Ry*(YA1-q[1,1:int(t1/h)+1]),1/Ry*(q[1,1:int(t1/h)+1]-YA2)])
        MuB[:,0:int(t1/h)]=np.array([1/Rx*(XB1-q[0,1:int(t1/h)+1]),1/Rx*(q[0,1:int(t1/h)+1]-XB2),1/Ry*(YB1-q[1,1:int(t1/h)+1]),1/Ry*(q[1,1:int(t1/h)+1]-YB2)])
        
        MuC[:,0:int(t2/h)]=np.array([1/Rx*(XC1-q[0,1+int(t1/h):1+int(t1/h)+int(t2/h)]),1/Rx*(q[0,1+int(t1/h):1+int(t1/h)+int(t2/h)]-XC2),1/Ry*(YC1-q[1,1+int(t1/h):1+int(t1/h)+int(t2/h)]),1/Ry*(q[1,1+int(t1/h):1+int(t1/h)+int(t2/h)]-YC2)])
        
        MuObs[:,0:int(T/h)]=np.array([1/Rx*(X_obs1-q[0,1:int(T/h)+1]),1/Rx*(q[0,1:int(T/h)+1]-X_obs2),1/Ry*(Y_obs1-q[1,1:int(T/h)+1]),1/Ry*(q[1,1:int(T/h)+1]-Y_obs2)])

#        MuM[:,0:int(T/h)]=np.array([1/Rx*(XM1-q[0,1:int(T/h)+1]),1/Rx*(q[0,1:int(T/h)+1]-XM2),1/Ry*(YM1-q[1,1:int(T/h)+1]),1/Ry*(q[1,1:int(T/h)+1]-YM2)])
        
        roA = conjunction(MuA)
        roB = conjunction(MuB)
        roAvB = disjunction(np.array([roA, roB]))
        roFAvB = eventually(roAvB)

        roC = conjunction(MuC)
        roFC = eventually(roC)
        
        roObs = disjunction(MuObs)
        roGObs = always(roObs)
        
#        roM = conjunction(MuM)
#        roGM = always(roM)
        
        roTotal = np.array([roFAvB, roFC, roGObs])
        ro = conjunction(roTotal.reshape(-1,1))# + 1e-3 * np.sum(LA.norm(u.reshape(2,-1),axis=0))

        return ro
    return v

def qpsolver(args):
    u_ref = args
    def obj(u):
        return (u[0]-u_ref[0])**2 + l*(u[1]-u_ref[1])**2 #u[0]**2 - 2*u_ref[0]*abs(u[0])*math.cos(u_ref[1]*h-u[1]*h)
    return obj

def qpJacobian(args):
    u_ref = args
    def Jac(u):
        return np.array([2*(u[0]-u_ref[0]), 2*l*(u[1]-u_ref[1])])
    return Jac

def B(q,o,r):
    return (q[0]-o[0])**2 + (q[1]-o[1])**2 - r**2

def con(args): 
    # constraints 
    # umin<u<umax
    # min<q<qmax

    q,o1,r1,o2,r2,o3,r3,o4,r4= args
        
    cons = ({'type': 'ineq', 'fun': lambda u: B(system(q,u.reshape(2,1))[:,1],o1,r1) + (alpha-1)*B(q,o1,r1)},\
            {'type': 'ineq', 'fun': lambda u: B(system(q,u.reshape(2,1))[:,1],o2,r2) + (alpha-1)*B(q,o2,r2)},\
            {'type': 'ineq', 'fun': lambda u: B(system(q,u.reshape(2,1))[:,1],o3,r3) + (alpha-1)*B(q,o3,r3)},\
            {'type': 'ineq', 'fun': lambda u: B(system(q,u.reshape(2,1))[:,1],o4,r4) + (alpha-1)*B(q,o4,r4)})
#             {'type': 'ineq', 'fun': lambda u: ((q[0]-o3[0])*math.cos(q[2])+(q[1]-o3[1])*math.sin(q[2]))*u[0] + 1*B(q,o3,r3)})
#                {'type': 'ineq', 'fun': lambda u: -system(q0,u) + qmax})
    return cons
    

#q0 = np.array([1,1,0])
Rx = 10
Ry = 10
t1 = 5
t2 = 5
T = t1 + t2
XA1=3; XA2=1; YA1=9; YA2=7
XB1=9; XB2=7; YB1=3; YB2=1
XC1=9; XC2=7; YC1=9; YC2=7
XM1=Rx; XM2=0; YM1=Ry; YM2=0
obs_center = np.array([5, 5])
obs_length = 1
X_obs1=obs_center[0]-obs_length; X_obs2=obs_center[0]+obs_length; Y_obs1=obs_center[1]-obs_length; Y_obs2=obs_center[1]+obs_length



rnn = torch.load('rnn_case1.pkl')
rnn.eval()


ro_sum = 0
nb_suc = 0
start = datetime.datetime.now()
ref_sum = datetime.datetime.now()-datetime.datetime.now()
ref_num = 0
Q=np.empty((0,3,21))
for i in range(n):
    o1 = np.array([3.5+3*random.random(),1.5])
    r1 = 0.5
    o2 = np.array([1.5,3.5+3*random.random()])
    r2 = 0.5
    o3 = np.array([3.5+3*random.random(),8.5])
    r3 = 0.5
    o4 = np.array([8.5,3.5+3*random.random()])
    r4 = 0.5

    x0 = 0.5+1.5*np.random.rand(1)
    y0 = 0.5+1.5*np.random.rand(1)
    theta0 = math.pi/2*np.random.rand(1)
    q0 = np.array([x0,y0,theta0]).reshape(-1)
    args = (np.empty((2,0)),q0, Rx, Ry, t1, t2, T, XA1, XA2, YA1, YA2, XB1, XB2, YB1, YB2, XC1, XC2, YC1, YC2, X_obs1, X_obs2, Y_obs1, Y_obs2, XM1, XM2, YM1, YM2)
    F_ro = robustness(args)
    q = np.zeros((3,int(T/h)+1))
    u = np.zeros((2,int(T/h)))
    
    q[:,0] = q0
    h_state = None
    for i in range(int(T/h)):
        start0 = datetime.datetime.now()
        q_tensor = torch.from_numpy(q[:,i]).float().view(1,1,3)
        u_tensor, h_state_n = rnn(q_tensor, h_state)
        u_ref = u_tensor.detach().numpy().reshape(-1)
        h_state = h_state_n
        end0 = datetime.datetime.now()
        ref_sum = ref_sum + end0-start0
        ref_num = ref_num + 1        
        args_cbf = u_ref #reference control
        args_cbf_cons = (q[:,i], o1,r1,o2,r2,o3,r3,o4,r4)
        res_qp = minimize(qpsolver(args_cbf), np.ones(2), method='SLSQP', jac = qpJacobian(args_cbf), constraints=con(args_cbf_cons))        
        u[:,i] = res_qp.x
        

        
        
        q[0,i+1]=q[0,i]+scale_w*u[0,i]/u[1,i]*(math.sin(q[2,i]+u[1,i]/scale_w*h)-math.sin(q[2,i]))
        q[1,i+1]=q[1,i]+scale_w*u[0,i]/u[1,i]*(math.cos(q[2,i])-math.cos(q[2,i]+u[1,i]/scale_w*h))
        q[2,i+1]=q[2,i]+u[1,i]/scale_w*h#*u[0,i]*h
        
    
    ro = F_ro(u)
    # Q = np.concatenate((Q,q.reshape(1,3,-1)),axis = 0)
    ax = plt.gca()
    ax.cla() # clear things for fresh plot
    rectI = patches.Rectangle((0.5,0.5),1.5,1.5,edgecolor='g',facecolor='none')
    rectA = patches.Rectangle((XA2,YA2),XA1-XA2,YA1-YA2,edgecolor='none',facecolor='lightblue')
    rectB = patches.Rectangle((XB2,YB2),XB1-XB2,YB1-YB2,edgecolor='none',facecolor='lightblue')
    rectC = patches.Rectangle((XC2,YC2),XC1-XC2,YC1-YC2,edgecolor='none',facecolor='y')
    rectObs = patches.Rectangle((X_obs1,Y_obs1),X_obs2-X_obs1,Y_obs2-Y_obs1,edgecolor='none',facecolor='r')
    ax.add_patch(rectI)
    ax.add_patch(rectA)
    ax.add_patch(rectB)
    ax.add_patch(rectC)
    ax.add_patch(rectObs)
    plt.text(0.65,0.65,'Init',fontsize=12)
    plt.text(XA2+0.02,YA2+1.5,'RegA',fontsize=12)
    plt.text(XB2+0.7,YB2+0.15,'RegB',fontsize=12)
    plt.text(XC2+0.02,YC2+0.15,'RegC',fontsize=12)
    plt.text(X_obs1+0.15,Y_obs1+0.15,'Obs',fontsize=12)
    circle1 = plt.Circle(o1, r1, color='r', fill=False)
    ax.add_artist(circle1)
    circle2 = plt.Circle(o2, r2, color='r', fill=False)
    ax.add_artist(circle2)
    circle3 = plt.Circle(o3, r3, color='r', fill=False)
    ax.add_artist(circle3)
    circle4 = plt.Circle(o4, r4, color='r', fill=False)
    ax.add_artist(circle4)
    plt.plot(q[0,:], q[1,:])
    plt.scatter(q[0,:], q[1,:],alpha=0.6, zorder = 30)
    ax.set_xlim((0, 10))
    ax.set_ylim((0, 10))
    ax.set_aspect(1)
#    plt.savefig('traj7.png',dpi=300,pad_inches=0.0,bbox_inches='tight')
    plt.show()
    
    print(ro)
    ro_sum = ro_sum + ro
    if ro > 0:
        nb_suc = nb_suc+1


    


end = datetime.datetime.now()
print('average time: ')
print((end-start)/n)
print('success rate:')
print(nb_suc/n)
print('average robustness:')
print(ro_sum/n)
print('average cbf time:')
print(ref_sum/ref_num)
