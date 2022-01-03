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
from numpy import linalg as LA
import random
import torch.nn.functional as F

INPUT_SIZE = 2
HIDDEN_SIZE = 64
NUM_LAYERS = 2
n=10              # Test the model for n times

def tansig(x,u_min,u_max):
    return u_min + (u_max-u_min)*(torch.tanh(x)+1)/2

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
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
            outs.append(tansig(self.out(r_out[:, time_step, :]),torch.tensor([-0.6,-0.6]),torch.tensor([0.6,0.6])))
        return torch.stack(outs, dim=1), h_state

h = 1
t1 = 7
t2 = 3
T = t1+t2
scale = 100
beta = 100
weight = 1e-4
alpha = 0.95
 
def system(q0,u): 
    # system dynamic
    nb_u = u.shape[1]
    q = np.zeros((2,nb_u+1))
    q[:,0] = q0
    for i in range(nb_u):
        q[:,i+1] = q[:,i] + u[:,i] * h
    return q

def con(args): 
    # constraints 
    # umin<u<umax
    # min<q<qmax

    umax = args
    
    cons = ({'type': 'ineq', 'fun': lambda u: -LA.norm(u.reshape(2,-1),axis=0) + umax})
#              {'type': 'ineq', 'fun': lambda u: -u + umax},\
#             {'type': 'ineq', 'fun': lambda u: system(q0,u) - qmin},\
#                {'type': 'ineq', 'fun': lambda u: -system(q0,u) + qmax})
    return cons

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


def robustness_est(q, Rx, Ry, XA1, XA2, YA1, YA2, XB1, XB2, YB1, YB2):

    
    muA = np.array([1/Rx*(XA1-q[0,:]),1/Rx*(q[0,:]-XA2),1/Ry*(YA1-q[1,:]),1/Ry*(q[1,:]-YA2)])
    muB = np.array([1/Rx*(XB1-q[0,:]),1/Rx*(q[0,:]-XB2),1/Ry*(YB1-q[1,:]),1/Ry*(q[1,:]-YB2)])
    roA = conjunction(muA)
    roB = conjunction(muB)
    g = np.zeros(int(t1/h)+1)
    for i in range(int(t1/h)+1):
        fA = eventually(roA[i:i+int(t2/h)+1])
        fB = eventually(roB[i:i+int(t2/h)+1])
        g[i] = conjunction(np.array([fA,fB]).reshape(2,-1))
    ro = always(g)
    return ro

def qpsolver(args):
    u_ref = args
    def obj(u):
        return (u[0]-u_ref[0])**2 + (u[1]-u_ref[1])**2 #u[0]**2 - 2*u_ref[0]*abs(u[0])*math.cos(u_ref[1]*h-u[1]*h)
    return obj

def qpJacobian(args):
    u_ref = args
    def Jac(u):
        return np.array([2*(u[0]-u_ref[0]), 2*(u[1]-u_ref[1])])
    return Jac

def B(q,o,r):
    return -(q[0]-o[0])**2 - (q[1]-o[1])**2 + r**2

def qpcon(args): 
    # constraints 
    # umin<u<umax
    # min<q<qmax

    q,o1,r1= args
        
    cons = ({'type': 'ineq', 'fun': lambda u: B(system(q,u.reshape(2,1))[:,1],o1,r1) + (alpha-1)*B(q,o1,r1)})
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



rnn = torch.load('rnn_case2.pkl')
rnn.eval()


ro_sum = 0
nb_suc = 0
start = datetime.datetime.now()

Rx = 2
Ry = 2
XA1=1.7; XA2=1.3; YA1=2.7; YA2=2.3
XB1=2.7; XB2=2.3; YB1=1.7; YB2=1.3

for i in range(n):
    q0 = np.array([2.5,1.5]) + 0.4 * (np.random.random(2)-0.5)
    o1 = np.array([2,2])
    r1 = 1

    q = np.zeros((2,int(T/h)+1))
    u = np.zeros((2,int(T/h)))
    
    q[:,0] = q0
    h_state = None
    for j in range(int(T/h)):
        q_tensor = torch.from_numpy(q[:,j]).float().view(1,1,2)
        u_tensor, h_state_n = rnn(q_tensor, h_state)
        u_ref = u_tensor.detach().numpy().reshape(-1)
        h_state = h_state_n
    
        args_cbf = u_ref #reference control
        args_cbf_cons = (q[:,j], o1,r1)
        res_qp = minimize(qpsolver(args_cbf), np.ones(2), method='SLSQP', jac = qpJacobian(args_cbf), constraints=qpcon(args_cbf_cons))        
        u[:,j] = res_qp.x
        

        q[:,j+1] = q[:,j] + u[:,j] * h
        
        
    ro = robustness_est(q, Rx, Ry, XA1, XA2, YA1, YA2, XB1, XB2, YB1, YB2)
    
    ax = plt.gca()
    ax.cla() # clear things for fresh plot
    rectA = patches.Rectangle((XA2,YA2),XA1-XA2,YA1-YA2,edgecolor='none',facecolor='lightblue')
    rectB = patches.Rectangle((XB2,YB2),XB1-XB2,YB1-YB2,edgecolor='none',facecolor='lightblue')
    plt.text(XA2+0.02,YA2+0.3,'RegA',fontsize=12)
    plt.text(XB2+0.1,YB2+0.05,'RegB',fontsize=12)
    ax.add_patch(rectA)
    ax.add_patch(rectB)
    circle1 = plt.Circle(o1, r1, color='g', fill=False)
    ax.add_artist(circle1)
    plt.plot(q[0,:], q[1,:])
    plt.scatter(q[0,:], q[1,:],zorder = 30)
    #plt.plot(q_initial[0,:], q_initial[1,:])
    ax.set_xlim((1, 3))
    ax.set_ylim((1, 3))
    plt.xlabel('x',fontsize=20)
    plt.ylabel('y',fontsize=20)
    plt.annotate('Initial state', xy=q0, xycoords='data', xytext=(-30, 30),
              textcoords='offset points', fontsize=15,arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
    ax.set_aspect(1)
#    plt.savefig('traj1a.png',dpi=300,pad_inches=0.0,bbox_inches='tight')
    plt.show()


    plt.subplot(211)
    plt.plot(np.arange(q.shape[1]),q[0,:],marker='o')
    plt.xlabel('t')
    plt.ylabel('x',fontsize=20)
    plt.subplot(212)
    plt.plot(np.arange(q.shape[1]),q[1,:],marker='o')
    plt.xlabel('time',fontsize=20)
    plt.ylabel('y',fontsize=20)
#    plt.savefig('traj1b.png',dpi=300,pad_inches=0.0,bbox_inches='tight')
    plt.show()
    
    print(ro)
    ro_sum = ro_sum + ro
    if ro > 0:
        nb_suc = nb_suc+1
    

end = datetime.datetime.now()
print('average time:')
print((end-start)/n)
print('success rate:')
print(nb_suc/n)
print('average robustness:')
print(ro_sum/n)

