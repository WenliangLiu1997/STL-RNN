#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:04:52 2020

@author: liuwenliang
"""
from scipy.optimize import minimize 
from scipy.stats.mstats import gmean
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import datetime
import random
start = datetime.datetime.now()
 
h=0.5
scale = 2000
scale_w = 3
l = 0.01
alpha = 0.7
n=1
weight = 0#1e-6

def system(q0,u): 
    # system dynamic
    nb_u = u.shape[1]
    q = np.zeros((3,nb_u+1))
    q[:,0] = q0
    for i in range(nb_u):
        q[0,i+1]=q[0,i]+scale_w*u[0,i]/u[1,i]*(math.sin(q[2,i]+u[1,i]/scale_w*h)-math.sin(q[2,i]))
        q[1,i+1]=q[1,i]+scale_w*u[0,i]/u[1,i]*(math.cos(q[2,i])-math.cos(q[2,i]+u[1,i]/scale_w*h))
        q[2,i+1]=q[2,i]+u[1,i]/scale_w*h
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
        ro = conjunction(roTotal.reshape(-1,1))

        return (-ro + weight * np.sum(LA.norm(u.reshape(2,-1),axis=0)))*scale
    return v

def qpsolver(args):
    u_ref = args
    def obj(u):
        return (u[0]-u_ref[0])**2 + l*(u[1]-u_ref[1])**2 
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

    q,o1,r1, o2, r2, o3,r3,o4,r4 = args
        
    cons = ({'type': 'ineq', 'fun': lambda u: B(system(q,u.reshape(2,1))[:,1],o1,r1) + (alpha-1)*B(q,o1,r1)},\
            {'type': 'ineq', 'fun': lambda u: B(system(q,u.reshape(2,1))[:,1],o2,r2) + (alpha-1)*B(q,o2,r2)},\
            {'type': 'ineq', 'fun': lambda u: B(system(q,u.reshape(2,1))[:,1],o3,r3) + (alpha-1)*B(q,o3,r3)},\
            {'type': 'ineq', 'fun': lambda u: B(system(q,u.reshape(2,1))[:,1],o4,r4) + (alpha-1)*B(q,o4,r4)})
#             {'type': 'ineq', 'fun': lambda u: ((q[0]-o3[0])*math.cos(q[2])+(q[1]-o3[1])*math.sin(q[2]))*u[0] + 1*B(q,o3,r3)})
#                {'type': 'ineq', 'fun': lambda u: -system(q0,u) + qmax})
    return cons
    

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




ro = np.zeros(n)
sum_ro_s = 0
nb_success = 0

Q = np.empty((0,3,int(T/h)+1))
U = np.empty((0,2,int(T/h)))

start_0 = datetime.datetime.now()
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
#    print(theta0)
    q0 = np.array([x0,y0,theta0]).reshape(-1)
    u_h = np.empty((2,0))
    u_record = np.empty((2,0))
    for k in range(int(T/h)):
#        print(k)
        flag = 0
        for j in range(5):
            start = datetime.datetime.now()
            if k==0:
                u0 = 0.5*(np.random.rand((int(T/h))*2)-0.5)  
            bds = np.transpose(np.concatenate((2*np.array([-np.zeros(int(T/h)-k),np.ones(int(T/h)-k)]),1*scale_w*np.array([-np.ones(int(T/h)-k),np.ones(int(T/h)-k)])),axis=1))
            args = (u_h, q0, Rx, Ry, t1, t2, T, XA1, XA2, YA1, YA2, XB1, XB2, YB1, YB2, XC1, XC2, YC1, YC2, X_obs1, X_obs2, Y_obs1, Y_obs2, XM1, XM2, YM1, YM2)
            res = minimize(robustness(args), u0, method='SLSQP', bounds = bds, options={'maxiter': 500,'ftol': 1e-8})
            end = datetime.datetime.now()
            if -res.fun/scale + weight * np.sum(LA.norm(res.x.reshape(2,-1),axis=0))>0:
#                print(res.message)
                u_record = np.concatenate((u_record,res.x.reshape(2,-1)[:,0].reshape(2,1)),axis=1)
                args_cbf = res.x.reshape(2,-1)[:,0] #reference control
                args_cbf_cons = (system(q0, u_h)[:,-1], o1,r1,o2,r2,o3,r3,o4,r4)
                res_qp = minimize(qpsolver(args_cbf), np.ones(2), method='SLSQP', jac = qpJacobian(args_cbf), constraints=con(args_cbf_cons))
#                print(res_qp.success)
                u0 = res.x.reshape(2,-1)[:,1:].reshape(-1)# + 0.1*(np.random.rand((int(T/h)-k-1)*2)-0.5)
                u_h = np.concatenate((u_h,res_qp.x.reshape(2,1)),axis=1)
#                u_h = np.concatenate((u_h,res.x.reshape(2,-1)[:,0].reshape(2,1)),axis=1)
                
                q = system(q0, u_h)
                
                
                #plot figure at each time step
                # ax = plt.gca()
                # ax.cla() # clear things for fresh plot
                # rectA = patches.Rectangle((XA2,YA2),XA1-XA2,YA1-YA2,edgecolor='none',facecolor='lightblue')
                # rectB = patches.Rectangle((XB2,YB2),XB1-XB2,YB1-YB2,edgecolor='none',facecolor='lightblue')
                # rectC = patches.Rectangle((XC2,YC2),XC1-XC2,YC1-YC2,edgecolor='none',facecolor='y')
                # rectObs = patches.Rectangle((X_obs1,Y_obs1),X_obs2-X_obs1,Y_obs2-Y_obs1,edgecolor='none',facecolor='r')
                # ax.add_patch(rectA)
                # ax.add_patch(rectB)
                # ax.add_patch(rectC)
                # ax.add_patch(rectObs)
                # circle1 = plt.Circle(o1, r1, color='r', fill=False)
                # ax.add_artist(circle1)
                # circle2 = plt.Circle(o2, r2, color='r', fill=False)
                # ax.add_artist(circle2)
                # circle3 = plt.Circle(o3, r3, color='r', fill=False)
                # ax.add_artist(circle3)
                # circle4 = plt.Circle(o4, r4, color='r', fill=False)
                # ax.add_artist(circle4)
                # plt.plot(q[0,:], q[1,:])
                # plt.scatter(q[0,:], q[1,:],alpha=0.6, zorder = 30)
                # ax.set_xlim((0, 10))
                # ax.set_ylim((0, 10))
                # ax.set_aspect(1)
                # plt.show()
                
                
                
                flag = 1
                
                
                break
            u0 = 0.5*(np.random.rand((int(T/h)-k)*2)-0.5)
        if flag == 0:
            break
        if k == int(T/h)-1:
            nb_success = nb_success + 1
            
            args = (np.empty((2,0)), q0, Rx, Ry, t1, t2, T, XA1, XA2, YA1, YA2, XB1, XB2, YB1, YB2, XC1, XC2, YC1, YC2, X_obs1, X_obs2, Y_obs1, Y_obs2, XM1, XM2, YM1, YM2)
            sum_ro_s = sum_ro_s - robustness(args)(u_h.reshape(-1))/scale + weight * np.sum(LA.norm(u_h,axis=0))     
            Q = np.concatenate((Q,q.reshape(1,3,-1)),axis = 0)
            U = np.concatenate((U,u_record.reshape(1,2,-1)),axis=0)
            
            # plot figure
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
            plt.show()
        
end_0 = datetime.datetime.now()
print('success rate:')
print(nb_success/n)
print('average time:')
print((end_0-start_0)/n)
#print('average time for success:')
#print(t_sum_s/nb_success)
#print('average robust:')
#print(np.mean(ro))
print('average robust for success:')
print(sum_ro_s/nb_success)


# np.save('Q_case1',Q)
# np.save('U_case1',U)


        
    
