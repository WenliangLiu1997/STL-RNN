#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:04:52 2020

@author: liuwenliang
"""
from scipy.optimize import minimize 
from scipy.optimize import differential_evolution
from scipy.stats.mstats import gmean
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import datetime
import math
import matplotlib.patches as patches

start = datetime.datetime.now()

h = 1
t1 = 7
t2 = 3
T = t1+t2
scale = 100
beta = 100
n=1100
weight = 1e-6
alpha = 0.95
vmax = 0.6
dist = 0.1

Rx = 2
Ry = 2
XA1=1.7; XA2=1.3; YA1=2.7; YA2=2.3
XB1=2.7; XB2=2.3; YB1=1.7; YB2=1.3
 
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

    q_h, Rx, Ry, XA1, XA2, YA1, YA2, XB1, XB2, YB1, YB2 = args
    def v(u):
        t_c = q_h.shape[1]-1
        q = np.concatenate((q_h[:,:-1], system(q_h[:,-1], u.reshape(2,-1))),axis=1)
        muA = np.array([1/Rx*(XA1-q[0,:]),1/Rx*(q[0,:]-XA2),1/Ry*(YA1-q[1,:]),1/Ry*(q[1,:]-YA2)])
        muB = np.array([1/Rx*(XB1-q[0,:]),1/Rx*(q[0,:]-XB2),1/Ry*(YB1-q[1,:]),1/Ry*(q[1,:]-YB2)])
        roA = conjunction(muA)
        roB = conjunction(muB)
        g = np.empty(0)
        if t_c <= int(t1/h):
            for i in range(min([t_c+1,int(t2/h)])-1):
                fA = eventually(roA[i+max(0,t_c-int(t2/h)+1):i+max(0,t_c-int(t2/h)+1)+int(t2/h)+1])
                fB = eventually(roB[i+max(0,t_c-int(t2/h)+1):i+max(0,t_c-int(t2/h)+1)+int(t2/h)+1])
                g = np.concatenate([g,conjunction(np.array([fA,fB]).reshape(2,-1))])
        else:
            for i in range(int(T/h)-t_c-1):
                fA = eventually(roA[i+max(0,t_c-int(t2/h)+1):i+max(0,t_c-int(t2/h)+1)+int(t2/h)+1])
                fB = eventually(roB[i+max(0,t_c-int(t2/h)+1):i+max(0,t_c-int(t2/h)+1)+int(t2/h)+1])
                g = np.concatenate([g,conjunction(np.array([fA,fB]).reshape(2,-1))])               
                
        return g
    
    cons = ({'type': 'ineq', 'fun': lambda u: v(u)})
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

def robustness(args):
    
    q_h, Rx, Ry, XA1, XA2, YA1, YA2, XB1, XB2, YB1, YB2 = args
    def v(u):
        t_c = q_h.shape[1]-1
        q = np.concatenate((q_h[:,:-1], system(q_h[:,-1], u.reshape(2,-1))),axis=1)
        muA = np.array([1/Rx*(XA1-q[0,:]),1/Rx*(q[0,:]-XA2),1/Ry*(YA1-q[1,:]),1/Ry*(q[1,:]-YA2)])
        muB = np.array([1/Rx*(XB1-q[0,:]),1/Rx*(q[0,:]-XB2),1/Ry*(YB1-q[1,:]),1/Ry*(q[1,:]-YB2)])
        roA = conjunction(muA)
        roB = conjunction(muB)
        if t_c <= int(t1/h):
            i = min([t_c+1,int(t2/h)])-1
            fA = eventually(roA[i+max(0,t_c-int(t2/h)+1):i+max(0,t_c-int(t2/h)+1)+int(t2/h)+1])
            fB = eventually(roB[i+max(0,t_c-int(t2/h)+1):i+max(0,t_c-int(t2/h)+1)+int(t2/h)+1])
            ro = conjunction(np.array([fA,fB]).reshape(2,-1))
        else:
            i = int(T/h)-t_c-1
            fA = eventually(roA[i+max(0,t_c-int(t2/h)+1):i+max(0,t_c-int(t2/h)+1)+int(t2/h)+1])
            fB = eventually(roB[i+max(0,t_c-int(t2/h)+1):i+max(0,t_c-int(t2/h)+1)+int(t2/h)+1])
            ro = conjunction(np.array([fA,fB]).reshape(2,-1))           
                
        return (-ro + weight * np.sum(LA.norm(u.reshape(2,-1),axis=0)))*scale # negative robustness
    return v

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
    return cons

Q = np.empty((0,2,int(T/h)+1))
U = np.empty((0,2,int(T/h)))
nb_suc = 0
bds = np.transpose(np.array([-np.ones(2*int(t2/h))*vmax,np.ones(2*int(t2/h))*vmax]))
sum_ro_s = 0



for i in range(n):
    print(i)
    q0 = np.array([2.5,1.5]) + 0.4 * (np.random.random(2)-0.5)
    q_h = q0.reshape(2,1)

    o1 = np.array([2,2])
    r1 = 1
    u_record = np.empty((2,0))
    for j in range(int(T/h)):
        flag = 0
        for k in range(10):
            if j == 0:
                u0 = 0.5*(np.random.rand(int(t2/h)*2)-0.5)
            args = (q_h, Rx, Ry, XA1, XA2, YA1, YA2, XB1, XB2, YB1, YB2)
            res = minimize(robustness(args), u0, method='SLSQP', bounds=bds, constraints=con(args), options={'maxiter': 500, 'ftol': 1e-12})

            if -res.fun/scale + weight * np.sum(LA.norm(res.x.reshape(2,-1),axis=0)) > 0:
                
                u_record = np.concatenate((u_record,res.x.reshape(2,-1)[:,0].reshape(2,1)),axis=1)
                args_cbf = res.x.reshape(2,-1)[:,0] #reference control
                args_cbf_cons = (q_h[:,-1], o1,r1)
                res_qp = minimize(qpsolver(args_cbf), np.ones(2), method='SLSQP', jac = qpJacobian(args_cbf), constraints=qpcon(args_cbf_cons))
                disturb = dist*(np.random.rand(2,1)-0.5)
                q_h = np.concatenate((q_h,system(q_h[:,-1],res_qp.x.reshape(2,1))[:,-1].reshape(2,1)+disturb),axis=1)
                
                u0 = np.concatenate((res.x.reshape(2,-1)[:,1:], 0.5*(np.random.rand(2,1)-0.5)),axis=1)
            

#                ax = plt.gca()
#                ax.cla() # clear things for fresh plot
#                rectA = patches.Rectangle((XA2,YA2),XA1-XA2,YA1-YA2,edgecolor='none',facecolor='lightblue')
#                rectB = patches.Rectangle((XB2,YB2),XB1-XB2,YB1-YB2,edgecolor='none',facecolor='lightblue')
#                ax.add_patch(rectA)
#                ax.add_patch(rectB)
#                circle1 = plt.Circle(o1, r1, color='g', fill=False)
#                ax.add_artist(circle1)
#                plt.plot(q_h[0,:], q_h[1,:])
#                plt.scatter(q_h[0,:], q_h[1,:],zorder = 30)
#                #plt.plot(q_initial[0,:], q_initial[1,:])
#                ax.set_xlim((1, 3))
#                ax.set_ylim((1, 3))
#                plt.show()
                
                flag = 1
                
                break
            u0 = 0.5*(np.random.rand(int(t2/h)*2)-0.5)
        if flag == 0:
#            print('Failed!')
            break
        if j == int(T/h)-1:
            final_robustness=robustness_est(q_h, Rx, Ry, XA1, XA2, YA1, YA2, XB1, XB2, YB1, YB2)     
            if final_robustness>0:
                nb_suc = nb_suc + 1
                
                sum_ro_s = sum_ro_s + final_robustness       
                Q = np.concatenate((Q,q_h.reshape(1,2,-1)),axis = 0)
                U = np.concatenate((U,u_record.reshape(1,2,-1)),axis=0)
            
#            plt.subplot(211)
#            plt.plot(np.arange(q_h.shape[1]),q_h[0,:],marker='o')
#            plt.subplot(212)
#            plt.plot(np.arange(q_h.shape[1]),q_h[1,:],marker='o')
#            plt.show()
        
    

end = datetime.datetime.now()
print('average time')
print ((end-start)/n)
print('success rate')
print(nb_suc/n)
print('average robust for success:')
print(sum_ro_s/nb_suc)
np.save('Q_case2',Q)
np.save('U_case2',U)

