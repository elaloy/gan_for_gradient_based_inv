# -*- coding: utf-8 -*-
"""
Created on Sat May 19 09:52:51 2018

@author: <elaloy elaloy@sckcen.be>

Gauss Newton inversion within the latent space of a generator network from GAN 
and finite-differencing to approximate the Jacobian.

"""
#%%
import os
import time
import numpy as np
import sys 
import torch
import random
import pickle
import scipy
import matplotlib.pyplot as plt
import argparse
import torch.backends.cudnn as cudnn
from scipy.signal import medfilt
sys.path.append('./generation')
from generator import Generator as Generator

def comp_res_J(z,obs,G,zx,zy,nz,netG,Prior='Normal',alfa=None,mv=None,CalcJ=True,cuda=True,Regularization=None,threshold=True,filtering=False,delta_z=0.25):
    #% Function to get residuals and Jacobian
    #Regularization is None
    #Prior is 'Normal' 
    
    if Prior=='Normal':
        zs=scipy.stats.norm.cdf(z, 0, 1)*2-1 # from standard normal to uniform
    else:
        zs=np.copy(z)
    zs=zs.reshape((1,nz,zx,zy))
    zs = torch.from_numpy(zs).float()
    if cuda:
        zs = zs.cuda()
    m_current = netG(zs).cpu().numpy()
    m = m_current[0,0,2:127,3:63]
    m = (m + 1) * 0.5  # Convert from [-1,1] to [0,1]
    if filtering:
        m = medfilt(m, kernel_size=(3, 3))

    if threshold:
        m[m < 0.5] = 0
        m[m >= 0.5] = 1

        m[m==0]=0.08 # m/ns
        m[m==1]=0.06 # m/ns
        
    else:
        m = 1 - m
        m= 0.06 + m*0.02

    if Regularization=='contrast':
        m=alfa*m+(1-alfa)*mv
    elif Regularization=='smearing':
        m=scipy.ndimage.filters.gaussian_filter(m,alfa)
    s=1/m # from velocity field to slowness field

    sim=G@s.flatten(order='F')
   
    e=obs-sim

    if CalcJ==True:
        JacApprox='3pts'
        sim_ref=sim
        J=np.zeros((obs.shape[0],zx*zy))
        for i in range(0,J.shape[1]):   
            z_p1=np.copy(z).flatten()
            z_p1[i]=z_p1[i]+delta_z
            if Prior=='Normal':
                zs_p1=scipy.stats.norm.cdf(z_p1, 0, 1)*2-1
            else:
                zs_p1=z_p1
            zs_p1=zs_p1.reshape((1,nz,zx,zy))
            zs_p1 = torch.from_numpy(zs_p1).float()
            if cuda:
                zs_p1 = zs_p1.cuda()
                
            m = netG(zs_p1).cpu().numpy()
            m = m[0,0,2:127,3:63]
            m = (m + 1) * 0.5  # Convert from [-1,1] to [0,1]
            if filtering:
                m = medfilt(m, kernel_size=(3, 3))
        
            if threshold:
                m[m < 0.5] = 0
                m[m >= 0.5] = 1
        
                m[m==0]=0.08 # m/ns
                m[m==1]=0.06 # m/ns
                
            else:
                m = 1 - m
                m= 0.06 + m*0.02
        
            if Regularization=='contrast':
                m=alfa*m+(1-alfa)*mv
            elif Regularization=='smearing':
                m=scipy.ndimage.filters.gaussian_filter(m,alfa)
            s=1/m # from velocity field to slowness field
            
            sim_p1=G@s.flatten(order='F')
            if JacApprox=='2pts':
                J[:,i]=(sim_p1-sim_ref)/delta_z
            if JacApprox=='3pts':
                z_p2=np.copy(z).flatten()
                z_p2[i]=z_p2[i]-delta_z
                if Prior=='Normal':
                    zs_p2=scipy.stats.norm.cdf(z_p2, 0, 1)*2-1
                else:
                    zs_p2=z_p2
                zs_p2=zs_p2.reshape((1,nz,zx,zy))
                zs_p2 = torch.from_numpy(zs_p2).float()
                if cuda:
                    zs_p2 = zs_p2.cuda()
                m = netG(zs_p2).cpu().numpy()
                m = m[0,0,2:127,3:63]
                m = (m + 1) * 0.5  # Convert from [-1,1] to [0,1]
                if filtering:
                    m = medfilt(m, kernel_size=(3, 3))
            
                if threshold:
                    m[m < 0.5] = 0
                    m[m >= 0.5] = 1
            
                    m[m==0]=0.08 # m/ns
                    m[m==1]=0.06 # m/ns
                    
                else:
                    m = 1 - m
                    m= 0.06 + m*0.02
            
                if Regularization=='contrast':
                    m=alfa*m+(1-alfa)*mv
                elif Regularization=='smearing':
                    m=scipy.ndimage.filters.gaussian_filter(m,alfa)
                s=1/m # from velocity field to slowness field
                sim_p2=G@s.flatten(order='F')
                J[:,i]=(sim_p1-sim_p2)/(2*delta_z)      
    else:
        J=None  
    return e,J,m_current

def run_inv_gn(niter, gpath,nc,nz,zx,zy,cuda,model_index,noise_index,threshold,filtering,
               FDCalcJ,invCe,maxit,it_stop,rmse_stop,Prior,D,delta_z,labda,labda_max,
               labda_min,labdaUpdate,VaryAlfa,AdaptJump,Regularization,mv,test_type,
               alfa_min,alfa_f):
    
    # Load true model and measurement data
    model_path = './true_model_'+str(model_index)+'_noise_'+str(noise_index)
    with open(model_path+'.pkl', 'rb') as fin:
        tmp=pickle.load(fin)
        
        if threshold:
            #z_true=tmp['z_true']
            model_true=tmp['m_true']
            d=tmp['d']
        else:
            model_true=tmp['m_true_cont']#125 x 60 in [0,1]
            d=tmp['d_cont']
       

    # forward setup
    from tomokernel_straight import tomokernel_straight_2D
    
    nx=60 # Here x is the horizontal axis (number of columns) and not the number of rows
    ny = 125 # Here y is the vertical axis (number of rows) and not the number of columns

    # The x-axis is varying the fastest 
    x = np.arange(0,(nx/10)+0.1,0.1)                      
    y = np.arange(0,(ny/10)+0.1,0.1) 
    sourcex = 0.01
    sourcez = np.arange(0.5,ny/10,0.5)                         
    receiverx = nx/10-0.01
    receiverz = np.arange(0.5,ny/10,0.5)   
    nsource = len(sourcez); nreceiver = len(receiverz)
    ndata=nsource*nreceiver
    data=np.zeros((ndata,4))
    # Calculate acquisition geometry (multiple-offset gather)
    for jj in range(0,nsource):
        for ii in range(0,nreceiver):
            data[ ( jj ) * nreceiver + ii , :] = np.array([sourcex, sourcez[jj], receiverx, receiverz[ii]])
    # Calculate forward modeling kernel (from Matlab code by Dr. James Irving, UNIL)
    G = tomokernel_straight_2D(data,x,y) # Distance of ray-segment in each cell for each ray
    G=np.array(G.todense())
    del data
    
    netG = Generator(cuda=cuda, gpath=gpath)
    for param in netG.parameters():
        param.requires_grad = False
    netG.eval()
    if cuda:
        netG.cuda()
    
    z_hist=np.zeros((maxit,zx*zy))+np.nan
    labda_hist=np.zeros((maxit))+np.nan
    rmse_hist=np.zeros((maxit))+np.nan
    e_hist=np.zeros((maxit,ndata))+np.nan
    improv_hist=np.zeros((maxit))+np.nan
    alfa_hist=np.zeros((maxit))+np.nan
    improv_hist[0]=1
    best_rmse=1000
    alfa=np.copy(alfa_min)
    
    z0=np.random.randn(zx*zy)
    z=np.copy(z0)
    iter_hist=np.nan

    istart=0
    iend=maxit
        
    for i in range(istart,iend):

        z_old = z
        e,J,m_current = comp_res_J(z,d,G,zx,zy,nz,netG,Prior,1/alfa,mv=None,CalcJ=FDCalcJ,cuda=cuda,Regularization=Regularization,threshold=threshold,filtering=filtering)
        
        rmse=np.sqrt(np.sum(e**2)/len(e))

        # Different ways of updating labda if tried
        if i > 0 and labdaUpdate=='alternate':
            if np.mod(i,2)==0:
                labda=100
            else:
                labda=1
             
        if i > 0 and labdaUpdate=='constant_SteepDesc':
            labda=np.minimum(labda*1.1,labda_max)
                
        if i > 0 and labdaUpdate=='constant_GN':
            labda=np.maximum(labda*0.9,labda_min)
                
        if i > 9 and labdaUpdate=='dynamic':            
            if rmse < rmse_hist[i-1]: # Decrease labda to get a more GN update
                labda=labda=np.maximum(labda*0.5,labda_min)
            elif rmse > rmse_hist[i-1]: # Increase labda to get a more steepest descent update
                labda=np.minimum(labda*2,labda_max)
                
        print('Current RMSE is ',rmse)
        if rmse < best_rmse:
            best_rmse=rmse
        
        # Store z, rmse and labda
        z_hist[i,:]=z.flatten()
        rmse_hist[i]=rmse
        alfa_hist[i]=alfa
        labda_hist[i]=labda
        e_hist[i]=e
        if i > 0 and (rmse>best_rmse):
            improv_hist[i]=0
        else:
            improv_hist[i]=1

        # Update z
        dhat=e+J@z
        A = J.T@invCe@J + labda*D@D.T
        z_new = np.linalg.inv(A)@J.T@invCe@dhat
        
        # Update alfa if regularization by vanishing smearing or gradual contrasting of the models is tried 
        if VaryAlfa==True:
            if np.mod(i,1)==0:
                alfa=np.minimum(np.maximum(alfa_min,alfa)*alfa_f,np.inf)
            print(alfa)
            alfa_hist[i]=alfa

        if i >= it_stop and best_rmse > rmse_stop:
            iter_hist=i
            print('Stop non-productive run')
            break   
        # Try to reduce the jump if the fit is not improving after some given iterations
        if i >= 20 and AdaptJump==True and np.sum(improv_hist[i-5:i])==0:
            beta = 0.5
            print('reduce jump')
        else:
            beta=1

        z = z_old + beta*(z_new - z_old)
        
        print('iteration ',str(i),' done - best RMSE = ',str(best_rmse))

    return best_rmse, rmse, z_hist, rmse_hist, labda_hist,e_hist,improv_hist,z0,iter_hist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--niter', type=int, default=10000,
                        help='number of iterations of the Gauss-Newton search')
    parser.add_argument('--gpath', type=str, default='./netG.pth',
                        help='path to saved generator weights to use')
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--nz', type=int, default=1, help='number of non-spatial dimensions in latent space z')
    parser.add_argument('--zx', type=int, default=5, help='number of grid elements in vertical spatial dimension of z')
    parser.add_argument('--zy', type=int, default=3, help='number of grid elements in horizontal spatial dimension of z')
    parser.add_argument('--nc', type=int, default=1, help='number of channels in original image space')
    parser.add_argument('--model_index', type=int, default=1, help='index to select true model')
    parser.add_argument('--delta_z', type=float, default=0.1,help='delta for finite-difference jacobian approximation')   
    parser.add_argument('--threshold', action='store_true', help='use a binary true model and create binary model proposals')
    parser.add_argument('--noise_index', type=int, default=1, help='index to select noise realization used to corrupt the true data')
    parser.add_argument('--Seed', type=int, default=2467,help='manual seed')
    opt = parser.parse_args()


    home_dir='D:/gan_for_gradient_based_inv'
    opt.gpath=home_dir+'/inversion/generation/netG_epoch_36.pth'
    opt.filtering=False # always False
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run "
              "with --cuda")

    if opt.Seed is None:
        opt.Seed = random.randint(1, 10000)
    print("Random Seed: ", opt.Seed)
    random.seed(opt.Seed)
    torch.manual_seed(opt.Seed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.Seed)
        cudnn.benchmark = True
    
    main_dir=home_dir+'/inversion'
    if opt.threshold:
        addstr='/data_GN_thres'+'_iter_'+str(opt.niter)
    else:
        addstr='/data_GN_cont'+'_iter_'+str(opt.niter)
    save_data_dir=main_dir+addstr
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)
    ntrials=100
    best_cost_hist=np.zeros((ntrials))+np.nan
    best_z_hist=np.zeros((ntrials,opt.nz,opt.zx,opt.zy))+np.nan
    best_iter_hist=np.zeros((ntrials))
    best_alfa_hist=np.zeros((ntrials))+np.nan
    final_cost_hist=np.zeros((ntrials))+np.nan
    best_z_hist=np.zeros((ntrials,opt.zx*opt.zy))+np.nan
    final_z_hist=np.zeros((ntrials,opt.zx*opt.zy))+np.nan
    iter_hist=np.zeros((ntrials))+opt.niter
    z0_hist=np.zeros((ntrials,opt.zx*opt.zy))+np.nan
    FDCalcJ=True # True: 2-pt finite-difference approximation of J
                 # False: Jacobian computed by autodifferentiation with pytorch - not implemented herein
    SaveTrial=False
    
    # Gauss Newton (GN) additional inversion settings
    # With the settings below we have the classical GN search described in the paper
    # For completeness, I have added the many variants we have tested and that did not work (fpr some, possibly because they don't make sense)
    Ce=np.eye(576)
    invCe=np.linalg.inv(Ce)
    maxit=opt.niter
    it_stop=maxit+1 # With a value larger than maxit, this is not used
    rmse_stop=2 # Since it_stop > maxit, this is not used
    Prior='Normal'
    D=np.eye(opt.zx*opt.zy)
    #opt.delta_z is the perturbation factor for the finite difference approximation of the Jacobian
    labda=1
    labda_max=100
    labda_min=0.01
    labdaUpdate='constant' 
    VaryAlfa=False
    AdaptJump=False
    Regularization=None
    mv=0.7*0.08+0.3*0.06
    if VaryAlfa==True:
        test_type='smearing' #'smearing' or 'contrast'
        alfa_min=0.5
    else:
        test_type='classical'
        alfa_min=1e6
    alfa_f=1.05
    
    
    t0=time.time()
    for j in range(0,ntrials):
        print('new run')
        best_rmse, rmse, z_hist, rmse_hist, labda_hist,e_hist,improv_hist, z0, iter_stop = \
                run_inv_gn(opt.niter, opt.gpath,opt.nc,opt.nz,opt.zx,opt.zy,
                            opt.cuda,opt.model_index,opt.noise_index,opt.threshold,opt.filtering,
                            FDCalcJ,invCe,maxit,it_stop,rmse_stop,Prior,D,opt.delta_z,
                            labda,labda_max,labda_min,labdaUpdate,VaryAlfa,AdaptJump,
                            Regularization,mv,test_type,alfa_min,alfa_f)
                
        print('Trial: ',str(j), ' Best RMSE is: ',str(best_rmse))
        best_cost_hist[j]=best_rmse
        final_cost_hist[j]=rmse
        final_z_hist[j,:]=z_hist[-1,:]
        ii=np.where(rmse_hist==np.min(rmse_hist[0:opt.niter]))[0][0]
        best_z_hist[j,:]=z_hist[ii,:]
        z0_hist[j,:]=z0
        iter_hist[j]=iter_stop
        if SaveTrial:
            with open(save_data_dir+'/sgan_gn_inv_trial_'+str(j)+'_model'+str(opt.model_index)+'_noise'+str(opt.noise_index)+'_delta_z_'+str(opt.delta_z)+'_threshold_'+str(opt.threshold)+'_init_rn_seed_'+str(opt.Seed)+'.pkl', 'wb') as fout:
                pickle.dump({'z_hist':z_hist,'rmse_hist':rmse_hist,'labda_hist':labda_hist,
                         'e_hist':e_hist,'improv_hist':improv_hist}, fout, protocol=-1)
    
    print(time.time()-t0)        
    with open(save_data_dir+'/sgan_gn_res_over_'+str(ntrials)+'_trials_'+test_type+'_model'+str(opt.model_index)+'_noise'+str(opt.noise_index)+'_delta_z_'+str(opt.delta_z)+'_threshold_'+str(opt.threshold)+'_init_rn_seed_'+str(opt.Seed)+'.pkl', 'wb') as fout:
        pickle.dump({'best_cost_hist':best_cost_hist,'final_cost_hist':final_cost_hist,
                     'best_z_hist':best_z_hist,'final_z_hist':final_z_hist,'z0_hist':z0_hist}, fout, protocol=-1)




