# -*- coding: utf-8 -*-
"""
Created on Sat May 19 09:52:51 2018

@author: <elaloy elaloy@sckcen.be>

Quasi Newton inversion using generator network from GAN and (pytorch) DL library
for backpropagation of the gradient of the loss w.r.t the latent space in case
the forward model F(G(z)) is not fully differentiable (here because of the thresholding
of G(z)) but an adjoint model providing the gradient of the loss function w.r.t the
original model space is available (here an analytic formulation).
This code is largely inspired by that accompanying the paper by
Alan Richardson (Arxiv, June 2018)
"""
import os, sys
import pickle
import argparse
import numpy as np
import torch
import torch.optim as optim
import random
import time
from scipy.signal import medfilt
import torch.backends.cudnn as cudnn
sys.path.append('./generation')
  
from generator import Generator as Generator

def run_gan_qn(lr, maxiter, gpath, init_z_file,nc,nz,zx,zy,cuda,adam,model_index,noise_index,clip):
    
    # Load initial z (z) if provided
    if init_z_file is None:
        z = torch.rand([1, nz, zx, zy]).to(device)*2-1
    else:
        z = torch.Tensor(init_z_file.reshape(1, nz, zx, zy)).to(device)

    z.requires_grad = True

    # Load true model and measurement data
    model_path = './true_data_model_'+str(model_index)+'_noise_'+str(noise_index)
    with open(model_path+'.pkl', 'rb') as fin:
        tmp=pickle.load(fin)
        #z_true=tmp['z_true']
        model_true=tmp['m_true']
        #d_true=tmp['d_true']
        d=tmp['d']

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
    A = tomokernel_straight_2D(data,x,y) # Distance of ray-segment in each cell for each ray
    A=np.array(A.todense())
    del data
    
    netG = Generator(cuda=True, gpath=gpath).to(device)
    for param in netG.parameters():
        param.requires_grad = False
    netG.eval()

    if adam:
        optimizer = optim.Adam([z], lr=lr)
    else:
        optimizer = optim.LBFGS([z], lr=lr)
    data_cost = []
    model_cost = []
    zs = []
    models = []
    
    filtering=False
    threshold=True

    def closure():
        # Produce model from z
        
        # clipping
        if clip == 'standard':
            z.data[z.data> 1] = 1
            z.data[z.data < -1] = -1
        if clip == 'stochastic':
            z.data[z.data > 1] = random.uniform(-1, 1)
            z.data[z.data < -1] = random.uniform(-1, 1)
        
        x0 = netG(z)
        
        # --- Here we quit pytorch ---
        x=x0.data.cpu().numpy() # copy x into cpu and numpy for further calculations

        zs.append(z.data.cpu().numpy().copy())

        # Compute cost and gradient on model
        s_model=x[0,0,2:127,3:63]
        s_model= (s_model + 1) * 0.5  # Convert from [-1,1] to [0,1]
        
        if filtering:
            s_model = medfilt(s_model, kernel_size=(3, 3))


        s_model[s_model < 0.5] = 0
        s_model[s_model >= 0.5] = 1

        s_model[s_model==0]=0.08 # m/ns
        s_model[s_model==1]=0.06 # m/ns
        s_model=1/s_model

        sim=A@s_model.flatten(order='F')
        e=d-sim

        cost = np.sum(np.power(e, 2))
       
        grad = -2*A.T@e.T
        # change from Fortran ordering used by forward solver to C ordering used by numpy
        grad=grad.reshape((125,60),order='F').reshape((7500))
        
        # embed grad within array of the size used by the GAN
        grad_fs=np.zeros((129,65))
        grad_fs[2:127,3:63]=grad.reshape((125,60))
        
        data_cost.append(cost)
        model_cost.append(np.linalg.norm(model_true - x[0,0,2:127,3:63]))

        # Update z (z) using gradient information 
        #--- Here we go back to pytorch ---
        optimizer.zero_grad()
        x0.backward(torch.Tensor(grad_fs.reshape(x0.shape)).to(device))
        return cost

    for it in range(maxiter):

        optimizer.step(closure)

        print(it, data_cost[-1], model_cost[-1])
        
        # Stop non-productive runs
        if it >= 100 and data_cost[-1] > 1e3:
            break

    x = netG(z)

    return z.data.cpu().numpy(), x.data.cpu().numpy(), data_cost, model_cost, zs, models

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--niter', type=int, default=10000,
                        help='number of iterations of optimizer')
    parser.add_argument('--clip', default='stochastic',
                        help='disabled|standard|stochastic')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate of optimizer')   
    parser.add_argument('--gpath', type=str, default='./netG.pth',
                        help='path to saved generator weights to use')
    parser.add_argument('--init_z_file', type=str, default=None,
                        help='path to initial latent vector to use')
#    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--nz', type=int, default=1, help='number of non-spatial dimensions in latent space z')
    parser.add_argument('--zx', type=int, default=5, help='number of grid elements in vertical spatial dimension of z')
    parser.add_argument('--zy', type=int, default=3, help='number of grid elements in horizontal spatial dimension of z')
    parser.add_argument('--nc', type=int, default=1, help='number of channels in original image space')
    parser.add_argument('--model_index', type=int, default=1, help='index to select true model')
    parser.add_argument('--noise_index', type=int, default=1, help='index to select noise realization used to corrupt the true data')
    parser.add_argument('--Seed', type=int, default=2467,help='manual seed')
    opt = parser.parse_args()

    home_dir='D:/gan_for_gradient_based_inv'
    opt.gpath=home_dir+'/inversion/generation/netG_epoch_36.pth'
    opt.adam=True
    print(opt)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if str(device) == 'cuda':
        opt.cuda=True
    else:
        opt.cuda=False

    if opt.Seed is None:
        opt.Seed = random.randint(1, 10000)
    print("Random Seed: ", opt.Seed)
    random.seed(opt.Seed)
    torch.manual_seed(opt.Seed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.Seed)
        cudnn.benchmark = True
    
    main_dir=home_dir+'/inversion'
    addstr='/data_QN_clip_'+str(opt.clip)
    save_data_dir=main_dir+addstr
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)
    ntrials=100
    best_cost_hist=np.zeros((ntrials))+np.nan
    best_z_hist=np.zeros((ntrials,opt.nz,opt.zx,opt.zy))+np.nan
    best_iter_hist=np.zeros((ntrials))
    t0=time.time()
    for j in range(0,ntrials):
        print('new run')
        z, final_model, data_cost, model_cost, zs, models = \
                run_gan_qn(opt.lr, opt.niter, opt.gpath,
                            opt.init_z_file, opt.nc,opt.nz,opt.zx,opt.zy,
                            opt.cuda, opt.adam,opt.model_index,opt.noise_index,opt.clip)
        with open(save_data_dir+'/sgan_qn_inv_trial_'+str(j)+'_model'+str(opt.model_index)+'_noise'+str(opt.noise_index)+'_lr_'+str(opt.lr)+'_init_rn_seed_'+str(opt.Seed)+'.pkl', 'wb') as fout:
            pickle.dump({'final_z':z,'final_model':final_model,'data_cost':data_cost,
                         'model_cost':model_cost,'zhist':zs}, fout, protocol=-1)
    
        ii=np.where(np.array(data_cost)==np.min(np.array(data_cost)))[0][0]
        best_iter_hist[j]=ii
        best_cost_hist[j]=data_cost[ii]
        best_z_hist[j,:]=np.array(zs)[ii,0,:]
    
    print(time.time()-t0)        
    with open(save_data_dir+'/sgan_qn_res_over_'+str(ntrials)+'_trials_model'+str(opt.model_index)+'_noise'+str(opt.noise_index)+'_lr_'+str(opt.lr)+'_init_rn_seed_'+str(opt.Seed)+'.pkl', 'wb') as fout:
        pickle.dump({'best_cost_hist':best_cost_hist,'best_z_hist':best_z_hist,'best_iter_hist':best_iter_hist,'seed':opt.Seed}, fout, protocol=-1)
    