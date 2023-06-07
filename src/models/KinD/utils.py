import torch
import torch.nn as nn
import  torch.nn.functional as F
import pytorch_ssim
from torch.autograd import Variable
import numpy as np
from math import exp



class KinDLoss_decom(nn.Module):
    def __init__(self):
        super().__init__()
       

    def forward(self, reflect_1, reflect_2, illumin_1, illumin_2, image, target, device="cuda"):
        I_low_3 = torch.cat([illumin_1, illumin_1, illumin_1], dim=1)
        I_high_3 = torch.cat([illumin_2, illumin_2, illumin_2], dim=1)
        image = image.transpose(1,3)
        target = target.transpose(1,3)


        recon_loss_low = torch.mean(torch.abs(reflect_1 * I_low_3 -  image))
        recon_loss_high = torch.mean(torch.abs(reflect_2 * I_high_3 - target))

        equal_R_loss = torch.mean(torch.abs(reflect_1 - reflect_2))

        i_mutual_loss = self.mutual_i_loss(illumin_1, illumin_2, device)

        i_input_mutual_loss_high = self.mutual_i_input_loss(illumin_2, target, device)
        i_input_mutual_loss_low = self.mutual_i_input_loss(illumin_1, image, device)

        loss_Decom = 1*recon_loss_high + 1*recon_loss_low \
                    + 0.01 * equal_R_loss + 0.2*i_mutual_loss \
                    + 0.15* i_input_mutual_loss_high + 0.15* i_input_mutual_loss_low
        
        return loss_Decom

    
    def gradient(self,input_tensor, direction, device="cuda"):

        a = input_tensor.shape[0]
    
        b = torch.zeros(input_tensor.shape[2],1)
        b = torch.zeros(input_tensor.shape[0],input_tensor.shape[1],input_tensor.shape[2],1)
        b = b.to(device)

        input_tensor = torch.cat((input_tensor,b),3)
    
        a = torch.zeros(1, input_tensor.shape[3])
        a = torch.zeros(input_tensor.shape[0],input_tensor.shape[1],1,input_tensor.shape[3])
        a = a.to(device)
        
        input_tensor = torch.cat((input_tensor,a), 2)
    
        c = [[0, 0], [-1, 1]]
        c = torch.FloatTensor(c)
        c = c.to(device)

        smooth_kernel_x = torch.reshape(c,(1,1,2,2))
        smooth_kernel_y = smooth_kernel_x.permute( [0, 1,3,2])

        if direction == "x":
            kernel = smooth_kernel_x
        elif direction == "y":
            kernel = smooth_kernel_y
        weight = nn.Parameter(data=kernel, requires_grad=False)
        gradient_orig = torch.abs(F.conv2d(input_tensor, weight, stride=1,padding=0))
        grad_min = torch.min(gradient_orig)
        grad_max = torch.max(gradient_orig)
        grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
        return grad_norm
    
    def mutual_i_input_loss(self, input_I_low, input_im, device="cuda"):
        
        input_gray = 0.299*input_im[:,0,:,:] + 0.587*input_im[:,1,:,:] + 0.114*input_im[:,2,:,:]
        input_gray = input_gray.unsqueeze(dim=1)
        low_gradient_x = self.gradient(input_I_low, "x", device)
        input_gradient_x = self.gradient(input_gray, "x", device)
        epsilon = 0.01*torch.ones_like(input_gradient_x)
        x_loss = torch.abs(torch.div(low_gradient_x, torch.max(input_gradient_x, epsilon)))
        low_gradient_y = self.gradient(input_I_low, "y", device)
        input_gradient_y = self.gradient(input_gray, "y", device)
        y_loss = torch.abs(torch.div(low_gradient_y, torch.max(input_gradient_y, epsilon)))
        mut_loss = torch.mean(x_loss + y_loss) 
        return mut_loss
    
    def mutual_i_loss(self, input_I_low, input_I_high, device="cuda"):
        low_gradient_x = self.gradient(input_I_low, "x", device)
        high_gradient_x = self.gradient(input_I_high, "x", device)
        x_loss = (low_gradient_x + high_gradient_x)* torch.exp(-10*(low_gradient_x+high_gradient_x))
        low_gradient_y = self.gradient(input_I_low, "y", device)
        high_gradient_y = self.gradient(input_I_high, "y", device)
        y_loss = (low_gradient_y + high_gradient_y) * torch.exp(-10*(low_gradient_y+high_gradient_y))
        mutual_loss = torch.mean( x_loss + y_loss) 
        return mutual_loss
    


class KinDLoss_restore(nn.Module):
    def __init__(self):
        super().__init__()

    def gaussian(self,window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self,window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
        return window

    def _ssim(self,img1, img2, window, window_size, channel, size_average = True):

        
        mu1 = F.conv2d(img1, window)#, groups = channel)
        mu2 = F.conv2d(img2, window)#, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2


        sigma1_sq = F.conv2d(img1*img1, window) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)



    def ssim(self,img1, img2, window_size = 11, size_average = True):
        (_, channel, _, _) = img1.size()
        window = self.create_window(window_size, channel)
        return self._ssim(img1, img2, window, window_size, channel, size_average)


    def gradient(self,input_tensor, direction, device="cuda"):

        a = input_tensor.shape[0]
    
        b = torch.zeros(input_tensor.shape[2],1)
        b = torch.zeros(input_tensor.shape[0],input_tensor.shape[1],input_tensor.shape[2],1)
        b = b.to(device)

        input_tensor = torch.cat((input_tensor,b),3)
    
        a = torch.zeros(1, input_tensor.shape[3])
        a = torch.zeros(input_tensor.shape[0],input_tensor.shape[1],1,input_tensor.shape[3])
        a = a.to(device)
        
        input_tensor = torch.cat((input_tensor,a), 2)
    
        c = [[0, 0], [-1, 1]]
        c = torch.FloatTensor(c)
        c = c.to(device)

        smooth_kernel_x = torch.reshape(c,(1,1,2,2))
        smooth_kernel_y = smooth_kernel_x.permute( [0, 1,3,2])

        if direction == "x":
            kernel = smooth_kernel_x
        elif direction == "y":
            kernel = smooth_kernel_y
        weight = nn.Parameter(data=kernel, requires_grad=False)
        gradient_orig = torch.abs(F.conv2d(input_tensor, weight, stride=1,padding=0))
        grad_min = torch.min(gradient_orig)
        grad_max = torch.max(gradient_orig)
        grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
        return grad_norm
    
    def grad_loss(self,input_r_low, input_r_high, device="cuda"):
        input_r_low_gray = 0.299*input_r_low[:,0,:,:] + 0.587*input_r_low[:,1,:,:] + 0.114*input_r_low[:,2,:,:]
        input_r_low_gray = input_r_low_gray.unsqueeze(dim=1)
        input_r_high_gray = 0.299*input_r_high[:,0,:,:] + 0.587*input_r_high[:,1,:,:] + 0.114*input_r_high[:,2,:,:]
        input_r_high_gray = input_r_high_gray.unsqueeze(dim=1)
        
        x_loss = torch.square(self.gradient(input_r_low_gray, 'x', device) - self.gradient(input_r_high_gray, 'x', device))
        y_loss = torch.square(self.gradient(input_r_low_gray, 'y', device) - self.gradient(input_r_high_gray, 'y', device))
        grad_loss_all = torch.mean(x_loss + y_loss)
        return grad_loss_all

    def ssim_loss(self,output_r, input_high_r, device="cuda"):

        output_r_1 = output_r[:,0:1,:,:]
        input_high_r_1 = input_high_r[:,0:1,:,:]
        ssim_r_1 = self.ssim(output_r_1, input_high_r_1)
        output_r_2 = output_r[:,1:2,:,:]
        input_high_r_2 = input_high_r[:,1:2,:,:]
        ssim_r_2 = self.ssim(output_r_2, input_high_r_2)
        output_r_3 = output_r[:,2:3,:,:]
        input_high_r_3 = input_high_r[:,2:3,:,:]
        ssim_r_3 = self.ssim(output_r_3, input_high_r_3)
        ssim_r = (ssim_r_1 + ssim_r_2 + ssim_r_3)/3.0
        loss_ssim1 = 1-ssim_r
        return loss_ssim1


    def forward(self, output_r, input_high_r, device="cuda"):
        loss_square = torch.mean(torch.square(output_r  - input_high_r))
        loss_ssim = self.ssim_loss(output_r, input_high_r, device)
        loss_grad = self.grad_loss(output_r, input_high_r, device)

        loss_restoration = loss_square + loss_grad + loss_ssim

        return loss_restoration
    

class KinDLoss_illumina(nn.Module):
    def __init__(self):
        super().__init__()


    def gradient(self,input_tensor, direction, device="cuda"):
        a = input_tensor.shape[0]
    
        b = torch.zeros(input_tensor.shape[2],1)
        b = torch.zeros(input_tensor.shape[0],input_tensor.shape[1],input_tensor.shape[2],1)
        b = b.to(device)

        input_tensor = torch.cat((input_tensor,b),3)
    
        a = torch.zeros(1, input_tensor.shape[3])
        a = torch.zeros(input_tensor.shape[0],input_tensor.shape[1],1,input_tensor.shape[3])
        a = a.to(device)
        
        input_tensor = torch.cat((input_tensor,a), 2)
    
        c = [[0, 0], [-1, 1]]
        c = torch.FloatTensor(c)
        c = c.to(device)

        smooth_kernel_x = torch.reshape(c,(1,1,2,2))
        smooth_kernel_y = smooth_kernel_x.permute( [0, 1,3,2])

        if direction == "x":
            kernel = smooth_kernel_x
        elif direction == "y":
            kernel = smooth_kernel_y
        weight = nn.Parameter(data=kernel, requires_grad=False)
        gradient_orig = torch.abs(F.conv2d(input_tensor, weight, stride=1,padding=0))
        grad_min = torch.min(gradient_orig)
        grad_max = torch.max(gradient_orig)
        grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
        return grad_norm
        
    def grad_loss(self, input_i_low, input_i_high, device="cuda"):
        x_loss = torch.square(self.gradient(input_i_low, 'x', device) - self.gradient(input_i_high, 'x', device))
        y_loss = torch.square(self.gradient(input_i_low, 'y', device) - self.gradient(input_i_high, 'y', device))
        grad_loss_all = torch.mean(x_loss + y_loss)
        return grad_loss_all

    def forward(self, output_i, input_high_i, device="cuda"):
        loss_grad = self.grad_loss(output_i, input_high_i, device)
        loss_square = torch.mean(torch.square(output_i  - input_high_i))

        loss_adjust =  loss_square + loss_grad 

        return loss_adjust