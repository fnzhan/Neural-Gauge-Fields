import numpy as np
import time
import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(20211202)
np.random.seed(20211202)

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]


class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1



class Base(torch.nn.Module):
    def __init__(self, aabb, gridSize, device, alphaMask=None, near_far=[2.0,6.0], alphaMask_thres=0.001, distance_scale=25,
                 rayMarch_weight_thres=0.0001, step_ratio=2.0):
        super(Base, self).__init__()
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device=device

        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.init_para(gridSize)
        self.init_model(device=device)


    def init_para(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize= torch.LongTensor(gridSize).to(self.device)
        self.units=self.aabbSize / (self.gridSize-1)
        self.stepSize=torch.mean(self.units)*self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_model(self, code_num=256, code_dim=256, scale=0.1, res=16, device=None):
        pass
    
    def compute_gauge(self, xyz_sampled):
        pass

    def compute_density(self, xyz_sampled):
        pass

    def compute_rgb(self, xyz_sampled):
        pass

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        pass

    def save(self, path):
        kwargs = {'aabb': self.aabb,
                  'gridSize': self.gridSize.tolist(),
                  'alphaMask_thres': self.alphaMask_thres,
                  'distance_scale': self.distance_scale,
                  'rayMarch_weight_thres': self.rayMarch_weight_thres,
                  'near_far': self.near_far,
                  'step_ratio': self.step_ratio,
                  }
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape': alpha_volume.shape})
            ckpt.update({'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples>0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = (t_min[...,None] + step)

        # print('*****rays', rays_o[...,None,:].shape, rays_d[...,None,:].shape, interpx[...,None].shape)
        # torch.Size([4096, 1, 3]) torch.Size([4096, 1, 3]) torch.Size([4096, 443, 1])
        # 1/0

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)

        # print('****ray_ndc****', N_samples, rays_pts.shape, interpx.shape) # 1039 torch.Size([4096, 1039, 3]) torch.Size([4096, 1039])
        # 1/0

        return rays_pts, interpx, ~mask_outbbox


    def compute_alpha(self, xyz_locs, length=1):

        # alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)

        density = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            # alpha_density = self.compute_density(xyz_sampled)
            alpha_xy, alpha_yz, alpha_xz, alpha_z, alpha_x, alpha_y = self.compute_gauge(xyz_sampled, iteration=-1)
            alpha_density = self.compute_density(alpha_xy, alpha_yz, alpha_xz, alpha_z, alpha_x, alpha_y)
            density[alpha_mask] = alpha_density
        alpha = 1 - torch.exp(-density*length).view(xyz_locs.shape[:-1])

        return alpha

    @torch.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):

        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
        # dense_xyz = dense_xyz
        # alpha = alpha.clamp(0,1) #.transpose(0,2).contiguous()[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]


        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0
        # alpha[alpha >= 0.05] = 1
        # alpha[alpha < 0.05] = 0

        # print('2', alpha[10].shape, alpha[10].sum())

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha>0.5]
        # print('3', valid_xyz[259], valid_xyz.shape)

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        # print('4', xyz_min, xyz_max)


        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        # 1/0
        return new_aabb

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
                mask_inbbox = t_max > t_min
            else:
                xyz_sampled, _, _ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox = (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered]




    def forward(self, rays_chunk, white_bg=True, is_train=False, N_samples=-1, iteration=0):

        output = {}
        # sample points
        viewdirs = rays_chunk[:, 3:6]

        xyz_sampled, z_vals, valid_ray = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        if self.alphaMask is not None:
            # print('***1***', xyz_sampled[valid_ray].shape)
            alphas = self.alphaMask.sample_alpha(xyz_sampled[valid_ray])
            alpha_mask = alphas > 0
            invalid_ray = ~valid_ray
            invalid_ray[valid_ray] |= (~alpha_mask)
            valid_ray = ~invalid_ray

        density = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        xy = torch.zeros((*xyz_sampled.shape[:2], 2), device=xyz_sampled.device) #.view(-1, 2)
        yz = torch.zeros((*xyz_sampled.shape[:2], 2), device=xyz_sampled.device) #.view(-1, 2)
        xz = torch.zeros((*xyz_sampled.shape[:2], 2), device=xyz_sampled.device) #.view(-1, 2)
        z = torch.zeros((*xyz_sampled.shape[:2], 1), device=xyz_sampled.device)
        x = torch.zeros((*xyz_sampled.shape[:2], 1), device=xyz_sampled.device)
        y = torch.zeros((*xyz_sampled.shape[:2], 1), device=xyz_sampled.device)


        if valid_ray.any():
            # print('***2***', xyz_sampled.shape, xyz_sampled[valid_ray].shape)
            xyz_sampled = self.normalize_coord(xyz_sampled)
            valid_xy, valid_yz, valid_xz, valid_z, valid_x, valid_y = self.compute_gauge(xyz_sampled[valid_ray], iteration=iteration)
            valid_density = self.compute_density(valid_xy, valid_yz, valid_xz, valid_z, valid_x, valid_y)
            density[valid_ray] = valid_density
            # valid_ray = valid_ray.view(-1)
            # print('2:', xy.shape, valid_ray.shape, valid_xy.shape)
            xy[valid_ray], yz[valid_ray], xz[valid_ray] = valid_xy, valid_yz, valid_xz
            z[valid_ray], x[valid_ray], y[valid_ray] = valid_z, valid_x, valid_y
            # 1/0

        alpha, weight, bg_weight = raw2alpha(density, dists * self.distance_scale)
        rgb_mask = weight > self.rayMarch_weight_thres  #torch.Size([4096, 1039])
        # print('1:', weight.shape, rgb_mask.shape, xy.shape)

        if rgb_mask.any():
            valid_rgb = self.compute_rgb(xy[rgb_mask], yz[rgb_mask], xz[rgb_mask], z[rgb_mask], x[rgb_mask], y[rgb_mask], viewdirs[rgb_mask])
            rgb[rgb_mask] = valid_rgb

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        rgb_map = rgb_map.clamp(0,1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        output['rgb_map'] = rgb_map
        output['depth_map'] = depth_map
        # output['reg_loss'] = reg_loss

        return output