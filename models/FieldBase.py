import numpy as np
import time
import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]

def positional_encoding(positions, freqs):
    '''encode positions with positional encoding
        positions: :math:`(...,D)`
        freqs: int
    Return:
        pts: :math:`(...,2DF)`
    '''
    # 2,3,40 = 240 -
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] +
                                                      (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

def InfoNest(positions, freqs, beta=1.0):

    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)
    # freq_bands = 1 / theta
    # print('*******', freq_bands[: 10]).

    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] +
                                                      (freqs * positions.shape[-1], ))
    cosp, sinp = torch.cos(pts), torch.sin(pts)
    # print('pe1 shape', (sinp*beta).shape) #pe1 torch.Size([584167, 120]).
    pe1 = (cosp - sinp * beta)  #torch.cos(pts) - torch.sin(pts) position, 3, bs. 0, 1.
    pe2 = (cosp + sinp * beta)  #torch.sin(pts) + torch.cos(pts).position.
    # print('*************', pe2[0, :16]). freq_bands = freq_bands.
    pe = torch.cat([pe1, pe2], dim=-1)

    return pe



def rot_pe(positions, freqs):
    '''encode positions with positional encoding
        positions: :math:`(...,D)`
        freqs: int
    Return:
        pts: :math:`(...,2DF)`
    '''
    # 2,3,40 = 240 -
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] +
                                                      (freqs * positions.shape[-1], ))  # (..., DF)
    # pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1). rot cosine.
    cos_pe = torch.cos(pts)
    sin_pe = torch.sin(pts)
    # value_mask = torch.ones_like(sin_pe).
    return cos_pe, sin_pe

class Base(torch.nn.Module):
    def __init__(self, aabb, gridSize, device, near_far=[2.0,6.0], alphaMask_thres=0.001, distance_scale=25,
                 rayMarch_weight_thres=0.0001, step_ratio=2.0, transform_type='discrete'):
        super(Base, self).__init__()
        self.aabb = aabb
        self.device=device

        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres

        self.near_far = near_far
        self.step_ratio = step_ratio
        self.transform_type = transform_type

        self.init_sample(gridSize)
        self.init_model(device=device)

        self.reg_iter = 30000
        # self.res = 16

    def init_sample(self, gridSize):
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
    
    def compute_feature(self, xyz_sampled):
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
                  # 'gauge_tensor': self.gauge_tensor
                  }
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        torch.save(ckpt, path)

        # if not:
        #     alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()

        # kwargs = self.get_kwargs()
        # ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        # if self.alphaMask is not None:
        #     alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
        #     ckpt.update({'alphaMask.shape':alpha_volume.shape})
        #     ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
        #     ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        # torch.save(ckpt, path)

    def load(self, ckpt):
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
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5):
        print('========> filtering rays ...')
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            # if bbox_only:
            vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
            rate_a = (self.aabb[1] - rays_o) / vec
            rate_b = (self.aabb[0] - rays_o) / vec
            t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
            t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
            mask_inbbox = t_max > t_min

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered]


    def compute_alpha(self, xyz_locs, length=1):

        alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            validsigma, _, _ = self.compute_feature(xyz_sampled)
            sigma[alpha_mask] = validsigma
        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])

        return alpha


    def forward(self, rays_chunk, white_bg=True, is_train=False, N_samples=-1, iteration=0):

        # sample points
        viewdirs = rays_chunk[:, 3:6]

        xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        # print('1********', xyz_sampled.shape) torch.Size([4096, 443, 3]) #

        density = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            # print('*****2', xyz_sampled.min(), xyz_sampled.max())
            # print('****grid_sample*****', xyz_sampled[ray_valid].shape, xyz_sampled[ray_valid].min()
            # xyz_sampled[ray_valid].max())
            # ** ** grid_sample ** ** *torch.Size([2330202, 3])
            valid_density, valid_rgb, output = self.compute_feature(xyz_sampled[ray_valid], viewdirs[ray_valid], iteration=iteration)
            density[ray_valid] = valid_density
            rgb[ray_valid] = valid_rgb
            # reg_loss = output['reg_loss']

        # valid_density, valid_rgb,  split. split.

        alpha, weight, bg_weight = raw2alpha(density, dists * self.distance_scale)
        valid_weight = weight[ray_valid]
        # app_mask = valid_weight > self.rayMarch_weight_thres.

        if ray_valid.any() and self.transform_type=='continuous' and valid_weight.sum()>0 and is_train and (iteration>500 and iteration<750):

            # target_gauge = output['target_gauge']
            # target_gauge_feat = self.reg_network(target_gauge.squeeze())
            # output['target_gauge_feat'] = target_gauge_feat
            # source_gauge_feat = output['source_gauge_feat']
            # mi_loss = ((source_gauge_feat - target_gauge_feat) ** 2).sum(-1)
            # mi_loss = (mi_loss * valid_weight).mean() * 0.

            # sampled_index = torch.multinomial(valid_weight, 1024, replacement=False)
            # xy, yz, xz = output['target_gauge']
            # sampled_xy, sampled_yz, sampled_xz = xy[sampled_index], yz[sampled_index], xz[sampled_index]
            # sampled_xy.

            self.gauge_transformation(output, valid_weight)
            reg_loss = torch.tensor(([0]), device=xyz_sampled.device)
            # reg_loss = prior_loss
            # print('emd loss', emd)
        elif ray_valid.any() and self.transform_type=='discrete' and valid_weight.sum()>0 and is_train:
            reg_loss = torch.tensor(([0]), device=xyz_sampled.device)
        else:
            reg_loss = torch.tensor(([0]), device=xyz_sampled.device)


        # if iteration == 1500:
        #     sampled_index = torch.multinomial(valid_weight, 1024, replacement=False)
        #     xy, yz, xz = output['target_gauge']
        #     sampled_xy, sampled_yz, sampled_xz = xy[sampled_index], yz[sampled_index], xz[sampled_index]
        #
        #     xy_origin = xyz_sampled[ray_valid][sampled_index][:, :2]
        #     # print('after_all********', xy.min(), xy.max())
        #     # print('before_surface********', xy_origin.min(), xy_origin.max())
        #     # print('after_surface********', sampled_xy.min(), sampled_xy.max())
        #
        #     fig = plt.figure(figsize=(6, 6))
        #     tmp = xy_origin.squeeze().detach().cpu().numpy()
        #     plt.scatter(tmp[:, 0], tmp[:, 1], 40 * 500 / len(tmp), [(0.55, 0.55, 0.95)], edgecolors="none")
        #     # plt.axis([-1, 1, -1, 1])
        #     plt.gca().set_aspect("equal", adjustable="box")
        #     plt.xticks([], [])
        #     plt.yticks([], [])
        #     plt.tight_layout()
        #     fig.savefig('emd_origin.png', bbox_inches='tight')
        #     plt.close()
        #
        #     fig = plt.figure(figsize=(6, 6))
        #     tmp = sampled_xy.squeeze().detach().cpu().numpy()
        #     plt.scatter(tmp[:, 0], tmp[:, 1], 40 * 500 / len(tmp), [(0.55, 0.55, 0.95)], edgecolors="none")
        #     # plt.axis([-1, 1, -1, 1])
        #     plt.gca().set_aspect("equal", adjustable="box")
        #     plt.xticks([], [])
        #     plt.yticks([], [])
        #     plt.tight_layout()
        #     fig.savefig('emd_final.png', bbox_inches='tight')
        #     plt.close()
        #     1/0



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
        output['reg_loss'] = reg_loss

        return output