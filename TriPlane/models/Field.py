import torch
from .FieldBase import *
from .networks import *
import torch.nn as nn
from torch import einsum
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mpl_toolkits.mplot3d import axes3d

torch.manual_seed(20211202)


class TriPlane(Base):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TriPlane, self).__init__(aabb, gridSize, device, **kargs)

    def init_model(self, res=256, dim=64, scale=0.1, device=None, gauge_start=0):

        self.plane_xy = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))
        self.plane_yz = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))
        self.plane_xz = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))

        gauge_res = 256
        self.gauge_xy = torch.nn.Parameter(torch.zeros((1, 2, gauge_res, gauge_res), device=device))
        self.gauge_yz = torch.nn.Parameter(torch.zeros((1, 2, gauge_res, gauge_res), device=device))
        self.gauge_xz = torch.nn.Parameter(torch.zeros((1, 2, gauge_res, gauge_res), device=device))

        self.rgb_decoder = rgb_decoder(feat_dim=48*3, view_pe=2, middle_dim=64).to(device)
        self.density_decoder = torch.nn.Linear(16*3, 1).to(device)
        init_weights(self.density_decoder, 'xavier_uniform')

        self.gauge_start = gauge_start

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [
                     {'params': self.plane_xy, 'lr': lr_init_spatialxyz},
                     {'params': self.plane_yz, 'lr': lr_init_spatialxyz},
                     {'params': self.plane_xz, 'lr': lr_init_spatialxyz},
                     {'params': self.rgb_decoder.parameters(), 'lr': lr_init_network},
                     {'params': self.density_decoder.parameters(), 'lr': lr_init_network},

                     {'params': self.gauge_xy, 'lr': lr_init_network*0.1},
                     {'params': self.gauge_yz, 'lr': lr_init_network*0.1},
                     {'params': self.gauge_xz, 'lr': lr_init_network*0.1},
                     ]
        return grad_vars

    def feature2density(self, density_features, density_shift=-10):
        # return F.softplus(density_features)
        return F.softplus(density_features+density_shift)


    def compute_gauge(self, valid_xyz, iteration=0):

        N, _ = valid_xyz.shape
        xy, yz, xz = valid_xyz[:, :2], valid_xyz[:, 1:], valid_xyz[:, ::2]

        if iteration >= self.gauge_start:
            dxy = F.grid_sample(self.gauge_xy, xy.view(1, N, 1, 2), align_corners=True).squeeze().permute(1, 0)
            dyz = F.grid_sample(self.gauge_yz, yz.view(1, N, 1, 2), align_corners=True).squeeze().permute(1, 0)
            dxz = F.grid_sample(self.gauge_xz, xz.view(1, N, 1, 2), align_corners=True).squeeze().permute(1, 0)
            target_xy, target_yz, target_xz = xy + dxy, yz + dyz, xz + dxz

            target_xy[:, 0] += dxz[:, 0]
            target_xy[:, 1] += dyz[:, 0]

            target_yz[:, 0] += dxy[:, 1]
            target_yz[:, 1] += dxz[:, 1]

            target_xz[:, 0] += dxy[:, 0]
            target_xz[:, 1] += dyz[:, 1]
        else:
            target_xy, target_yz, target_xz = xy, yz, xz

        return target_xy, target_yz, target_xz

    def compute_density(self, xy, yz, xz):
        xy, yz, xz = xy.unsqueeze(0).unsqueeze(2), yz.unsqueeze(0).unsqueeze(2), xz.unsqueeze(0).unsqueeze(2)
        xy_feat = F.grid_sample(self.plane_xy[:, :16, ...], xy, align_corners=True).view(-1, xy.shape[1])
        xy_feat = xy_feat.permute(1, 0)
        yz_feat = F.grid_sample(self.plane_yz[:, :16, ...], yz, align_corners=True).view(-1, xy.shape[1])
        yz_feat = yz_feat.permute(1, 0)
        xz_feat = F.grid_sample(self.plane_xz[:, :16, ...], xz, align_corners=True).view(-1, xy.shape[1])
        xz_feat = xz_feat.permute(1, 0)


        xyz_feat_density = torch.cat([xy_feat, yz_feat, xz_feat], dim=-1)

        density_feat = self.density_decoder(xyz_feat_density).reshape(-1)
        density = self.feature2density(density_feat)
        return density

    def compute_rgb(self, xy, yz, xz, view_sampled):
        xy, yz, xz = xy.unsqueeze(0).unsqueeze(2), yz.unsqueeze(0).unsqueeze(2), xz.unsqueeze(0).unsqueeze(2)
        # target_xy, target_yz, target_xz = valid_gauge
        # gauge = torch.zeros((*xyz_sampled.shape[:2], 2), device=xyz_sampled.device)
        xy_feat = F.grid_sample(self.plane_xy[:, 16:, ...], xy, align_corners=True).view(-1, xy.shape[1])
        xy_feat = xy_feat.permute(1, 0)
        yz_feat = F.grid_sample(self.plane_yz[:, 16:, ...], yz, align_corners=True).view(-1, xy.shape[1])
        yz_feat = yz_feat.permute(1, 0)
        xz_feat = F.grid_sample(self.plane_xz[:, 16:, ...], xz, align_corners=True).view(-1, xy.shape[1])
        xz_feat = xz_feat.permute(1, 0)
        xyz_feat_rgb = torch.cat([xy_feat, yz_feat, xz_feat], dim=-1)
        rgb = self.rgb_decoder(xyz_feat_rgb, view_sampled)
        return rgb


    def up_sampling(self, res):

        self.plane_xy = torch.nn.Parameter(F.interpolate(self.plane_xy.data, size=(res[1], res[0]), mode='bilinear', align_corners=True))
        self.plane_yz = torch.nn.Parameter(F.interpolate(self.plane_yz.data, size=(res[2], res[1]), mode='bilinear', align_corners=True))
        self.plane_xz = torch.nn.Parameter(F.interpolate(self.plane_xz.data, size=(res[2], res[0]), mode='bilinear', align_corners=True))
        self.init_para(res)
        print(f'upsamping to {res}')


    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        self.plane_xy = torch.nn.Parameter(self.plane_xy.data[..., t_l[1]:b_r[1], t_l[0]:b_r[0]])
        self.plane_yz = torch.nn.Parameter(self.plane_yz.data[..., t_l[2]:b_r[2], t_l[1]:b_r[1]])
        self.plane_xz = torch.nn.Parameter(self.plane_xz.data[..., t_l[2]:b_r[2], t_l[0]:b_r[0]])

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.init_para((newSize[0], newSize[1], newSize[2]))

    def vectorDiffs(self, vector_comps):
        total = 0

        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]

            dotp = torch.matmul(vector_comps[idx].view(n_comp, n_size),
                                vector_comps[idx].view(n_comp, n_size).transpose(-1, -2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp - 1, n_comp + 1)[..., :-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)

    def density_L1(self):
        total = torch.mean(torch.abs(self.plane_xy)) + torch.mean(torch.abs(self.plane_yz)) \
                + torch.mean(torch.abs(self.plane_xz))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2  # + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2  # + reg(self.app_line[idx]) * 1e-3
        return total