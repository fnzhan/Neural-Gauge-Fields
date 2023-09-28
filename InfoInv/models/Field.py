import torch
from .FieldBase import *
from .networks import *
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(20211202)


class TriPlane(Base):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TriPlane, self).__init__(aabb, gridSize, device, **kargs)

    def init_model(self, res=256, dim=96, scale=0.1, device=None):

        self.plane_xy = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))
        self.plane_yz = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))
        self.plane_xz = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))

        self.density_dim = 24
        self.rgb_dim = dim - self.density_dim

        self.density_decoder = density_decoder(feat_dim=self.density_dim*3, middle_dim=32).to(device)
        self.rgb_decoder = rgb_decoder(feat_dim=self.rgb_dim*3, view_pe=2, middle_dim=64).to(device)


    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [
                     {'params': self.plane_xy, 'lr': lr_init_spatialxyz},
                     {'params': self.plane_yz, 'lr': lr_init_spatialxyz},
                     {'params': self.plane_xz, 'lr': lr_init_spatialxyz},

                     {'params': self.rgb_decoder.parameters(), 'lr': lr_init_network},
                     {'params': self.density_decoder.parameters(), 'lr': lr_init_network},

                     ]
        return grad_vars

    def feature2density(self, density_features, density_shift=-10):
        return F.softplus(density_features+density_shift)


    def transform(self, valid_xyz):

        N, _ = valid_xyz.shape
        xy, yz, xz = valid_xyz[:, :2], valid_xyz[:, 1:], valid_xyz[:, ::2]

        target_xy, target_yz, target_xz = xy, yz, xz

        return target_xy, target_yz, target_xz

    def compute_density(self, xy, yz, xz, infoinv=True):

        xyz = torch.cat([xy, yz[:, 1:]],  dim=-1)
        xyz_pe = positional_encoding(xyz, 4).permute(1, 0)

        xy, yz, xz = xy.unsqueeze(0).unsqueeze(2), yz.unsqueeze(0).unsqueeze(2), xz.unsqueeze(0).unsqueeze(2)

        xy_feat = F.grid_sample(self.plane_xy[:, :self.density_dim, ...], xy, align_corners=True).view(-1, xy.shape[1])
        yz_feat = F.grid_sample(self.plane_yz[:, :self.density_dim, ...], yz, align_corners=True).view(-1, xy.shape[1])
        xz_feat = F.grid_sample(self.plane_xz[:, :self.density_dim, ...], xz, align_corners=True).view(-1, xy.shape[1])

        if infoinv:
            xy_feat, yz_feat, xz_feat = xy_feat * xyz_pe, yz_feat * xyz_pe, xz_feat * xyz_pe

        plane_feat = torch.cat([xy_feat, yz_feat, xz_feat]).T

        feature = self.density_decoder(plane_feat)
        density = self.feature2density(feature).reshape(-1)
        return density

    def compute_rgb(self, xy, yz, xz, view_sampled, infoinv=True):

        xyz = torch.cat([xy, yz[:, 1:]],  dim=-1)
        xyz_pe = positional_encoding(xyz, 12).permute(1, 0)

        xy, yz, xz = xy.unsqueeze(0).unsqueeze(2), yz.unsqueeze(0).unsqueeze(2), xz.unsqueeze(0).unsqueeze(2)

        xy_feat = F.grid_sample(self.plane_xy[:, self.density_dim:, ...], xy, align_corners=True).view(-1, xy.shape[1])
        yz_feat = F.grid_sample(self.plane_yz[:, self.density_dim:, ...], yz, align_corners=True).view(-1, xy.shape[1])
        xz_feat = F.grid_sample(self.plane_xz[:, self.density_dim:, ...], xz, align_corners=True).view(-1, xy.shape[1])

        if infoinv:
            xy_feat, yz_feat, xz_feat = xy_feat * xyz_pe, yz_feat * xyz_pe, xz_feat * xyz_pe

        plane_feat = torch.cat([xy_feat, yz_feat, xz_feat]).T
        rgb = self.rgb_decoder(plane_feat, view_sampled)

        return rgb


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
