import torch
from .FieldBase import *
from .networks import *
import torch.nn as nn
from torch import einsum
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mpl_toolkits.mplot3d import axes3d

torch.manual_seed(20211202)


class TensoRF(Base):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensoRF, self).__init__(aabb, gridSize, device, **kargs)

    def init_model(self, res=256, dim=64, scale=0.1, device=None):

        self.plane_xy = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))
        self.plane_yz = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))
        self.plane_xz = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))

        self.line_z = torch.nn.Parameter(scale * torch.randn((1, dim, res, 1), device=device))
        self.line_x = torch.nn.Parameter(scale * torch.randn((1, dim, res, 1), device=device))
        self.line_y = torch.nn.Parameter(scale * torch.randn((1, dim, res, 1), device=device))

        gauge_res = 256
        self.gauge_xy = torch.nn.Parameter(torch.zeros((1, 2, gauge_res, gauge_res), device=device))
        self.gauge_yz = torch.nn.Parameter(torch.zeros((1, 2, gauge_res, gauge_res), device=device))
        self.gauge_xz = torch.nn.Parameter(torch.zeros((1, 2, gauge_res, gauge_res), device=device))

        self.gauge_z = torch.nn.Parameter(torch.zeros((1, 1, gauge_res, 1), device=device))
        self.gauge_x = torch.nn.Parameter(torch.zeros((1, 1, gauge_res, 1), device=device))
        self.gauge_y = torch.nn.Parameter(torch.zeros((1, 1, gauge_res, 1), device=device))

        self.rgb_decoder = rgb_decoder(feat_dim=48*3, view_pe=2, middle_dim=64).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [
                     {'params': self.plane_xy, 'lr': lr_init_spatialxyz},
                     {'params': self.plane_yz, 'lr': lr_init_spatialxyz},
                     {'params': self.plane_xz, 'lr': lr_init_spatialxyz},

                     {'params': self.line_z, 'lr': lr_init_spatialxyz},
                     {'params': self.line_x, 'lr': lr_init_spatialxyz},
                     {'params': self.line_y, 'lr': lr_init_spatialxyz},

                     {'params': self.rgb_decoder.parameters(), 'lr': lr_init_network},

                     {'params': self.gauge_xy, 'lr': lr_init_network*0.1},
                     {'params': self.gauge_yz, 'lr': lr_init_network*0.1},
                     {'params': self.gauge_xz, 'lr': lr_init_network*0.1},

                     {'params': self.gauge_z, 'lr': lr_init_network * 0.1},
                     {'params': self.gauge_x, 'lr': lr_init_network * 0.1},
                     {'params': self.gauge_y, 'lr': lr_init_network * 0.1},
                     ]
        return grad_vars

    def feature2density(self, density_features, density_shift=-10):
        return F.softplus(density_features+density_shift)

    # def compute_gauge(self, valid_xyz, iteration=0):
    #
    #     N, _ = valid_xyz.shape
    #     xy, yz, xz = valid_xyz[:, :2], valid_xyz[:, 1:], valid_xyz[:, ::2]
    #     z, x, y = valid_xyz[:, 2:], valid_xyz[:, :1], valid_xyz[:, 1:2]
    #
    #     if iteration > 3000:
    #         dxy = F.grid_sample(self.gauge_xy, xy.view(1, N, 1, 2), align_corners=True).view(-1, xy.shape[0]).permute(1, 0)
    #         dyz = F.grid_sample(self.gauge_yz, yz.view(1, N, 1, 2), align_corners=True).view(-1, yz.shape[0]).permute(1, 0)
    #         dxz = F.grid_sample(self.gauge_xz, xz.view(1, N, 1, 2), align_corners=True).view(-1, xz.shape[0]).permute(1, 0)
    #
    #         # dxyz = torch.zeros_like(valid_xyz)
    #         target_xyz = valid_xyz + dxy + dyz + dxz
    #         target_xy, target_yz, target_xz = target_xyz[:, :2], target_xyz[:, 1:], target_xyz[:, ::2]
    #         target_z, target_x, target_y = target_xyz[:, 2:], target_xyz[:, :1], target_xyz[:, 1:2]
    #
    #     else:
    #         target_xy, target_yz, target_xz = xy, yz, xz
    #         target_z, target_x, target_y = z, x, y
    #
    #     return target_xy, target_yz, target_xz, target_z, target_x, target_y


    def compute_gauge(self, valid_xyz, iteration=0):

        N, _ = valid_xyz.shape
        xy, yz, xz = valid_xyz[:, :2], valid_xyz[:, 1:], valid_xyz[:, ::2]
        z, x, y = valid_xyz[:, 2:], valid_xyz[:, :1], valid_xyz[:, 1:2]

        if iteration > 3000:
            dxy = F.grid_sample(self.gauge_xy, xy.view(1, N, 1, 2), align_corners=True).view(-1, xy.shape[0]).permute(1, 0)
            dyz = F.grid_sample(self.gauge_yz, yz.view(1, N, 1, 2), align_corners=True).view(-1, yz.shape[0]).permute(1, 0)
            dxz = F.grid_sample(self.gauge_xz, xz.view(1, N, 1, 2), align_corners=True).view(-1, xz.shape[0]).permute(1, 0)
            target_xy, target_yz, target_xz = xy + dxy[:, :2], yz + dyz[:, :2], xz + dxz[:, :2]

            z_, x_, y_ = torch.stack((torch.zeros_like(z), z), dim=-1).unsqueeze(0), \
                         torch.stack((torch.zeros_like(x), x), dim=-1).unsqueeze(0), \
                         torch.stack((torch.zeros_like(y), y), dim=-1).unsqueeze(0)
            dz = F.grid_sample(self.gauge_z, z_, align_corners=True).view(-1, z_.shape[1]).permute(1, 0)
            dx = F.grid_sample(self.gauge_x, x_, align_corners=True).view(-1, x_.shape[1]).permute(1, 0)
            dy = F.grid_sample(self.gauge_y, y_, align_corners=True).view(-1, y_.shape[1]).permute(1, 0)
            target_z = z + dz
            target_x = x + dx
            target_y = y + dy

        else:
            target_xy, target_yz, target_xz = xy, yz, xz
            target_z, target_x, target_y = z, x, y

        return target_xy, target_yz, target_xz, target_z, target_x, target_y

    def compute_density(self, xy, yz, xz, z, x, y):
        xy, yz, xz = xy.unsqueeze(0).unsqueeze(2), yz.unsqueeze(0).unsqueeze(2), xz.unsqueeze(0).unsqueeze(2)
        z, x, y = torch.stack((torch.zeros_like(z), z), dim=-1).unsqueeze(0), \
                  torch.stack((torch.zeros_like(x), x), dim=-1).unsqueeze(0), \
                  torch.stack((torch.zeros_like(y), y), dim=-1).unsqueeze(0)
        # z, x, y = z.unsqueeze(0), x.unsqueeze(0), y.unsqueeze(0)


        xy_feat = F.grid_sample(self.plane_xy[:, :16, ...], xy, align_corners=True).view(-1, xy.shape[1])
        yz_feat = F.grid_sample(self.plane_yz[:, :16, ...], yz, align_corners=True).view(-1, xy.shape[1])
        xz_feat = F.grid_sample(self.plane_xz[:, :16, ...], xz, align_corners=True).view(-1, xy.shape[1])
        plane_feat = torch.cat([xy_feat, yz_feat, xz_feat])

        z_feat = F.grid_sample(self.line_z[:, :16, ...], z, align_corners=True).view(-1, z.shape[1])
        x_feat = F.grid_sample(self.line_x[:, :16, ...], x, align_corners=True).view(-1, x.shape[1])
        y_feat = F.grid_sample(self.line_y[:, :16, ...], y, align_corners=True).view(-1, y.shape[1])
        line_feat = torch.cat([z_feat, x_feat, y_feat])

        density_feat = torch.sum(plane_feat * line_feat, dim=0)
        density = self.feature2density(density_feat)
        # density_feat = self.density_decoder(plane_feat_density).reshape(-1)
        # density = self.feature2density(density_feat)
        return density

    def compute_rgb(self, xy, yz, xz, z, x, y, view_sampled):
        xy, yz, xz = xy.unsqueeze(0).unsqueeze(2), yz.unsqueeze(0).unsqueeze(2), xz.unsqueeze(0).unsqueeze(2)
        z, x, y = torch.stack((torch.zeros_like(z), z), dim=-1).unsqueeze(0), \
                  torch.stack((torch.zeros_like(x), x), dim=-1).unsqueeze(0), \
                  torch.stack((torch.zeros_like(y), y), dim=-1).unsqueeze(0)

        xy_feat = F.grid_sample(self.plane_xy[:, 16:, ...], xy, align_corners=True).view(-1, xy.shape[1])
        yz_feat = F.grid_sample(self.plane_yz[:, 16:, ...], yz, align_corners=True).view(-1, xy.shape[1])
        xz_feat = F.grid_sample(self.plane_xz[:, 16:, ...], xz, align_corners=True).view(-1, xy.shape[1])
        plane_feat = torch.cat([xy_feat, yz_feat, xz_feat])

        z_feat = F.grid_sample(self.line_z[:, 16:, ...], z, align_corners=True).view(-1, z.shape[1])
        x_feat = F.grid_sample(self.line_x[:, 16:, ...], x, align_corners=True).view(-1, x.shape[1])
        y_feat = F.grid_sample(self.line_y[:, 16:, ...], y, align_corners=True).view(-1, y.shape[1])
        line_feat = torch.cat([z_feat, x_feat, y_feat])

        rgb_feat = (plane_feat * line_feat).T
        rgb = self.rgb_decoder(rgb_feat, view_sampled)

        return rgb




    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)
        newSize = b_r - t_l
        self.aabb = new_aabb

        self.plane_xy = torch.nn.Parameter(self.plane_xy.data[..., t_l[1]:b_r[1], t_l[0]:b_r[0]])
        self.plane_yz = torch.nn.Parameter(self.plane_yz.data[..., t_l[2]:b_r[2], t_l[1]:b_r[1]])
        self.plane_xz = torch.nn.Parameter(self.plane_xz.data[..., t_l[2]:b_r[2], t_l[0]:b_r[0]])

        self.line_z = torch.nn.Parameter(self.line_z.data[..., t_l[2]:b_r[2], :])
        self.line_x = torch.nn.Parameter(self.line_x.data[..., t_l[0]:b_r[0], :])
        self.line_y = torch.nn.Parameter(self.line_y.data[..., t_l[1]:b_r[1], :])

        self.gauge_xy = torch.nn.Parameter(self.gauge_xy.data[..., t_l[1]:b_r[1], t_l[0]:b_r[0]])
        self.gauge_yz = torch.nn.Parameter(self.gauge_yz.data[..., t_l[2]:b_r[2], t_l[1]:b_r[1]])
        self.gauge_xz = torch.nn.Parameter(self.gauge_xz.data[..., t_l[2]:b_r[2], t_l[0]:b_r[0]])

        self.gauge_z = torch.nn.Parameter(self.gauge_z.data[..., t_l[2]:b_r[2], :])
        self.gauge_x = torch.nn.Parameter(self.gauge_x.data[..., t_l[0]:b_r[0], :])
        self.gauge_y = torch.nn.Parameter(self.gauge_y.data[..., t_l[1]:b_r[1], :])

        # self.gauge_xy = torch.nn.Parameter(torch.zeros((1, 2, newSize[1], newSize[0]), device=self.device))
        # self.gauge_yz = torch.nn.Parameter(torch.zeros((1, 2, newSize[2], newSize[1]), device=self.device))
        # self.gauge_xz = torch.nn.Parameter(torch.zeros((1, 2, newSize[2], newSize[0]), device=self.device))
        #
        # self.gauge_z = torch.nn.Parameter(torch.zeros((1, 1, newSize[2], 1), device=self.device))
        # self.gauge_x = torch.nn.Parameter(torch.zeros((1, 1, newSize[0], 1), device=self.device))
        # self.gauge_y = torch.nn.Parameter(torch.zeros((1, 1, newSize[1], 1), device=self.device))


        self.init_para((newSize[0], newSize[1], newSize[2]))



    def up_sampling(self, res):

        self.plane_xy = torch.nn.Parameter(F.interpolate(self.plane_xy.data, size=(res[1], res[0]), mode='bilinear', align_corners=True))
        self.plane_yz = torch.nn.Parameter(F.interpolate(self.plane_yz.data, size=(res[2], res[1]), mode='bilinear', align_corners=True))
        self.plane_xz = torch.nn.Parameter(F.interpolate(self.plane_xz.data, size=(res[2], res[0]), mode='bilinear', align_corners=True))

        self.line_z = torch.nn.Parameter(F.interpolate(self.line_z.data, size=(res[2], 1), mode='bilinear', align_corners=True))
        self.line_x = torch.nn.Parameter(F.interpolate(self.line_x.data, size=(res[0], 1), mode='bilinear', align_corners=True))
        self.line_y = torch.nn.Parameter(F.interpolate(self.line_y.data, size=(res[1], 1), mode='bilinear', align_corners=True))

        self.gauge_xy = torch.nn.Parameter(F.interpolate(self.gauge_xy.data, size=(res[1], res[0]), mode='bilinear', align_corners=True))
        self.gauge_yz = torch.nn.Parameter(F.interpolate(self.gauge_yz.data, size=(res[2], res[1]), mode='bilinear', align_corners=True))
        self.gauge_xz = torch.nn.Parameter(F.interpolate(self.gauge_xz.data, size=(res[2], res[0]), mode='bilinear', align_corners=True))

        self.gauge_z = torch.nn.Parameter(F.interpolate(self.gauge_z.data, size=(res[2], 1), mode='bilinear', align_corners=True))
        self.gauge_x = torch.nn.Parameter(F.interpolate(self.gauge_x.data, size=(res[0], 1), mode='bilinear', align_corners=True))
        self.gauge_y = torch.nn.Parameter(F.interpolate(self.gauge_y.data, size=(res[1], 1), mode='bilinear', align_corners=True))

        # self.gauge_xy = torch.nn.Parameter(torch.zeros((1, 2, newSize[1], newSize[0]), device=self.device))
        # self.gauge_yz = torch.nn.Parameter(torch.zeros((1, 2, newSize[2], newSize[1]), device=self.device))
        # self.gauge_xz = torch.nn.Parameter(torch.zeros((1, 2, newSize[2], newSize[0]), device=self.device))
        #
        # self.gauge_z = torch.nn.Parameter(torch.zeros((1, 1, newSize[2], 1), device=self.device))
        # self.gauge_x = torch.nn.Parameter(torch.zeros((1, 1, newSize[0], 1), device=self.device))
        # self.gauge_y = torch.nn.Parameter(torch.zeros((1, 1, newSize[1], 1), device=self.device))

        self.init_para(res)
        print(f'upsamping to {res}')

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



class TriPlane(Base):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TriPlane, self).__init__(aabb, gridSize, device, **kargs)

    def init_model(self, res=256, dim=64, scale=0.1, device=None):

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

        if iteration >= 4000:
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


        # if iteration % 1000 == 0:
        #     x_pos = np.arange(0, 1, 1/32)
        #     y_pos = np.arange(0, 1, 1/32)
        #     X_pos, Y_pos = np.meshgrid(x_pos, y_pos)
        #     gauge_xy = self.gauge_xy[0].detach().cpu().numpy()
        #     X_dir, Y_dir = gauge_xy[0, 112:144, 112:144], gauge_xy[1, 112:144, 112:144] #.flatten()   128 - 16  128+16
        #
        #     n = np.min(gauge_xy)
        #     color_array = np.sqrt(((Y_dir - n) / 2) ** 2 + ((X_dir - n) / 2) ** 2)
        #
        #     fig, ax = plt.subplots(figsize=(10, 10))
        #     ax.quiver(X_pos, Y_pos, X_dir, Y_dir, color_array, alpha=0.8) #, scale=0.2)
        #     ax.xaxis.set_ticks([])
        #     ax.yaxis.set_ticks([])
        #     ax.axis([0, 1., 0, 1.])
        #     ax.set_aspect('equal')
        #
        #     fig.savefig('gauge_fields_{}.png'.format(iteration))
        #     plt.close()

        # target_xy, target_yz, target_xz = target_xy.unsqueeze(0).unsqueeze(2), target_yz.unsqueeze(0).unsqueeze(2), \
        #                                   target_xz.unsqueeze(0).unsqueeze(2)

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





# class TriplaneNGF(Base):
#     def __init__(self, aabb, gridSize, device, **kargs):
#         super(TriplaneNGF, self).__init__(aabb, gridSize, device, **kargs)
#
#     def init_model(self, res=256, dim=64, scale=0.1, device=None):
#
#         self.plane_xy = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))
#         self.plane_yz = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))
#         self.plane_xz = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))
#
#         gauge_res = 64
#         self.gauge_xyz = torch.nn.Parameter(torch.zeros((1, 3, gauge_res, gauge_res, gauge_res), device=device))
#
#         # self.gauge_xy = torch.nn.Parameter(torch.zeros((1, 2, gauge_res, gauge_res), device=device))
#         # self.gauge_yz = torch.nn.Parameter(torch.zeros((1, 2, gauge_res, gauge_res), device=device))
#         # self.gauge_xz = torch.nn.Parameter(torch.zeros((1, 2, gauge_res, gauge_res), device=device))
#
#         self.rgb_decoder = rgb_decoder(feat_dim=48 * 3, view_pe=2, middle_dim=64).to(device)
#         self.density_decoder = torch.nn.Linear(16 * 3, 1).to(device)
#         init_weights(self.density_decoder, 'xavier_uniform')
#
#         # self.gauge_network = continuous_transform(input_dim=6, output_dim=2).to(device) # planes: [1,0,0], [0,1,0], [0,0,1].
#         self.reg_network = continuous_reg(input_dim=2, output_dim=3).to(device)
#
#         x, y = np.meshgrid(np.linspace(-0.9, 0.9, 32), np.linspace(-0.9, 0.9, 32), indexing="xy")
#         grid = np.array(list(zip(x.ravel(), y.ravel())))
#         self.sampled_prior = torch.from_numpy(grid).type(torch.FloatTensor).to(device)
#         self.EMD = SamplesLoss("sinkhorn", p=2, blur=0.01)
#
#     def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
#         grad_vars = [
#             {'params': self.plane_xy, 'lr': lr_init_spatialxyz},
#             {'params': self.plane_yz, 'lr': lr_init_spatialxyz},
#             {'params': self.plane_xz, 'lr': lr_init_spatialxyz},
#             {'params': self.rgb_decoder.parameters(), 'lr': lr_init_network},
#             {'params': self.density_decoder.parameters(), 'lr': lr_init_network},
#         ]
#         return grad_vars
#
#     def get_optparam_groups_fixed_gauge(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
#         grad_vars = [
#             {'params': self.plane_xy, 'lr': lr_init_spatialxyz},
#             {'params': self.plane_yz, 'lr': lr_init_spatialxyz},
#             {'params': self.plane_xz, 'lr': lr_init_spatialxyz},
#             {'params': self.rgb_decoder.parameters(), 'lr': lr_init_network},
#             {'params': self.density_decoder.parameters(), 'lr': lr_init_network},
#
#             {'params': self.gauge_xyz, 'lr': lr_init_network * 0.1},
#         ]
#         return grad_vars
#
#     def feature2density(self, density_features, density_shift=-10):
#         # return F.softplus(density_features)
#         return F.softplus(density_features + density_shift)
#
#     def compute_feature(self, xyz_sampled, view_sampled, iteration=0):
#         output = {}
#         N, _ = xyz_sampled.shape
#
#         xy, yz, xz = xyz_sampled[:, :2], xyz_sampled[:, 1:], xyz_sampled[:, ::2]
#
#         # target_xy, target_yz, target_xz = xy, yz, xz
#         xyz = xyz_sampled
#
#         if iteration >= 2000:
#             dxyz = F.grid_sample(self.gauge_xyz, xyz.view(1, N, 1, 1, 3), align_corners=True).squeeze().permute(1, 0)
#             target_xy, target_yz, target_xz = xy + dxyz[:, :2], yz + dxyz[:, 1:], xz + dxyz[:, ::2]
#         else:
#             target_xy, target_yz, target_xz = xy, yz, xz
#
#         # visualization:
#         if iteration % 3000 == 0:
#             x, y, z = np.meshgrid(np.arange(0, 1, 1/16),
#                                   np.arange(0, 1, 1/16),
#                                   np.arange(0, 1, 1/16))
#             gauge_xyz = self.gauge_xyz.detach().cpu().numpy()
#             x_dir, y_dir, z_dir = gauge_xyz[0, 0, ::4, ::4, ::4], gauge_xyz[0, 1, ::4, ::4, ::4], gauge_xyz[0, 2, ::4, ::4, ::4]
#
#             # n = np.min(gauge_xyz)
#             # color_array = np.sqrt(((x_dir - n) / 2) ** 2 + ((y_dir - n) / 2) ** 2 + ((z_dir - n) / 2) ** 2)
#
#             # fig, ax = plt.subplots(figsize=(10, 10))
#             fig = plt.figure(figsize=(10, 10))
#             ax = fig.add_subplot(projection='3d')
#             q = ax.quiver(x, y, z, x_dir, y_dir, z_dir, length=2.0, lw=2)
#             # q.set_array(np.random.rand(np.prod(x.shape)))
#             ax.xaxis.set_ticks([])
#             ax.yaxis.set_ticks([])
#             ax.zaxis.set_ticks([])
#             ax.axis([0, 1., 0, 1.])
#             ax.set_aspect('equal')
#
#             fig.savefig('gauge_fields_{}.png'.format(iteration))
#             plt.figure().clear()
#             plt.close('all')
#             plt.cla()
#             plt.clf()
#             # 1/0
#
#         output["target_gauge"] = target_xy
#         target_xy, target_yz, target_xz = target_xy.unsqueeze(0).unsqueeze(2), target_yz.unsqueeze(0).unsqueeze(2), target_xz.unsqueeze(0).unsqueeze(2)
#         output['source_gauge_feat'] = xyz_sampled.detach()
#
#         xy_feat = F.grid_sample(self.plane_xy, target_xy, align_corners=True).view(-1, *xyz_sampled.shape[:1])
#         xy_feat = xy_feat.permute(1, 0)
#         yz_feat = F.grid_sample(self.plane_yz, target_yz, align_corners=True).view(-1, *xyz_sampled.shape[:1])
#         yz_feat = yz_feat.permute(1, 0)
#         xz_feat = F.grid_sample(self.plane_xz, target_xz, align_corners=True).view(-1, *xyz_sampled.shape[:1])
#         xz_feat = xz_feat.permute(1, 0)
#         xyz_feat_density = torch.cat([xy_feat[:, :16], yz_feat[:, :16], xz_feat[:, :16]], dim=-1)
#         xyz_feat_rgb = torch.cat([xy_feat[:, 16:], yz_feat[:, 16:], xz_feat[:, 16:]], dim=-1)
#
#         density_feat = self.density_decoder(xyz_feat_density).reshape(-1)
#         density = self.feature2density(density_feat)
#         rgb = self.rgb_decoder(xyz_feat_rgb, view_sampled)
#
#         return density, rgb, output
#
#     def up_sampling_plane(self):
#
#         # for i in range(len(self.vecMode)):
#         #     vec_id = self.vecMode[i]
#         #     mat_id_0, mat_id_1 = self.matMode[i]
#         self.plane_xy = torch.nn.Parameter(
#             F.interpolate(self.plane_xy.data, size=(128, 128), mode='bilinear', align_corners=True))
#         self.plane_yz = torch.nn.Parameter(
#             F.interpolate(self.plane_yz.data, size=(128, 128), mode='bilinear', align_corners=True))
#         self.plane_xz = torch.nn.Parameter(
#             F.interpolate(self.plane_xz.data, size=(128, 128), mode='bilinear', align_corners=True))
#         # return plane_coef