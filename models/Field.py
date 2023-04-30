from .FieldBase import *
from .networks import *
import torch.nn as nn
from torch import einsum
import matplotlib.pyplot as plt
import torch.nn.functional as F
from geomloss import SamplesLoss


class TensorNGFinfo(Base):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorNGFinfo, self).__init__(aabb, gridSize, device, **kargs)


    def init_model(self, code_num=256, code_dim=128, scale=0.1, device=None):
        self.device = device

        # self.codebook = torch.nn.Parameter(scale * torch.randn((code_num, code_dim))).to(device)
        self.codebook1 = nn.Embedding(code_num, code_dim).to(device)
        self.codebook1.weight.data.uniform_(-1.0 / code_num, 1.0 / code_num)

        self.codebook2 = nn.Embedding(code_num, code_dim).to(device)
        self.codebook2.weight.data.uniform_(-1.0 / code_num, 1.0 / code_num)

        density_decode = []
        density_decode.append(nn.Linear(code_dim*2, code_dim*2))
        density_decode.append(nn.LeakyReLU(0.2))
        density_decode.append(nn.Linear(code_dim*2, 1))
        self.density_decode = nn.Sequential(*density_decode).to(device)

        x = np.linspace(0., 1., self.res)
        y = np.linspace(0., 1., self.res)
        z = np.linspace(0., 1., self.res)
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        xyz_grid1 = np.concatenate(
            (np.expand_dims(x_grid, axis=-1), np.expand_dims(y_grid, axis=-1), np.expand_dims(z_grid, axis=-1)), axis=-1)
        self.xyz_grid1 = torch.from_numpy(xyz_grid1).float().to(self.device).detach()

        x = np.linspace(0., 1., self.res*2)
        y = np.linspace(0., 1., self.res*2)
        z = np.linspace(0., 1., self.res*2)
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        xyz_grid2 = np.concatenate(
            (np.expand_dims(x_grid, axis=-1), np.expand_dims(y_grid, axis=-1), np.expand_dims(z_grid, axis=-1)), axis=-1)
        self.xyz_grid2 = torch.from_numpy(xyz_grid2).float().to(self.device).detach()

        self.gauge_transform1 = ray_bending2().to(device)
        self.gauge_transform2 = ray_bending2().to(device)
        self.inverse_gauge = Mapping2Dto3D().to(device)

        # self.Indexto3D = Indexto3D().to(device)


    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [{'params': self.codebook1.parameters(), 'lr': lr_init_spatialxyz},
                     {'params': self.codebook2.parameters(), 'lr': lr_init_spatialxyz},
                     {'params': self.gauge_transform1.parameters(), 'lr': lr_init_network},
                     {'params': self.gauge_transform2.parameters(), 'lr': lr_init_network},
                     {'params': self.density_decode.parameters(), 'lr': lr_init_network},
                     {'params': self.inverse_gauge.parameters(), 'lr': lr_init_network},
                     ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_init_network}]
        return grad_vars

    # def differentiable_topk(self, logit, k, draw=False, layer=0):
    #     # *_, n = x.shape
    #     x = logit.clone()
    #     topk_tensors = torch.zeros_like(x)
    #
    #     index_ls = []
    #     values_sum = 1e-8
    #
    #     for i in range(k):
    #         is_last = i == (k - 1)
    #         values, indices = x.topk(1, dim=-1)
    #         # print('***', values.shape, indices.shape)
    #         # 1/0
    #         # indices = x.max(-1, keepdim=True)[1]
    #         index_ls.append(indices.unsqueeze(-1))
    #
    #         values_sum = values_sum + values
    #
    #         topks = torch.zeros_like(x).scatter_(-1, indices, values)
    #         topk_tensors = topk_tensors + topks
    #         if not is_last:
    #             x.scatter_(-1, indices, float('-inf'))
    #
    #     ret = topk_tensors / values_sum - logit.detach() + logit
    #
    #     if draw:
    #         index_ls = torch.cat(index_ls, dim=-1)
    #         tmp = index_ls.cpu().detach().numpy().reshape((-1))
    #
    #         # plt.figure()
    #         # plt.style.use('seaborn-white')
    #         # kwargs = dict(edgecolor='grey', alpha=0.5, bins=256, color='lime', log=True)
    #         # plt.hist(tmp, **kwargs)
    #         # plt.savefig("/CT/Multimodal-NeRF/work/NGF/tmp.png", bbox_inches='tight')
    #         # plt.close()
    #
    #         print('***unique***', layer, np.unique(tmp).shape)
    #         # 1/0
    #     return ret

    def differentiable_topk(self, logit, k, draw=False, layer=0):
        # *_, n = x.shape
        x = logit.clone()

        index_ls = []
        values, indices = x.topk(256, dim=-1)
        index_ls.append(indices.unsqueeze(-1))

        values_sum = values.sum(-1, keepdim=True)
        # print('****', values_sum.shape)

        topk_tensors = torch.zeros_like(x).scatter_(-1, indices, values)

        ret = topk_tensors / values_sum - logit.detach() + logit

        # print('*******', topk_tensors)
        # 1/0

        return ret


    def compute_feature(self, xyz_sampled):

        logit1 = self.gauge_transform1(xyz_sampled)
        logit2 = self.gauge_transform2(xyz_sampled)

        # topk1_w = self.differentiable_topk(logit1, 1, draw=False, layer=1)
        # topk2_w = self.differentiable_topk(logit2, 1, draw=False, layer=2)

        topk1_w = logit1
        topk2_w = logit2

        # plt.figure()
        # plt.style.use('seaborn-white')
        # dis = logit1.sum(0).sum(0).sum(0)
        # y_axis = dis.cpu().detach().numpy().reshape((-1))
        # # y_axis = np.clip(y_axis, 0, 50)
        # # y_axis = np.log10(y_axis)
        # # kwargs = dict(edgecolor='k', alpha=0.3, bins=256, color='blue', log=True)
        # # plt.hist(tmp, **kwargs)
        # x_axis = np.linspace(0, 256, 256)
        # kwargs = dict(edgecolor='grey', alpha=0.5, color='lime', log=True, width=1.0)
        # # kwargs = dict(edgecolor='grey', alpha=0.3, color='dodgerblue', log=True, width=1.0)
        # # edgecolor = 'grey', alpha = 0.3, bins = 256, color = 'dodgerblue'
        #
        # plt.bar(x_axis, y_axis, **kwargs)
        # plt.savefig("/CT/Multimodal-NeRF/work/NGF/tmp_hybrid.png", bbox_inches='tight')
        # plt.close()
        # 1/0

        sampled_vector1 = einsum('n c, c d -> n d', topk1_w, self.codebook1.weight) #.unsqueeze(0).permute(0, 4, 1, 2, 3)
        sampled_vector2 = einsum('n c, c d -> n d', topk2_w, self.codebook2.weight) #.unsqueeze(0).permute(0, 4, 1, 2, 3)
        # z_q = einsum('b n h w, n d -> b d h w', logit1, self.codebook1.weight)

        sampled_vector = torch.concat([sampled_vector1, sampled_vector2], dim=-1)
        density_feature = self.density_decode(sampled_vector).reshape(-1)

        reg_loss = xyz_sampled.sum(-1) * 0

        # 1/0
        # print(xyz_sampled.shape)

        # logit1 = self.gauge_transform1(self.xyz_grid1)
        # logit2 = self.gauge_transform2(self.xyz_grid2)
        #
        # p_output1 = logit1.sum(0).sum(0).sum(0) / float(16*16*16)
        # q_output = (torch.ones(256) / float(256)).float().to(self.device).detach()
        # js_loss1 = ((p_output1 - q_output) ** 2).sum() * 1e4
        #
        # p_output2 = logit2.sum(0).sum(0).sum(0) / float(32*32*32)
        # js_loss2 = ((p_output2 - q_output) ** 2).sum() * 1e4
        # reg_loss = js_loss1
        # reg_loss = reg_loss + js_loss2


        return density_feature, sampled_vector, reg_loss


class TensorNGF2(Base):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorNGF2, self).__init__(aabb, gridSize, device, **kargs)


    def init_model(self, code_num=256, code_dim=128, scale=0.1, device=None):
        self.device = device

        # self.codebook = torch.nn.Parameter(scale * torch.randn((code_num, code_dim))).to(device)
        self.codebook1 = nn.Embedding(code_num, code_dim).to(device)
        self.codebook1.weight.data.uniform_(-1.0 / code_num, 1.0 / code_num)

        self.codebook2 = nn.Embedding(code_num, code_dim).to(device)
        self.codebook2.weight.data.uniform_(-1.0 / code_num, 1.0 / code_num)

        density_decode = []
        density_decode.append(nn.Linear(code_dim*2, code_dim*2))
        density_decode.append(nn.LeakyReLU(0.2))
        density_decode.append(nn.Linear(code_dim*2, 1))
        self.density_decode = nn.Sequential(*density_decode).to(device)

        self.gauge_transform1 = ray_bending2().to(device)
        self.gauge_transform2 = ray_bending2().to(device)


    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [{'params': self.codebook1.parameters(), 'lr': lr_init_spatialxyz},
                     {'params': self.codebook2.parameters(), 'lr': lr_init_spatialxyz},
                     {'params': self.gauge_transform1.parameters(), 'lr': lr_init_network},
                     {'params': self.gauge_transform2.parameters(), 'lr': lr_init_network},
                     {'params': self.density_decode.parameters(), 'lr': lr_init_network},]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_init_network}]
        return grad_vars


    def compute_feature(self, xyz_sampled):

        logit1 = self.gauge_transform1(xyz_sampled)
        logit2 = self.gauge_transform2(xyz_sampled)

        # plt.figure()
        # plt.style.use('seaborn-white')
        # dis = logit1[:4096].sum(0)
        # y_axis = dis.cpu().detach().numpy().reshape((-1))
        # y_axis = np.clip(y_axis, 0, 50)
        # # y_axis = np.log10(y_axis)
        # # kwargs = dict(edgecolor='k', alpha=0.3, bins=256, color='blue', log=True)
        # # plt.hist(tmp, **kwargs)
        # x_axis = np.linspace(0, 256, 256)
        # kwargs = dict(edgecolor='grey', alpha=0.3, color='dodgerblue', log=True, width=1.0)
        # # edgecolor = 'grey', alpha = 0.3, bins = 256, color = 'dodgerblue'
        #
        # plt.bar(x_axis, y_axis, **kwargs)
        # plt.savefig("/CT/Multimodal-NeRF/work/NGF/tmp_hybrid.png", bbox_inches='tight')
        # plt.close()
        # 1/0

        sampled_vector1 = einsum('n c, c d -> n d', logit1, self.codebook1.weight) #.unsqueeze(0).permute(0, 4, 1, 2, 3)
        sampled_vector2 = einsum('n c, c d -> n d', logit2, self.codebook2.weight) #.unsqueeze(0).permute(0, 4, 1, 2, 3)
        # z_q = einsum('b n h w, n d -> b d h w', logit1, self.codebook1.weight)

        sampled_vector = torch.concat([sampled_vector1, sampled_vector2], dim=-1)
        density_feature = self.density_decode(sampled_vector).reshape(-1)

        reg_loss = xyz_sampled.sum(-1) * 0

        return density_feature, sampled_vector, reg_loss



class TensorNGF1reg(Base):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorNGF1reg, self).__init__(aabb, gridSize, device, **kargs)


    def init_model(self, code_num=256, code_dim=128, scale=0.1, device=None):
        self.device = device

        # self.codebook = torch.nn.Parameter(scale * torch.randn((code_num, code_dim))).to(device)
        self.codebook1 = nn.Embedding(code_num, code_dim).to(device)
        self.codebook1.weight.data.uniform_(-1.0 / code_num, 1.0 / code_num)

        self.codebook2 = nn.Embedding(code_num, code_dim).to(device)
        self.codebook2.weight.data.uniform_(-1.0 / code_num, 1.0 / code_num)

        density_decode = []
        density_decode.append(nn.Linear(code_dim*2, code_dim*2))
        density_decode.append(nn.LeakyReLU(0.2))
        density_decode.append(nn.Linear(code_dim*2, 1))
        self.density_decode = nn.Sequential(*density_decode).to(device)

        x = np.linspace(0., 1., self.res)
        y = np.linspace(0., 1., self.res)
        z = np.linspace(0., 1., self.res)
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        xyz_grid1 = np.concatenate(
            (np.expand_dims(x_grid, axis=-1), np.expand_dims(y_grid, axis=-1), np.expand_dims(z_grid, axis=-1)), axis=-1)
        self.xyz_grid1 = torch.from_numpy(xyz_grid1).float().to(self.device).detach()

        x = np.linspace(0., 1., self.res*2)
        y = np.linspace(0., 1., self.res*2)
        z = np.linspace(0., 1., self.res*2)
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        xyz_grid2 = np.concatenate(
            (np.expand_dims(x_grid, axis=-1), np.expand_dims(y_grid, axis=-1), np.expand_dims(z_grid, axis=-1)), axis=-1)
        self.xyz_grid2 = torch.from_numpy(xyz_grid2).float().to(self.device).detach()

        self.gauge_transform1 = discrete_transform().to(device)
        self.gauge_transform2 = discrete_transform().to(device)

        # self.Indexto3D = Indexto3D().to(device)


    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [{'params': self.codebook1.parameters(), 'lr': lr_init_spatialxyz},
                     {'params': self.codebook2.parameters(), 'lr': lr_init_spatialxyz},
                     {'params': self.gauge_transform1.parameters(), 'lr': lr_init_network},
                     {'params': self.gauge_transform2.parameters(), 'lr': lr_init_network},
                     {'params': self.density_decode.parameters(), 'lr': lr_init_network},
                     {'params': self.inverse_gauge.parameters(), 'lr': lr_init_network},
                     ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_init_network}]
        return grad_vars

    def argmax_softmax(self, logits, tau=1, dim=-1, draw=False):
        # logits = logits / tau
        # y_soft = logits.softmax(dim)
        y_soft = logits
        index = y_soft.max(dim, keepdim=True)[1]
        # tmp = index.cpu().detach().numpy().reshape((-1))
        # print('***unique***', np.unique(tmp).shape)

        if draw:
            plt.figure()
            plt.style.use('seaborn-white')
            # print('**********', index, index.shape)
            tmp = index.cpu().detach().numpy().reshape((-1))
            print('***unique***', np.unique(tmp).shape)
            kwargs = dict(edgecolor='grey', alpha=0.3, bins=256, color='blue', log=True)
            plt.hist(tmp, **kwargs)
            plt.savefig("/CT/Multimodal-NeRF/work/NGF/tmp.png", bbox_inches='tight')
            plt.close()
            # 1/0

        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret

    def js_div(self, N, p_output, q_output=None, get_softmax=True):
        """
        Function that measures JS divergence between target and output logits:
        """
        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        # if get_softmax:
        #     p_output = F.softmax(p_output)
        #     q_output = F.softmax(q_output)
        # q_output = np.linspace(0., 1., self.res*2)
        q_output = (torch.ones(N) / float(N)).float().to(self.device).detach()
        # print('******', p_output.sum(), q_output.sum(), p_output, q_output)
        log_mean_output = ((p_output + q_output) / 2).log()
        return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2


    def compute_feature(self, xyz_sampled):

        # print('********', self.xyz_grid1.type())
        input_points = xyz_sampled.unsqueeze(0).unsqueeze(2).unsqueeze(3).detach()

        logit1 = self.gauge_transform1(self.xyz_grid1)
        one_hot1 = self.argmax_softmax(logit1)
        logit2 = self.gauge_transform2(self.xyz_grid2)
        one_hot2 = self.argmax_softmax(logit2)  #self.argmax_softmax(logits, tau=temp, dim=1)

        # regularization
        inverse_3d = self.inverse_gauge(one_hot1)
        reg_loss = ((self.xyz_grid1 - inverse_3d) ** 2).sum(-1).mean() * 0.0

        p_output1 = logit1.sum(0).sum(0).sum(0) / (16*16*16)
        # js_loss1 = self.js_div(256, p_output1) * 1e7
        q_output = (torch.ones(256) / float(256)).float().to(self.device).detach()
        js_loss1 = ((p_output1 - q_output) ** 2).sum() * 1e3

        p_output2 = logit2.sum(0).sum(0).sum(0) / (32*32*32)
        js_loss2 = ((p_output2 - q_output) ** 2).sum() * 1e3
        # js_loss2 = self.js_div(256, p_output2) * 1e7
        # print('******js_loss*******', js_loss1, js_loss2)

        reg_loss += js_loss1
        reg_loss += js_loss2


        code_vector1 = einsum('x y z n, n d -> x y z d', one_hot1, self.codebook1.weight).unsqueeze(0).permute(0, 4, 1, 2, 3)
        code_vector2 = einsum('x y z n, n d -> x y z d', one_hot2, self.codebook2.weight).unsqueeze(0).permute(0, 4, 1, 2, 3)
        # z_q = einsum('b n h w, n d -> b d h w', logit1, self.codebook1.weight)

        # print('********', code_vector1.shape)
        # 1/0

        sampled_vector1 = F.grid_sample(code_vector1, input_points, align_corners=True)
        sampled_vector1 = sampled_vector1.squeeze().permute(1, 0)

        sampled_vector2 = F.grid_sample(code_vector2, input_points, align_corners=True)
        sampled_vector2 = sampled_vector2.squeeze().permute(1, 0)

        sampled_vector = torch.concat([sampled_vector1, sampled_vector2], dim=-1)

        density_feature = self.density_decode(sampled_vector).reshape(-1)

        return density_feature, sampled_vector, reg_loss



class TensorNGF1(Base):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorNGF1, self).__init__(aabb, gridSize, device, **kargs)


    def init_model(self, code_num=256, code_dim=128, scale=0.1, device=None):
        self.device = device

        # self.codebook = torch.nn.Parameter(scale * torch.randn((code_num, code_dim))).to(device)
        self.codebook1 = nn.Embedding(code_num, code_dim).to(device)
        self.codebook1.weight.data.uniform_(-1.0 / code_num, 1.0 / code_num)

        self.codebook2 = nn.Embedding(code_num, code_dim).to(device)
        self.codebook2.weight.data.uniform_(-1.0 / code_num, 1.0 / code_num)

        density_decode = []
        density_decode.append(nn.Linear(code_dim*2, code_dim*2))
        density_decode.append(nn.LeakyReLU(0.2))
        density_decode.append(nn.Linear(code_dim*2, 1))
        self.density_decode = nn.Sequential(*density_decode).to(device)

        x = np.linspace(0., 1., self.res)
        y = np.linspace(0., 1., self.res)
        z = np.linspace(0., 1., self.res)
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        xyz_grid1 = np.concatenate(
            (np.expand_dims(x_grid, axis=-1), np.expand_dims(y_grid, axis=-1), np.expand_dims(z_grid, axis=-1)), axis=-1)
        self.xyz_grid1 = torch.from_numpy(xyz_grid1).float().to(self.device).detach()

        x = np.linspace(0., 1., self.res*2)
        y = np.linspace(0., 1., self.res*2)
        z = np.linspace(0., 1., self.res*2)
        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        xyz_grid2 = np.concatenate(
            (np.expand_dims(x_grid, axis=-1), np.expand_dims(y_grid, axis=-1), np.expand_dims(z_grid, axis=-1)), axis=-1)
        self.xyz_grid2 = torch.from_numpy(xyz_grid2).float().to(self.device).detach()

        self.gauge_transform1 = ray_bending2().to(device)
        self.gauge_transform2 = ray_bending2().to(device)


    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [{'params': self.codebook1.parameters(), 'lr': lr_init_spatialxyz},
                     {'params': self.codebook2.parameters(), 'lr': lr_init_spatialxyz},
                     {'params': self.gauge_transform1.parameters(), 'lr': lr_init_network},
                     {'params': self.gauge_transform2.parameters(), 'lr': lr_init_network},
                     {'params': self.density_decode.parameters(), 'lr': lr_init_network},]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_init_network}]
        return grad_vars

    def argmax_softmax(self, logits, tau=1, dim=-1, draw=False):
        # logits = logits / tau
        # y_soft = logits.softmax(dim)
        y_soft = logits
        index = y_soft.max(dim, keepdim=True)[1]

        if draw:
            plt.figure()
            plt.style.use('seaborn-white')
            # print('**********', index, index.shape)
            tmp = index.cpu().detach().numpy().reshape((-1))
            print('***unique***', np.unique(tmp).shape)
            # kwargs = dict(edgecolor='k', alpha=0.3, bins=256, color='blue', log=True)
            kwargs = dict(edgecolor='grey', alpha=0.3, bins=256, color='blue', log=True)
            plt.hist(tmp, **kwargs)
            # plt.yscale('log')
            # plt.show()
            plt.savefig("/CT/Multimodal-NeRF/work/NGF/tmp.png", bbox_inches='tight')
            plt.close()
            1/0

        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret


    def compute_feature(self, xyz_sampled):

        # print('********', self.xyz_grid1.type())
        xyz_sampled = xyz_sampled.unsqueeze(0).unsqueeze(2).unsqueeze(3).detach()

        logit1 = self.gauge_transform1(self.xyz_grid1)
        one_hot1 = self.argmax_softmax(logit1, draw=True)
        logit2 = self.gauge_transform2(self.xyz_grid2)
        one_hot2 = self.argmax_softmax(logit2)  #self.argmax_softmax(logits, tau=temp, dim=1)

        # 1/0

        code_vector1 = einsum('x y z n, n d -> x y z d', one_hot1, self.codebook1.weight).unsqueeze(0).permute(0, 4, 1, 2, 3)
        code_vector2 = einsum('x y z n, n d -> x y z d', one_hot2, self.codebook2.weight).unsqueeze(0).permute(0, 4, 1, 2, 3)
        # z_q = einsum('b n h w, n d -> b d h w', logit1, self.codebook1.weight)

        # print('********', code_vector1.shape)
        # 1/0

        sampled_vector1 = F.grid_sample(code_vector1, xyz_sampled, align_corners=True)
        sampled_vector1 = sampled_vector1.squeeze().permute(1, 0)

        sampled_vector2 = F.grid_sample(code_vector2, xyz_sampled, align_corners=True)
        sampled_vector2 = sampled_vector2.squeeze().permute(1, 0)

        sampled_vector = torch.concat([sampled_vector1, sampled_vector2], dim=-1)

        density_feature = self.density_decode(sampled_vector).reshape(-1)

        return density_feature, sampled_vector, None



class MlpNGF(Base):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(MlpNGF, self).__init__(aabb, gridSize, device, **kargs)


    def init_model(self, code_num=256, code_dim=128, scale=0.1, device=None):
        self.device = device

        input_dim, width, layers, num_freqs = 2, 256, 5, 10
        block = []
        block.append(nn.Linear(input_dim + 2 * input_dim * num_freqs, width))
        for i in range(layers):
            block.append(nn.Linear(width, width))
            block.append(nn.LeakyReLU(0.2))
        block.append(nn.Linear(width, width))
        self.rgb_decode = nn.Sequential(*block).to(device)
        init_seq(self.rgb_decode)

        input_dim, width, layers, num_freqs = 3, 256, 10, 10
        block = []
        block.append(nn.Linear(input_dim + 2 * input_dim * num_freqs, width))
        for i in range(layers):
            block.append(nn.Linear(width, width))
            block.append(nn.ReLU())
        block.append(nn.Linear(width, 1))
        self.density_decode = nn.Sequential(*block).to(device)
        init_seq(self.density_decode)

        self.continuous_gauge_transform = continuous_gauge().to(device)
        self.inverse_gauge = Mapping2Dto3D().to(device)


    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [
                     {'params': self.continuous_gauge_transform.parameters(), 'lr': lr_init_network},
                     {'params': self.inverse_gauge.parameters(), 'lr': lr_init_network},
                     {'params': self.rgb_decode.parameters(), 'lr': lr_init_network},
                     {'params': self.density_decode.parameters(), 'lr': lr_init_network},]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_init_network}]
        return grad_vars


    def compute_feature(self, xyz_sampled):
        # print('****', xyz_sampled.shape)
        # 1/0

        uv = self.continuous_gauge_transform(xyz_sampled[:, :2])

        # inverse_3d = self.inverse_gauge(uv)
        # reg_loss = ((xyz_sampled.detach() - inverse_3d) ** 2).sum(-1) * 100

        rgb_feature = self.rgb_decode(torch.cat([uv, positional_encoding(uv, 10)], dim=-1))
        # rgb_feature = self.rgb_decode(uv)
        density_feature = self.density_decode(torch.cat([xyz_sampled, positional_encoding(xyz_sampled, 10)], dim=-1))

        # plt.figure()
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False
        # plt.title("UV Samples", fontsize=8)
        # plt.xlim(xmax=1, xmin=-1)
        # plt.ylim(ymax=1, ymin=-1)
        # uv = uv[::10].detach().cpu().numpy()
        # plt.plot(uv[:, 0], uv[:, 1], markersize=4., color=(0.8, 0., 0.))
        # plt.savefig(f'/CT/Multimodal-NeRF/work/NGF/tmp.png')

        reg_loss = xyz_sampled.sum(-1) * 0


        return density_feature.reshape(-1), rgb_feature, reg_loss

    def compute_uv(self, uv, viewdir):
        rgb_feature = self.rgb_decode(torch.cat([uv, positional_encoding(uv, 10)], dim=-1))
        valid_rgbs = self.renderModule(uv, viewdir, rgb_feature)
        return valid_rgbs







class TensorTriplane(Base):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorTriplane, self).__init__(aabb, gridSize, device, **kargs)

    def init_svd_volume(self, res, device):

        torch.nn.Parameter(
            scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])))

        # self.plane = self.init_one_svd(self.n_comp[0], self.gridSize, 0.1, device)
        self.density_decode = torch.nn.Linear(self.app_dim, 1, bias=False).to(device)
        # self.rgb_decode = torch.nn.Linear(self.n_comp[0], self.app_dim, bias=False).to(device)

        block1 = []
        width = 256
        block1.append(nn.Linear(2, width))
        block1.append(nn.LeakyReLU(0.2))
        for i in range(5):
            block1.append(nn.Linear(width, width))
            block1.append(nn.LeakyReLU(0.2))
        block1.append(nn.Linear(width, self.app_dim))
        self.block1 = nn.Sequential(*block1).to(device)

        self.gauge_transform1 = ray_bending2().to(device)
        self.gauge_transform2 = ray_bending2().to(device)


    # def init_one_svd(self, n_comp, gridSize, scale, device):
    #     plane_coef, line_coef = [], []
    #     for i in range(len(self.vecMode)):
    #         mat_id_0, mat_id_1 = self.matMode[i]
    #         plane_coef.append(torch.nn.Parameter(
    #             scale * torch.randn((1, n_comp, gridSize[mat_id_1], gridSize[mat_id_0]))))
    #
    #     return torch.nn.ParameterList(plane_coef).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [
                     # {'params': self.plane, 'lr': lr_init_spatialxyz},
                     {'params': self.density_decode.parameters(), 'lr': lr_init_network},
                     {'params': self.block1.parameters(), 'lr': lr_init_network},

                     {'params': self.gauge_transform0.parameters(), 'lr': lr_init_network},
                     {'params': self.gauge_transform1.parameters(), 'lr': lr_init_network},
                     {'params': self.gauge_transform2.parameters(), 'lr': lr_init_network},

                     {'params': self.inverse_gauge0.parameters(), 'lr': lr_init_network},
                     {'params': self.inverse_gauge1.parameters(), 'lr': lr_init_network},
                     {'params': self.inverse_gauge2.parameters(), 'lr': lr_init_network},
                     ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_init_network}]
        return grad_vars

    def compute_feature_mlp(self, xyz_sampled):

        uv_sampled = self.gauge_transform0(xyz_sampled.view(-1, 3), self.matMode[0])
        # print(xyz_sampled.shape)
        # uv_sampled = xyz_sampled[..., [0, 1]] + uv_sampled_res
        uv_feature = self.block1(uv_sampled)

        # coordinate_plane = self.matMode[0]
        density_feature = self.density_decode(uv_feature).reshape(-1)
        # rgb_feature = self.rgb_decode(sigma_feature) # torch.Size([985242, 27])
        # print('*****uv:', uv_sampled.min(), uv_sampled.max())
        return density_feature, uv_feature, uv_sampled


# Tensor Plane:  Triplane_Tensor TrilaneTensor MLP_Plane Tensor_Plane SphereMLP  PlaneMLP. MLP_Plane MLP_Plane.
class PlaneTensor(Base):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(PlaneTensor, self).__init__(aabb, gridSize, device, **kargs)

    def init_model(self, res=128, dim=8, scale=0.1, device=None):

        self.plane = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))
        self.rgb_decoder = rgb_decoder(feat_dim=dim, view_pe=2, middle_dim=64).to(device)
        self.density_decoder = torch.nn.Linear(dim, 1).to(device)
        init_weights(self.density_decoder, 'xavier_uniform')

        # block = []
        # block.append(nn.Linear(dim, 1))
        # block.append(nn.ReLU())
        # block.append(nn.Linear(width, 1))
        # self.density_decode = nn.Sequential(*block).to(device)
        # init_seq(self.density_decoder)

        self.gauge_network = continuous_transform().to(device)
        self.reg_network = continuous_reg(input_dim=2, output_dim=3).to(device)

        x, y = np.meshgrid(np.linspace(-0.9, 0.9, 32), np.linspace(-0.9, 0.9, 32), indexing="xy")
        grid = np.array(list(zip(x.ravel(), y.ravel())))
        self.sampled_prior = torch.from_numpy(grid).type(torch.FloatTensor).to(device)
        self.EMD = SamplesLoss("sinkhorn", p=2, blur=0.01)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [
                     {'params': self.plane, 'lr': lr_init_spatialxyz},
                     {'params': self.rgb_decoder.parameters(), 'lr': lr_init_network},
                     {'params': self.density_decoder.parameters(), 'lr': lr_init_network},

                     {'params': self.gauge_network.parameters(), 'lr': lr_init_network * 0.25},
                     {'params': self.reg_network.parameters(), 'lr': lr_init_network},
                     ]
        return grad_vars

    def feature2density(self, density_features, density_shift=-10):
        # return F.softplus(density_features)
        return F.softplus(density_features+density_shift)

    def compute_feature(self, xyz_sampled, view_sampled, iteration=0):
        output = {}
        # uv_transformed = xyz_sampled[:, :2]

        target_gauge, source_gauge_feat = self.gauge_network(xyz_sampled)
        # if iteration > self.reg_iter:
        #     target_gauge = target_gauge.detach()
        output["target_gauge"] = target_gauge
        target_gauge = target_gauge.unsqueeze(0).unsqueeze(2) # N, H*W, 1, 2
        output['source_gauge_feat'] = xyz_sampled.detach()

        uv_feat = F.grid_sample(self.plane, target_gauge, align_corners=True).view(-1, *xyz_sampled.shape[:1])
        uv_feat = uv_feat.permute(1, 0)
        # uv_feat = torch.cat([uv_feat, target_gauge.squeeze()], dim=-1)

        density_feat = self.density_decoder(uv_feat).reshape(-1)
        density = self.feature2density(density_feat)
        rgb = self.rgb_decoder(uv_feat, view_sampled)

        return density, rgb, output



# # Tensor Plane:  Triplane_Tensor TrilaneTensor MLP_Plane Tensor_Plane SphereMLP  PlaneMLP. MLP_Plane MLP_Plane.
# class PlaneMLP(Base):
#     def __init__(self, aabb, gridSize, device, **kargs):
#         super(PlaneMLP, self).__init__(aabb, gridSize, device, **kargs)
#
#     def init_model(self, res=None, dim=None, scale=None, device=None):
#
#         # self.plane = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))
#
#         input_dim, width, layers, num_freqs = 2, 256, 5, 10
#         block = []
#         block.append(nn.Linear(input_dim + 2 * input_dim * num_freqs, width))
#         for i in range(layers):
#             block.append(nn.Linear(width, width))
#             block.append(nn.LeakyReLU(0.2))
#         block.append(nn.Linear(width, width))
#         self.backbone = nn.Sequential(*block).to(device)
#         init_seq(self.backbone)
#
#         # input_dim = 2
#         # block = []
#         # block.append(nn.Linear(input_dim + 2 * input_dim * num_freqs, width))
#         # for i in range(layers):
#         #     block.append(nn.Linear(width, width))
#         #     block.append(nn.LeakyReLU(0.2))
#         # block.append(nn.Linear(width, width))
#         # self.density_backbone = nn.Sequential(*block).to(device)
#         # init_seq(self.density_backbone)
#
#         self.rgb_decoder = rgb_decoder(feat_dim=width, view_pe=2, middle_dim=128).to(device)
#         self.density_decoder = torch.nn.Linear(width, 1).to(device)
#         init_weights(self.density_decoder, 'xavier_uniform')
#
#         self.gauge_network = continuous_transform().to(device)
#         self.reg_network = continuous_reg(input_dim=2, output_dim=3).to(device)
#
#         x, y = np.meshgrid(np.linspace(-0.9, 0.9, 32), np.linspace(-0.9, 0.9, 32), indexing="xy")
#         grid = np.array(list(zip(x.ravel(), y.ravel())))
#         self.sampled_prior = torch.from_numpy(grid).type(torch.FloatTensor).to(device)
#         self.EMD = SamplesLoss("sinkhorn", p=2, blur=0.01)
#
#     def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
#         grad_vars = [
#                      {'params': self.backbone.parameters(), 'lr': lr_init_network},
#                      {'params': self.rgb_decoder.parameters(), 'lr': lr_init_network},
#                      {'params': self.density_decoder.parameters(), 'lr': lr_init_network},
#
#                      {'params': self.gauge_network.parameters(), 'lr': lr_init_network*0.25},
#                      {'params': self.reg_network.parameters(), 'lr': lr_init_network},
#                      ]
#         return grad_vars
#
#     def feature2density(self, density_features, density_shift=-10):
#         # return F.softplus(density_features)
#         return F.softplus(density_features+density_shift)
#
#     def compute_feature(self, xyz_sampled, view_sampled, iteration=0):
#         output = {}
#         # uv_transformed = xyz_sampled[:, :2]
#
#         target_gauge, source_gauge_feat = self.gauge_network(xyz_sampled)
#         output["target_gauge"] = target_gauge
#         output['source_gauge_feat'] = xyz_sampled.detach()
#
#         uv_feat = self.backbone(torch.cat([target_gauge, positional_encoding(target_gauge, 10)], dim=-1))
#
#         rgb = self.rgb_decoder(uv_feat, view_sampled)
#         # density_feat = self.density_backbone(torch.cat([xyz_sampled, positional_encoding(xyz_sampled, 10)], dim=-1))
#         density_feat = self.density_decoder(uv_feat).reshape(-1)
#         density = self.feature2density(density_feat)
#
#         return density, rgb, output
#
#     def compute_uv(self, uv, viewdir):
#         uv_feat = self.backbone(torch.cat([uv, positional_encoding(uv, 10)], dim=-1))
#         rgb = self.rgb_decoder(uv_feat, viewdir)
#         # valid_rgbs = self.renderModule(uv, viewdir, rgb_feature)
#         return rgb



# Tensor Plane:  Triplane_Tensor TrilaneTensor MLP_Plane Tensor_Plane SphereMLP  PlaneMLP. MLP_Plane MLP_Plane.
class PlaneMLP(Base):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(PlaneMLP, self).__init__(aabb, gridSize, device, **kargs)

    def init_model(self, res=None, dim=None, scale=None, device=None):

        # self.plane = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))

        input_dim, width, layers, num_freqs = 2, 256, 5, 10
        block = []
        block.append(nn.Linear(input_dim + 2 * input_dim * num_freqs, width))
        for i in range(layers):
            block.append(nn.Linear(width, width))
            block.append(nn.LeakyReLU(0.2))
        block.append(nn.Linear(width, width))
        self.rgb_backbone = nn.Sequential(*block).to(device)
        init_seq(self.rgb_backbone)

        input_dim = 3
        block = []
        block.append(nn.Linear(input_dim + 2 * input_dim * num_freqs, width))
        for i in range(layers):
            block.append(nn.Linear(width, width))
            block.append(nn.LeakyReLU(0.2))
        block.append(nn.Linear(width, width))
        self.density_backbone = nn.Sequential(*block).to(device)
        init_seq(self.density_backbone)

        self.rgb_decoder = rgb_decoder(feat_dim=width, view_pe=2, middle_dim=128).to(device)
        self.density_decoder = torch.nn.Linear(width, 1).to(device)
        init_weights(self.density_decoder, 'xavier_uniform')

        self.gauge_network = continuous_transform().to(device)
        self.reg_network = continuous_reg(input_dim=2, output_dim=3).to(device)

        x, y = np.meshgrid(np.linspace(-0.9, 0.9, 32), np.linspace(-0.9, 0.9, 32), indexing="xy")
        grid = np.array(list(zip(x.ravel(), y.ravel())))
        self.sampled_prior = torch.from_numpy(grid).type(torch.FloatTensor).to(device)
        self.EMD = SamplesLoss("sinkhorn", p=2, blur=0.01)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [
                     {'params': self.rgb_backbone.parameters(), 'lr': lr_init_network},
                     {'params': self.rgb_decoder.parameters(), 'lr': lr_init_network},
                     {'params': self.density_backbone.parameters(), 'lr': lr_init_network},
                     {'params': self.density_decoder.parameters(), 'lr': lr_init_network},

                     {'params': self.gauge_network.parameters(), 'lr': lr_init_network*0.25},
                     {'params': self.reg_network.parameters(), 'lr': lr_init_network},
                     ]
        return grad_vars

    def feature2density(self, density_features, density_shift=-10):
        # return F.softplus(density_features)
        return F.softplus(density_features+density_shift)

    def compute_feature(self, xyz_sampled, view_sampled, iteration=0):
        output = {}
        # uv_transformed = xyz_sampled[:, :2]

        target_gauge, source_gauge_feat = self.gauge_network(xyz_sampled)
        output["target_gauge"] = target_gauge
        output['source_gauge_feat'] = xyz_sampled.detach()

        rgb_feat = self.rgb_backbone(torch.cat([target_gauge, positional_encoding(target_gauge, 10)], dim=-1))
        rgb = self.rgb_decoder(rgb_feat, view_sampled)

        density_feat = self.density_backbone(torch.cat([xyz_sampled, positional_encoding(xyz_sampled, 10)], dim=-1))
        density_feat = self.density_decoder(density_feat).reshape(-1)
        density = self.feature2density(density_feat)

        return density, rgb, output

    def compute_uv(self, uv, viewdir):
        rgb_feat = self.rgb_backbone(torch.cat([uv, positional_encoding(uv, 10)], dim=-1))
        rgb = self.rgb_decoder(rgb_feat, viewdir)
        # valid_rgbs = self.renderModule(uv, viewdir, rgb_feature)
        return rgb


class TriplaneNGF(Base):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TriplaneNGF, self).__init__(aabb, gridSize, device, **kargs)

    def init_model(self, res=256, dim=64, scale=0.1, device=None):

        self.plane_xy = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))
        self.plane_yz = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))
        self.plane_xz = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))

        self.rgb_decoder = rgb_decoder(feat_dim=48*3, view_pe=2, middle_dim=64).to(device)
        self.density_decoder = torch.nn.Linear(16*3, 1).to(device)
        init_weights(self.density_decoder, 'xavier_uniform')

        self.gauge_network = continuous_transform(input_dim=6, output_dim=2).to(device) # planes: [1,0,0], [0,1,0], [0,0,1].
        self.reg_network = continuous_reg(input_dim=2, output_dim=3).to(device)

        x, y = np.meshgrid(np.linspace(-0.9, 0.9, 32), np.linspace(-0.9, 0.9, 32), indexing="xy")
        grid = np.array(list(zip(x.ravel(), y.ravel())))
        self.sampled_prior = torch.from_numpy(grid).type(torch.FloatTensor).to(device)
        self.EMD = SamplesLoss("sinkhorn", p=2, blur=0.01)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [
                     {'params': self.plane_xy, 'lr': lr_init_spatialxyz},
                     {'params': self.plane_yz, 'lr': lr_init_spatialxyz},
                     {'params': self.plane_xz, 'lr': lr_init_spatialxyz},
                     {'params': self.rgb_decoder.parameters(), 'lr': lr_init_network},
                     {'params': self.density_decoder.parameters(), 'lr': lr_init_network},

                     {'params': self.gauge_network.parameters(), 'lr': lr_init_network * 0.25},
                     {'params': self.reg_network.parameters(), 'lr': lr_init_network},
                     ]
        return grad_vars

    def feature2density(self, density_features, density_shift=-10):
        # return F.softplus(density_features)
        return F.softplus(density_features+density_shift)
    

    def compute_feature(self, xyz_sampled, view_sampled, iteration=0):
        output = {}
        # uv_transformed = xyz_sampled[:, :2]
        N, _ = xyz_sampled.shape

        xy_idx, yz_idx, xz_idx = torch.tensor([[1, 0, 0]]), torch.tensor([[0, 1, 0]]), torch.tensor([[0, 0, 1]])
        xy_idx, yz_idx, xz_idx = xy_idx.repeat(N, 1).to(self.device), yz_idx.repeat(N, 1).to(self.device), xz_idx.repeat(N, 1).to(self.device)

        target_xy, _ = self.gauge_network(torch.cat([xy_idx, xyz_sampled], dim=-1))
        target_yz, _ = self.gauge_network(torch.cat([yz_idx, xyz_sampled], dim=-1))
        target_xz, _ = self.gauge_network(torch.cat([xz_idx, xyz_sampled], dim=-1))

        output["target_gauge"] = target_xy
        target_xy, target_yz, target_xz = target_xy.unsqueeze(0).unsqueeze(2), target_yz.unsqueeze(0).unsqueeze(2), target_xz.unsqueeze(0).unsqueeze(2)
        output['source_gauge_feat'] = xyz_sampled.detach()

        xy_feat = F.grid_sample(self.plane_xy, target_xy, align_corners=True).view(-1, *xyz_sampled.shape[:1])
        xy_feat = xy_feat.permute(1, 0)
        yz_feat = F.grid_sample(self.plane_yz, target_yz, align_corners=True).view(-1, *xyz_sampled.shape[:1])
        yz_feat = yz_feat.permute(1, 0)
        xz_feat = F.grid_sample(self.plane_xz, target_xz, align_corners=True).view(-1, *xyz_sampled.shape[:1])
        xz_feat = xz_feat.permute(1, 0)
        xyz_feat_density = torch.cat([xy_feat[:, :16], yz_feat[:, :16], xz_feat[:, :16]], dim=-1)
        xyz_feat_rgb = torch.cat([xy_feat[:, 16:], yz_feat[:, 16:], xz_feat[:, 16:]], dim=-1)

        density_feat = self.density_decoder(xyz_feat_density).reshape(-1)
        density = self.feature2density(density_feat)
        rgb = self.rgb_decoder(xyz_feat_rgb, view_sampled)

        return density, rgb, output


class TensoRF(Base):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensoRF, self).__init__(aabb, gridSize, device, **kargs)

    def init_model(self, res=300, dim=28, scale=0.1, device=None):

        self.plane_xy = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))
        self.plane_yz = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))
        self.plane_xz = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))

        self.line_z = torch.nn.Parameter(scale * torch.randn((1, dim, res, 1), device=device))
        self.line_x = torch.nn.Parameter(scale * torch.randn((1, dim, res, 1), device=device))
        self.line_y = torch.nn.Parameter(scale * torch.randn((1, dim, res, 1), device=device))

        self.basis_mat = torch.nn.Linear(22*3, 22*3, bias=False).to(device)

        self.rgb_decoder = rgb_decoder(feat_dim=22*3, view_pe=2, middle_dim=64).to(device)
        self.density_decoder = torch.nn.Linear(6*3, 1).to(device)
        init_weights(self.density_decoder, 'xavier_uniform')

        self.gauge_network = continuous_transform(input_dim=6).to(device) # planes: [1,0,0], [0,1,0], [0,0,1].
        self.reg_network = continuous_reg(input_dim=2, output_dim=3).to(device)

        x, y = np.meshgrid(np.linspace(-0.9, 0.9, 32), np.linspace(-0.9, 0.9, 32), indexing="xy")
        grid = np.array(list(zip(x.ravel(), y.ravel())))
        self.sampled_prior = torch.from_numpy(grid).type(torch.FloatTensor).to(device)
        self.EMD = SamplesLoss("sinkhorn", p=2, blur=0.01)

        # self.codebook = nn.Embedding(2**23 + 2**22 + 2**21 + 2**20 + 2**19 + 2**18, 2).to(device)
        # self.codebook = nn.ModuleList([nn.Embedding(2 ** 20, 2) for i in range(8)])

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [
                     {'params': self.plane_xy, 'lr': lr_init_spatialxyz},
                     {'params': self.plane_yz, 'lr': lr_init_spatialxyz},
                     {'params': self.plane_xz, 'lr': lr_init_spatialxyz},

                     {'params': self.line_z, 'lr': lr_init_spatialxyz},
                     {'params': self.line_x, 'lr': lr_init_spatialxyz},
                     {'params': self.line_y, 'lr': lr_init_spatialxyz},

                     {'params': self.basis_mat.parameters(), 'lr': lr_init_network},

                     {'params': self.rgb_decoder.parameters(), 'lr': lr_init_network},
                     {'params': self.density_decoder.parameters(), 'lr': lr_init_network},
                     {'params': self.gauge_network.parameters(), 'lr': lr_init_network * 0.25},
                     {'params': self.reg_network.parameters(), 'lr': lr_init_network},

                     # {'params': self.codebook.parameters(), 'lr': lr_init_network},
                     ]
        return grad_vars

    def feature2density(self, density_features, density_shift=-10):
        # return F.softplus(density_features)
        return F.softplus(density_features+density_shift)

    def compute_feature(self, xyz_sampled, view_sampled, iteration=0):
        output = {}
        N, _ = xyz_sampled.shape
        target_xy, target_yz, target_xz = xyz_sampled[:, :2], xyz_sampled[:, 1:], xyz_sampled[:, ::2]
        target_xy, target_yz, target_xz = target_xy.unsqueeze(0).unsqueeze(2), target_yz.unsqueeze(0).unsqueeze(2), target_xz.unsqueeze(0).unsqueeze(2)

        target_z, target_x, target_y = xyz_sampled[:, 2:], xyz_sampled[:, :1], xyz_sampled[:, 1:2]
        target_z, target_x, target_y = torch.stack((torch.zeros_like(target_z), target_z), dim=-1), \
                                       torch.stack((torch.zeros_like(target_x), target_x), dim=-1), \
                                       torch.stack((torch.zeros_like(target_y), target_y), dim=-1),
        target_z, target_x, target_y = target_z.unsqueeze(0), target_x.unsqueeze(0), target_y.unsqueeze(0)

        xy_feat = F.grid_sample(self.plane_xy, target_xy, align_corners=True).view(-1, *xyz_sampled.shape[:1])
        z_feat = F.grid_sample(self.line_z, target_z, align_corners=True).view(-1, *xyz_sampled.shape[:1])
        yz_feat = F.grid_sample(self.plane_yz, target_yz, align_corners=True).view(-1, *xyz_sampled.shape[:1])
        x_feat = F.grid_sample(self.line_x, target_x, align_corners=True).view(-1, *xyz_sampled.shape[:1])
        xz_feat = F.grid_sample(self.plane_xz, target_xz, align_corners=True).view(-1, *xyz_sampled.shape[:1])
        y_feat = F.grid_sample(self.line_y, target_y, align_corners=True).view(-1, *xyz_sampled.shape[:1])

        plane_feat, line_feat = torch.cat([xy_feat, yz_feat, xz_feat]), torch.cat([z_feat, x_feat, y_feat])
        plane_feat_rgb, line_feat_rgb = torch.cat([xy_feat[6:, :], yz_feat[6:, :], xz_feat[6:, :]]), \
                                        torch.cat([z_feat[6:, :], x_feat[6:, :], y_feat[6:, :]])

        # plane_feat_density, line_feat_density = torch.cat([xy_feat[:6, :], yz_feat[6:, :], xz_feat[6:, :]]), \
        #                                 torch.cat([z_feat[:6, :], x_feat[6:, :], y_feat[6:, :]])

        # feat = torch.sum(plane_feat * line_feat, dim=0)
        # density = self.feature2density(feat_density)

        feat_rgb = self.basis_mat((plane_feat_rgb * line_feat_rgb).T)
        rgb = self.rgb_decoder(feat_rgb, view_sampled)

        return density, rgb, output

    def density_L1(self):
        # total = 0
        # for idx in range(len(self.density_plane)):
        total = torch.mean(torch.abs(self.plane_xy[:, :6, :, :])) + torch.mean(torch.abs(self.line_z[:, :6, :, :]))
        total = total + torch.mean(torch.abs(self.plane_yz[:, :6, :, :])) + torch.mean(torch.abs(self.line_x[:, :6, :, :]))
        total = total + torch.mean(torch.abs(self.plane_xz[:, :6, :, :])) + torch.mean(torch.abs(self.line_y[:, :6, :, :]))
        # + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total


class TensoRFSplit(Base):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensoRFSplit, self).__init__(aabb, gridSize, device, **kargs)
    #     x = c. Density.

    def init_model(self, res=300, dim=72, scale=0.1, device=None):

        self.plane_xy = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))
        self.plane_yz = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))
        self.plane_xz = torch.nn.Parameter(scale * torch.randn((1, dim, res, res), device=device))

        self.line_z = torch.nn.Parameter(scale * torch.randn((1, dim, res, 1), device=device))
        self.line_x = torch.nn.Parameter(scale * torch.randn((1, dim, res, 1), device=device))
        self.line_y = torch.nn.Parameter(scale * torch.randn((1, dim, res, 1), device=device))

        self.basis_mat = torch.nn.Linear(60*3, 60*3, bias=False).to(device)


        self.rgb_decoder = rgb_decoder(feat_dim=60*3, view_pe=2, middle_dim=64).to(device)
        self.density_decoder = torch.nn.Linear(12*3, 1).to(device)
        init_weights(self.density_decoder, 'xavier_uniform')

        # self.gauge_network = continuous_transform(input_dim=6).to(device) # planes: [1,0,0], [0,1,0], [0,0,1].
        # self.reg_network = continuous_reg(input_dim=2, output_dim=3).to(device)

        # x, y = np.meshgrid(np.linspace(-0.9, 0.9, 32), np.linspace(-0.9, 0.9, 32), indexing="xy")
        # grid = np.array(list(zip(x.ravel(), y.ravel())))
        # self.sampled_prior = torch.from_numpy(grid).type(torch.FloatTensor).to(device)
        # self.EMD = SamplesLoss("sinkhorn", p=2, blur=0.01)

        # self.codebook = nn.Embedding(2**23 + 2**22 + 2**21 + 2**20 + 2**19 + 2**18, 2).to(device)
        # self.codebook = nn.ModuleList([nn.Embedding(2 ** 20, 2) for i in range(8)])

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [
                     {'params': self.plane_xy, 'lr': lr_init_spatialxyz},
                     {'params': self.plane_yz, 'lr': lr_init_spatialxyz},
                     {'params': self.plane_xz, 'lr': lr_init_spatialxyz},

                     {'params': self.line_z, 'lr': lr_init_spatialxyz},
                     {'params': self.line_x, 'lr': lr_init_spatialxyz},
                     {'params': self.line_y, 'lr': lr_init_spatialxyz},

                     {'params': self.basis_mat.parameters(), 'lr': lr_init_network},

                     {'params': self.rgb_decoder.parameters(), 'lr': lr_init_network},
                     {'params': self.density_decoder.parameters(), 'lr': lr_init_network},
                     # {'params': self.gauge_network.parameters(), 'lr': lr_init_network * 0.25},
                     # {'params': self.reg_network.parameters(), 'lr': lr_init_network},

                     # {'params': self.codebook.parameters(), 'lr': lr_init_network},
                     ]
        return grad_vars

    def feature2density(self, density_features, density_shift=-10):
        # return F.softplus(density_features)
        return F.softplus(density_features+density_shift)

    def compute_feature(self, xyz_sampled, view_sampled, iteration=0):

        # pe_rgb = positional_encoding(xyz_sampled, 10).permute(1,0)  #2*3*10 = 60, 2,3,2 = 12    72
        # pe_den = positional_encoding(xyz_sampled, 2).permute(1,0)

        pe_rgb = InfoNest(xyz_sampled, 30).permute(1,0)
        pe_den = InfoNest(xyz_sampled, 6).permute(1,0)

        # pe_rgb, pe_den = 1.0, 1.0

        output = {}
        N, _ = xyz_sampled.shape
        target_xy, target_yz, target_xz = xyz_sampled[:, :2], xyz_sampled[:, 1:], xyz_sampled[:, ::2]
        target_xy, target_yz, target_xz = target_xy.unsqueeze(0).unsqueeze(2), target_yz.unsqueeze(0).unsqueeze(2), target_xz.unsqueeze(0).unsqueeze(2)

        target_z, target_x, target_y = xyz_sampled[:, 2:], xyz_sampled[:, :1], xyz_sampled[:, 1:2]
        target_z, target_x, target_y = torch.stack((torch.zeros_like(target_z), target_z), dim=-1), \
                                       torch.stack((torch.zeros_like(target_x), target_x), dim=-1), \
                                       torch.stack((torch.zeros_like(target_y), target_y), dim=-1),
        target_z, target_x, target_y = target_z.unsqueeze(0), target_x.unsqueeze(0), target_y.unsqueeze(0)

        xy_feat = F.grid_sample(self.plane_xy, target_xy, align_corners=True).view(-1, *xyz_sampled.shape[:1])
        z_feat = F.grid_sample(self.line_z, target_z, align_corners=True).view(-1, *xyz_sampled.shape[:1])
        yz_feat = F.grid_sample(self.plane_yz, target_yz, align_corners=True).view(-1, *xyz_sampled.shape[:1])
        x_feat = F.grid_sample(self.line_x, target_x, align_corners=True).view(-1, *xyz_sampled.shape[:1])
        xz_feat = F.grid_sample(self.plane_xz, target_xz, align_corners=True).view(-1, *xyz_sampled.shape[:1])
        y_feat = F.grid_sample(self.line_y, target_y, align_corners=True).view(-1, *xyz_sampled.shape[:1])

        plane_feat_density, line_feat_density = torch.cat([xy_feat[:12, :], yz_feat[:12, :], xz_feat[:12, :]]) * pe_den, \
                                                torch.cat([z_feat[:12, :], x_feat[:12, :], y_feat[:12, :]])
        plane_feat_rgb, line_feat_rgb = torch.cat([xy_feat[12:, :], yz_feat[12:, :], xz_feat[12:, :]]) * pe_rgb, \
                                        torch.cat([z_feat[12:, :], x_feat[12:, :], y_feat[12:, :]])

        # pe_feat = positional_encoding(xyz_sampled, 30)
        # xyz_feat_rgb = xyz_feat_rgb * pe_feat

        feat_density = torch.sum(plane_feat_density * line_feat_density, dim=0)
        density = self.feature2density(feat_density)

        feat_rgb = self.basis_mat((plane_feat_rgb * line_feat_rgb).T)
        rgb = self.rgb_decoder(feat_rgb, view_sampled)

        return density, rgb, output



class TensoNGF(Base):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensoNGF, self).__init__(aabb, gridSize, device, **kargs)

    def init_model(self, res=256, rgb_dim=48, density_dim=16, scale=0.1, device=None):
        self.res, self.scale, self.device = res, scale, device
        self.rgb_dim, self.density_dim = rgb_dim, density_dim
        self.dim = self.rgb_dim + self.density_dim

        self.gauge_xy = torch.zeros((1, 2, res, res), device=device)
        self.gauge_yz = torch.zeros((1, 2, res, res), device=device)
        self.gauge_xz = torch.zeros((1, 2, res, res), device=device)

        self.plane_xy = torch.nn.Parameter(scale * torch.randn((1, self.dim, res, res), device=device))
        self.plane_yz = torch.nn.Parameter(scale * torch.randn((1, self.dim, res, res), device=device))
        self.plane_xz = torch.nn.Parameter(scale * torch.randn((1, self.dim, res, res), device=device))

        self.plane_xy1 = torch.nn.Parameter(scale * torch.randn((1, self.dim, res, res), device=device))
        self.plane_yz1 = torch.nn.Parameter(scale * torch.randn((1, self.dim, res, res), device=device))
        self.plane_xz1 = torch.nn.Parameter(scale * torch.randn((1, self.dim, res, res), device=device))

        self.line_z = torch.nn.Parameter(scale * torch.randn((1, self.dim, res, 1), device=device))
        self.line_x = torch.nn.Parameter(scale * torch.randn((1, self.dim, res, 1), device=device))
        self.line_y = torch.nn.Parameter(scale * torch.randn((1, self.dim, res, 1), device=device))

        self.basis_mat = torch.nn.Linear(self.rgb_dim*3, self.rgb_dim*3, bias=False).to(device)
        self.rgb_decoder = rgb_decoder(feat_dim=self.rgb_dim*3, view_pe=2, middle_dim=64).to(device)
        self.density_decoder = torch.nn.Linear(self.density_dim*3, 1).to(device)
        init_weights(self.density_decoder, 'xavier_uniform')
        # self.reg_network = continuous_reg(input_dim=2, output_dim=3).to(device).

        x, y = np.meshgrid(np.linspace(-0.99, 0.99, 32), np.linspace(-0.99, 0.99, 32), indexing="xy")
        grid = np.array(list(zip(x.ravel(), y.ravel())))
        self.sampled_prior = torch.from_numpy(grid).type(torch.FloatTensor).to(device)
        self.EMD = SamplesLoss("sinkhorn", p=2, blur=0.01)

        # self.xy_randn = torch.randn((1, 2, 32, 32), device=device) * 0.25
        # self.yz_randn = torch.randn((1, 2, 32, 32), device=device) * 0.25
        # self.xz_randn = torch.randn((1, 2, 32, 32), device=device) * 0.25

        self.xyz_randn = torch.randn((4, 4, 4, 3), device=device) * 0.25


    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [
                     {'params': self.plane_xy, 'lr': lr_init_spatialxyz},
                     {'params': self.plane_yz, 'lr': lr_init_spatialxyz},
                     {'params': self.plane_xz, 'lr': lr_init_spatialxyz},

                     {'params': self.plane_xy1, 'lr': lr_init_spatialxyz},
                     {'params': self.plane_yz1, 'lr': lr_init_spatialxyz},
                     {'params': self.plane_xz1, 'lr': lr_init_spatialxyz},

                     {'params': self.line_z, 'lr': lr_init_spatialxyz},
                     {'params': self.line_x, 'lr': lr_init_spatialxyz},
                     {'params': self.line_y, 'lr': lr_init_spatialxyz},

                     {'params': self.basis_mat.parameters(), 'lr': lr_init_network},
                     {'params': self.rgb_decoder.parameters(), 'lr': lr_init_network},
                     {'params': self.density_decoder.parameters(), 'lr': lr_init_network},
                     # {'params': self.gauge_network.parameters(), 'lr': lr_init_network},
                     # {'params': self.reg_network.parameters(), 'lr': lr_init_network}.
                    ]
        return grad_vars

    def feature2density(self, density_features, density_shift=-10):
        # return F.softplus(density_features).
        return F.softplus(density_features+density_shift)

    def compute_feature(self, xyz_sampled, view_sampled, iteration=0):
        output = {}
        N, _ = xyz_sampled.shape
        xy, yz, xz = xyz_sampled[:, :2].view(1, N, 1, 2), xyz_sampled[:, 1:].view(1, N, 1, 2), xyz_sampled[:, ::2].view(1, N, 1, 2)
        # xy, xz, target_xz = target_xy.view(1, N, 1, 2), target_yz.view(1, N, 1, 2), target_xz.view(1, N, 1, 2).

        xyz_sampled_ = xyz_sampled.unsqueeze(0).unsqueeze(2).unsqueeze(3).detach()
        xyz_randn = self.xyz_randn.unsqueeze(0).permute(0, 4, 1, 2, 3)  #1,x,y,z,dim | 1,dim,x,y,z
        # code_vector2 = self.codebook2(self.code_idx2).unsqueeze(0).permute(0, 4, 1, 2, 3)
        xyz_offset = F.grid_sample(xyz_randn, xyz_sampled_, align_corners=True)
        xyz_offset = xyz_offset.squeeze().permute(1, 0)
        xy_, yz_, xz_ = xyz_offset[:, :2].view(1, N, 1, 2), xyz_offset[:, 1:].view(1, N, 1, 2), xyz_offset[:, ::2].view(
            1, N, 1, 2)
        xy, yz, xz = xy + xy_, yz + yz_, xz + xz_





        # if iteration == 750:
        #     self.plane_xy = torch.nn.Parameter(self.scale * torch.randn((1, self.dim, self.res, self.res), device=self.device))
        #     self.plane_yz = torch.nn.Parameter(self.scale * torch.randn((1, self.dim, self.res, self.res), device=self.device))
        #     self.plane_xz = torch.nn.Parameter(self.scale * torch.randn((1, self.dim, self.res, self.res), device=self.device))

        # if iteration > 750:
        #     xy_ = xy + F.grid_sample(self.gauge_xy.detach(), xy.detach(), align_corners=True).view(1, 2, 1, N).permute(0, 3, 2, 1)
        #     yz_ = yz + F.grid_sample(self.gauge_yz.detach(), yz.detach(), align_corners=True).view(1, 2, 1, N).permute(0, 3, 2, 1)
        #     xz_ = xz + F.grid_sample(self.gauge_xz.detach(), xz.detach(), align_corners=True).view(1, 2, 1, N).permute(0, 3, 2, 1)


        # xy_ = xy + F.grid_sample(self.xy_randn.detach(), xy.detach(), align_corners=True).view(1, 2, 1, N).permute(0, 3, 2, 1)
        # yz_ = yz + F.grid_sample(self.yz_randn.detach(), yz.detach(), align_corners=True).view(1, 2, 1, N).permute(0, 3, 2, 1)
        # xz_ = xz + F.grid_sample(self.xz_randn.detach(), xz.detach(), align_corners=True).view(1, 2, 1, N).permute(0, 3, 2, 1)

        xy_feat = F.grid_sample(self.plane_xy, xy.detach(), align_corners=True).view(-1, *xyz_sampled.shape[:1])
        yz_feat = F.grid_sample(self.plane_yz, yz.detach(), align_corners=True).view(-1, *xyz_sampled.shape[:1])
        xz_feat = F.grid_sample(self.plane_xz, xz.detach(), align_corners=True).view(-1, *xyz_sampled.shape[:1])

        # xy_feat += F.grid_sample(self.plane_xy, xy.detach(), align_corners=True).view(-1, *xyz_sampled.shape[:1])
        # yz_feat += F.grid_sample(self.plane_yz, yz.detach(), align_corners=True).view(-1, *xyz_sampled.shape[:1])
        # xz_feat += F.grid_sample(self.plane_xz, xz.detach(), align_corners=True).view(-1, *xyz_sampled.shape[:1])

        z, x, y = xyz_sampled[:, 2:], xyz_sampled[:, :1], xyz_sampled[:, 1:2]
        z, x, y = torch.stack((torch.zeros_like(z), z), dim=-1), torch.stack((torch.zeros_like(x), x), dim=-1), torch.stack((torch.zeros_like(y), y), dim=-1)
        z, x, y = z.unsqueeze(0), x.unsqueeze(0), y.unsqueeze(0)

        z_feat = F.grid_sample(self.line_z, z, align_corners=True).view(-1, *xyz_sampled.shape[:1])
        x_feat = F.grid_sample(self.line_x, x, align_corners=True).view(-1, *xyz_sampled.shape[:1])
        y_feat = F.grid_sample(self.line_y, y, align_corners=True).view(-1, *xyz_sampled.shape[:1])

        plane_feat_density, line_feat_density = torch.cat([xy_feat[:self.density_dim, :], yz_feat[:self.density_dim, :], xz_feat[:self.density_dim, :]]), \
                                                torch.cat([z_feat[:self.density_dim, :], x_feat[:self.density_dim, :], y_feat[:self.density_dim, :]])
        plane_feat_rgb, line_feat_rgb = torch.cat([xy_feat[self.density_dim:, :], yz_feat[self.density_dim:, :], xz_feat[self.density_dim:, :]]), \
                                        torch.cat([z_feat[self.density_dim:, :], x_feat[self.density_dim:, :], y_feat[self.density_dim:, :]])

        feat_density = torch.sum(plane_feat_density * line_feat_density, dim=0)
        density = self.feature2density(feat_density)

        feat_rgb = self.basis_mat((plane_feat_rgb * line_feat_rgb).T)
        rgb = self.rgb_decoder(feat_rgb, view_sampled)

        output["target_gauge"] = [xy.squeeze(), yz.squeeze(), xz.squeeze()]
        return density, rgb, output
    # shrink. autograd, require.

    def gauge_transformation(self, output, valid_weight, N=1024):

        self.gauge_xy.requires_grad = True
        self.gauge_yz.requires_grad = True
        self.gauge_xz.requires_grad = True
        # self.gauge_xy, self.gauge_yz, self.gauge_xz = self.gauge_x
        # print('*************', valid_weight.shape)
        # N = valid_weight.shape[0]

        for i in range(1):
            sampled_index = torch.multinomial(valid_weight, N, replacement=False)
            # print(sampled_index[:10])
            xy, yz, xz = output['target_gauge']
            sampled_xy, sampled_yz, sampled_xz = xy[sampled_index], yz[sampled_index], xz[sampled_index]
            xy, yz, xz = sampled_xy.clone().detach(), sampled_yz.clone().detach(), sampled_xz.clone().detach()

            gauged_xy = xy + F.grid_sample(self.gauge_xy, xy.view(1, N, 1, 2), align_corners=True).squeeze().permute(1, 0)
            gauged_yz = yz + F.grid_sample(self.gauge_yz, yz.view(1, N, 1, 2), align_corners=True).squeeze().permute(1, 0)
            gauged_xz = xz + F.grid_sample(self.gauge_xz, xz.view(1, N, 1, 2), align_corners=True).squeeze().permute(1, 0)

            prior_loss = self.EMD(gauged_xy, self.sampled_prior) + self.EMD(gauged_yz, self.sampled_prior) + self.EMD(gauged_xz, self.sampled_prior)
            # [g_xy] = torch.autograd.grad(prior_loss, [self.gauge_xy])
            [g_xy, g_yz, g_xz] = torch.autograd.grad(prior_loss, [self.gauge_xy, self.gauge_yz, self.gauge_xz])
            self.gauge_xy.data -= 0.1 * len(gauged_xy) * g_xy
            self.gauge_yz.data -= 0.1 * len(gauged_yz) * g_yz
            self.gauge_xz.data -= 0.1 * len(gauged_xz) * g_xz
        # 1/0.
        # project = 1024

        # fig = plt.figure(figsize=(6, 6))
        # xy = xy.detach().cpu().numpy()
        #
        # # xy = xy.shape.gauge_xy
        # for i in range(len(gauged_xy)):
        #     x_values = [xy[i][0], gauged_xy[i][0]]
        #     y_values = [xy[i][1], gauged_xy[i][1]]
        #     plt.plot(x_values, y_values)
        # fig.savefig('emd_match.png', bbox_inches='tight'). xyz

        gauged_xy = gauged_xy.detach().cpu().numpy()
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(gauged_xy[:, 0], gauged_xy[:, 1], 40 * 500 / len(gauged_xy), [(0.55, 0.55, 0.95)], edgecolors="none")
        plt.axis([-1, 1, -1, 1])
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.tight_layout()

        fig.savefig('emd_mapping.png', bbox_inches='tight')
        plt.close()
        # 1 / 0. xyz.shape. xy. large occupy. occupy.


class TensorNGP(Base):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorNGP, self).__init__(aabb, gridSize, device, **kargs)

    def spatial_hash(self, x_grid, y_grid, z_grid, res, code_num=256, p1=1, p2=2654435761, p3=805459861):
        x, y, z = np.floor(x_grid * res), np.floor(y_grid * res), np.floor(z_grid * res)
        x, y, z = x.astype(int), y.astype(int), z.astype(int)
        xy = np.bitwise_xor(x * p1, y * p2)
        xyz = np.bitwise_xor(xy, z * p3)
        return xyz % code_num

    def init_model(self, code_num=256, code_dim=120, scale=0.1, device=None):
        self.device = device
        self.res = 16

        # self.codebook = torch.nn.Parameter(scale * torch.randn((code_num, code_dim))).to(device)
        self.codebook = nn.Embedding(code_num, code_dim).to(device)
        self.codebook.weight.data.uniform_(-1.0 / code_num, 1.0 / code_num)

        self.codebook2 = nn.Embedding(code_num, code_dim).to(device)
        self.codebook2.weight.data.uniform_(-1.0 / code_num, 1.0 / code_num)

        density_decoder = []
        density_decoder.append(nn.Linear(code_dim*2, code_dim*2))
        density_decoder.append(nn.LeakyReLU(0.2))
        density_decoder.append(nn.Linear(code_dim*2, 1))
        self.density_decoder = nn.Sequential(*density_decoder).to(device)

        self.rgb_decoder = rgb_decoder(feat_dim=code_dim * 2, view_pe=2, middle_dim=64).to(device)

        x = np.linspace(0., 1., self.res)
        y = np.linspace(0., 1., self.res)
        z = np.linspace(0., 1., self.res)
        xyz_grid = np.meshgrid(x, y, z, indexing='ij')
        code_idx = self.spatial_hash(xyz_grid[0], xyz_grid[1], xyz_grid[2], self.res)
        self.code_idx = torch.from_numpy(code_idx).to(self.device).detach()

        x = np.linspace(0., 1., self.res*2)
        y = np.linspace(0., 1., self.res*2)
        z = np.linspace(0., 1., self.res*2)
        xyz_grid = np.meshgrid(x, y, z, indexing='ij')
        code_idx = self.spatial_hash(xyz_grid[0], xyz_grid[1], xyz_grid[2], self.res*2)
        self.code_idx2 = torch.from_numpy(code_idx).to(self.device).detach()

        self.beta = torch.nn.Parameter(torch.ones(1, code_dim).to(device))
        # self.beta =  # 16*120 1*120 = 1*120.  1 /  . c
        # self.freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)

        freq_bands = (2 ** torch.arange(40).float()).to(self.device)
        self.freq_bands = torch.nn.Parameter(1/(freq_bands+1e-12))

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [{'params': self.codebook.parameters(), 'lr': lr_init_spatialxyz},
                     {'params': self.codebook2.parameters(), 'lr': lr_init_spatialxyz},
                     # {'params': self.density_decode.parameters(), 'lr': lr_init_network},
                     # {'params': self.gauge_transform2.parameters(), 'lr': lr_init_network},
                     {'params': self.density_decoder.parameters(), 'lr': lr_init_network},
                     {'params': self.rgb_decoder.parameters(), 'lr': lr_init_network},
                     {'params': self.freq_bands, 'lr': 0.00001},
                     ]
        # if isinstance(self.renderModule, torch.nn.Module):
        #     grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_init_network}]
        return grad_vars

    def feature2density(self, density_features, density_shift=-10):
        # return F.softplus(density_features).
        return F.softplus(density_features+density_shift)

    def compute_feature(self, xyz_sampled, view_sampled, iteration=0):
        output = {}

        xyz_sampled_ = xyz_sampled.unsqueeze(0).unsqueeze(2).unsqueeze(3).detach()

        code_vector1 = self.codebook(self.code_idx).unsqueeze(0).permute(0, 4, 1, 2, 3)
        code_vector2 = self.codebook2(self.code_idx2).unsqueeze(0).permute(0, 4, 1, 2, 3)

        sampled_vector1 = F.grid_sample(code_vector1, xyz_sampled_, align_corners=True)
        sampled_vector1 = sampled_vector1.squeeze().permute(1, 0)

        sampled_vector2 = F.grid_sample(code_vector2, xyz_sampled_, align_corners=True)
        sampled_vector2 = sampled_vector2.squeeze().permute(1, 0)
        feat = torch.concat([sampled_vector1, sampled_vector2], dim=-1)

        # pe_feat = positional_encoding(xyz_sampled, 40)  #2*3*40 = 240.

        # feat_ = torch.stack([-feat[..., 1::2], feat[..., ::2]], dim=1)
        # feat_ = feat_.view(feat.shape)
        # cos_pe, sin_pe = rot_pe(xyz_sampled, 80)
        # feat_final = feat * cos_pe + feat_ * sin_pe

        # InfoNest. 12624174
        feat_ = InfoNest(xyz_sampled, 40)    #, beta=self.beta) #theta=self.freq_bands) #, )
        # feat_final = feat_ * feat

        feat_final = feat

        density_feature = self.density_decoder(feat_final).reshape(-1)
        density = self.feature2density(density_feature)

        rgb = self.rgb_decoder(feat_final, view_sampled)
        # reg_loss = xyz_sampled.sum(-1).
        return density, rgb, output