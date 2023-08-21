import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gauge_fields import GaugeTransform, InverseGauge
from .renderer import find_ray_generation_method, ray_march, alpha_blend, simple_tone_map, radiance_render
from .decoder import GeometryMlpDecoder, TextureMlpDecoder
from util import get_scheduler

class NeuTex(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.net_geometry_decoder = GeometryMlpDecoder(pos_freqs=10, hidden_size=256, num_layers=10)

        self.inverse_gauge = InverseGauge(self.opt.points_per_primitive, self.opt.primitive_type,)

        self.gauge_transform = GaugeTransform(self.opt.primitive_type,)

        self.net_texture = TextureMlpDecoder(3, 10, 6, uv_dim=2 if self.opt.primitive_type == "square" else 3,
            layers=[5, 3], width=256, clamp=False, primitive_type=self.opt.primitive_type, target_texture=self.opt.target_texture)

        self.raygen = find_ray_generation_method("cube")

    def forward(self, camera_position=None, ray_direction=None, background_color=None):
        output = {}

        orig_ray_pos, ray_dist, ray_valid, ray_ts = self.raygen(camera_position, ray_direction, self.opt.sample_num, jitter=0.05)
        ray_pos = orig_ray_pos  # (N, rays, samples, 3)
        mlp_output = self.net_geometry_decoder(ray_pos)

        # compute_atlasnet:
        point_array_2d, points_3d = self.inverse_gauge(regular_point_count=None)
        output["points"] = points_3d.view(points_3d.shape[0], -1, points_3d.shape[-1]).permute(0, 2, 1)
        uv = self.gauge_transform(ray_pos)
        point_features = self.net_texture(uv, ray_direction[:, :, None, :])


        density = mlp_output["density"][..., None]
        radiance = point_features[..., :3]
        bsdf = [density, radiance]
        bsdf = torch.cat(bsdf, dim=-1)

        (ray_color, point_color, opacity, acc_transmission, blend_weight, background_transmission, background_blend_weight) \
            = ray_march(ray_direction, ray_pos, ray_dist, ray_valid, bsdf, None, None, radiance_render, alpha_blend)
        if background_color is not None:
            ray_color += (background_color[:, None, :] * background_blend_weight[:, :, None])
        ray_color = simple_tone_map(ray_color)
        output["color"] = ray_color
        output["transmittance"] = background_blend_weight

        # if compute_inverse_mapping:
        output["points_original"] = ray_pos
        output["points_inverse"] = self.inverse_gauge.map(uv)
        output["points_inverse_weights"] = blend_weight

        return output






class BaseModel:
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return self.__class__.__name__

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device("cuda:{}".format(self.gpu_ids[0]) if self.gpu_ids else torch.device("cpu"))
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        torch.backends.cudnn.benchmark = True

        self.loss_names = []  # losses to report
        self.model_names = []  # Models that will be used
        self.visual_names = []  # visuals to show at test time

    def set_input(self, input: dict):
        self.input = input

    def forward(self):
        """Run the forward pass. Read from self.input, set self.output"""
        raise NotImplementedError()

    def setup(self, opt):
        """Creates schedulers if train, Load and print networks if resume"""
        if self.is_train:
            self.schedulers = [get_scheduler(optim, opt) for optim in self.optimizers]
        if opt.load_subnetworks_dir:
            nets = opt.load_subnetworks.split(",")
            self.load_subnetworks(
                opt.load_subnetworks_epoch,
                opt.load_subnetworks.split(","),
                opt.load_subnetworks_dir,
            )
            print("loading pretrained {}".format(nets))
        if not self.is_train or opt.resume_dir:
            self.load_networks(opt.resume_epoch)
        if opt.freeze_subnetworks:
            nets = opt.freeze_subnetworks.split(",")
            self.freeze_subnetworks(opt.freeze_subnetworks.split(","))
            print("freezing {}".format(nets))

        self.print_networks()

    def eval(self):
        """turn on eval mode"""
        for net in self.get_networks():
            net.eval()

    def train(self):
        for net in self.get_networks():
            net.train()

    def test(self):
        with torch.no_grad():
            self.forward()

    def get_networks(self) -> [nn.Module]:
        ret = []
        for name in self.model_names:
            assert isinstance(name, str)
            net = getattr(self, "{}".format(name))
            assert isinstance(net, nn.Module)
            ret.append(net)
        return ret

    def get_current_visuals(self):
        ret = {}
        for name in self.visual_names:
            assert isinstance(name, str)
            ret[name] = getattr(self, name)
        return ret

    def get_current_losses(self):
        ret = {}
        for name in self.loss_names:
            assert isinstance(name, str)
            ret[name] = getattr(self, "loss_" + name)
        return ret

    def get_subnetworks(self) -> dict:
        raise NotImplementedError()

    def freeze_subnetworks(self, network_names):
        nets = self.get_subnetworks()
        for name in network_names:
            nets[name].requires_grad_(False)

    def unfreeze_subnetworks(self, network_names):
        nets = self.get_subnetworks()
        for name in network_names:
            nets[name].requires_grad_(True)

    def save_subnetworks(self, epoch):
        nets = self.get_subnetworks()
        for name, net in nets.items():
            save_filename = "{}_subnet_{}.pth".format(epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)
            try:
                if isinstance(net, nn.DataParallel):
                    net = net.module
                torch.save(net.state_dict(), save_path)
            except Exception as e:
                print(e)

    def load_subnetworks(self, epoch, names=None, resume_dir=None):
        networks = self.get_subnetworks()
        if names is None:
            names = set(networks.keys())
        else:
            names = set(names)

        for name, net in networks.items():
            if name not in names:
                continue

            load_filename = "{}_subnet_{}.pth".format(epoch, name)
            load_path = os.path.join(resume_dir if resume_dir is not None else self.opt.resume_dir, load_filename,)

            if not os.path.isfile(load_path):
                print("cannot load", load_path)
                continue

            state_dict = torch.load(load_path, map_location=self.device)
            if isinstance(net, nn.DataParallel):
                net = net.module

            net.load_state_dict(state_dict, strict=True)

    def save_networks(self, epoch, other_states={}):
        for name, net in zip(self.model_names, self.get_networks()):
            save_filename = "{}_net_{}.pth".format(epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)

            try:
                if isinstance(net, nn.DataParallel):
                    net = net.module
                torch.save(net.state_dict(), save_path)
            except Exception as e:
                print(e)

        save_filename = "{}_states.pth".format(epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(other_states, save_path)

    def load_networks(self, epoch):
        for name, net in zip(self.model_names, self.get_networks()):
            print("loading", name)
            assert isinstance(name, str)
            load_filename = "{}_net_{}.pth".format(epoch, name)
            load_path = os.path.join(self.opt.resume_dir, load_filename)

            if not os.path.isfile(load_path):
                print("cannot load", load_path)
                continue

            state_dict = torch.load(load_path, map_location=self.device)
            if isinstance(net, nn.DataParallel):
                net = net.module

            net.load_state_dict(state_dict, strict=False)

    def print_networks(self):
        print("------------------- Networks -------------------")
        for name, net in zip(self.model_names, self.get_networks()):
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            print("[Network {}] Total number of parameters: {:.3f}M".format(name, num_params / 1e6))
        print("------------------------------------------------")

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        for i, optim in enumerate(self.optimizers):
            lr = optim.param_groups[0]["lr"]

    def set_current_step(self, step=None):
        self.current_step = step


def create_model(opt):
    instance = Model()
    instance.initialize(opt)
    print("model [{}] was created".format(instance.name()))
    return instance

class Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument("--sample_num", required=True, type=int, help="number of samples along ray",)

        parser.add_argument("--loss_color_weight", required=False, default=1, type=float, help="rendering loss",)

        parser.add_argument("--loss_bg_weight", required=False, default=1, type=float, help="transmittance loss",)

        parser.add_argument("--loss_origin_weight", required=False, default=1, type=float, help="penalize points far from origin",)

        parser.add_argument("--loss_inverse_mapping_weight", required=False, default=0, type=float, help="inverse mapping loss",)

        parser.add_argument("--primitive_type", type=str, choices=["square", "sphere"], required=True, help="template",)

        parser.add_argument("--points_per_primitive", type=int, required=True, help="number of points per primitive",)

        parser.add_argument("--target_texture", type=str, required=False, default='None', help="texture editing",)


    def initialize(self, opt):
        super().initialize(opt)
        self.model_names = ["NeuTex"]
        self.NeuTex = NeuTex(opt)

        assert self.opt.gpu_ids, "gpu is required"
        if self.opt.gpu_ids:
            self.NeuTex.to(self.device)
            self.NeuTex = torch.nn.DataParallel(self.NeuTex, self.opt.gpu_ids)

        if self.is_train:
            self.schedulers = []
            self.optimizers = []

            params = list(self.NeuTex.parameters())
            self.optimizer = torch.optim.Adam(params, lr=opt.lr)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        self.input = {}
        for key in input:
            self.input[key] = input[key].to(self.device)

    def forward(self):
        self.output = self.NeuTex(self.input["campos"], self.input["raydir"], self.input["background_color"])

        if "gt_image" in self.input:
            self.compute_loss()

        with torch.no_grad():
            if "gt_image" in self.input:
                self.visual_names = ["gt_image"]
                self.gt_image = self.input["gt_image"]
            if "color" in self.output:
                self.visual_names.append("ray_color")
                self.ray_color = self.output["color"]
            if "foreground_blend_weight" in self.output:
                self.visual_names.append("transmittance")
                self.transmittance = self.output["foreground_blend_weight"][..., None].expand_as(self.ray_color)

    def compute_loss(self):
        self.loss_names = []
        self.loss_total = 0
        self.loss_names.append("total")

        if self.opt.loss_color_weight > 0:
            self.loss_color = F.mse_loss(self.output["color"], self.input["gt_image"])
            self.loss_total += self.opt.loss_color_weight * self.loss_color
            self.loss_names.append("color")

        if self.opt.loss_bg_weight > 0:
            if "transmittance" in self.input:
                self.loss_bg = F.mse_loss(self.output["transmittance"], self.input["transmittance"])
            else:
                self.loss_bg = 0
            self.loss_total += self.opt.loss_bg_weight * self.loss_bg
            self.loss_names.append("bg")

        if self.opt.loss_origin_weight > 0:
            self.loss_origin = (((self.output["points"] ** 2).sum(-2) - 1).clamp(min=0).sum())
            self.loss_total += self.opt.loss_origin_weight * self.loss_origin
            self.loss_names.append("origin")

        if self.opt.loss_inverse_mapping_weight > 0:
            gt_points = self.output["points_original"]
            points = self.output["points_inverse"] # n,1,v
            pw = self.output["points_inverse_weights"]
            dist = ((gt_points - points) ** 2).sum(-1)
            dist = (dist * pw).sum(-1)
            dist = dist.mean()

            self.loss_inverse_mapping = dist
            self.loss_total += (self.opt.loss_inverse_mapping_weight * self.loss_inverse_mapping)
            self.loss_names.append("inverse_mapping")

    def backward(self):
        self.optimizer.zero_grad()
        if self.opt.is_train:
            self.loss_total.backward()
            self.optimizer.step()

    def optimize_parameters(self):
        self.forward()
        self.backward()

    def test(self):
        with torch.no_grad():
            self.output = self.NeuTex(self.input["campos"], self.input["raydir"], self.input["background_color"])
            if "gt_image" in self.input:
                self.visual_names = ["gt_image"]
                self.gt_image = self.input["gt_image"]
            if "color" in self.output:
                self.visual_names.append("ray_color")
                self.ray_color = self.output["color"]
            if "foreground_blend_weight" in self.output:
                self.visual_names.append("transmittance")
                self.transmittance = self.output["foreground_blend_weight"][..., None].expand_as(self.ray_color)

    def get_subnetworks(self):
        return {
            "geometry": self.NeuTex.module.net_geometry_decoder,
            "inverse": self.NeuTex.module.inverse_gauge,
            "gauge": self.NeuTex.module.gauge_transform,
            "texture": self.NeuTex.module.net_texture,
        }

    def coordinate_deformation(self, template, viewdir=[0, 0, 1], icosphere_division=6):
        # clamp_texture = True
        with torch.no_grad():
            import trimesh


            if template == 'sphere':
                mesh = trimesh.creation.icosphere(icosphere_division)
                grid = torch.tensor(mesh.vertices).to(self.device).float()
                grid = grid.view(1, grid.shape[0], 3)
                meshes = []
                mesh = trimesh.creation.icosphere(icosphere_division)
            else:
                # side_length = 100
                # uv = np.stack(np.meshgrid(*([np.linspace(-1, 1, side_length)] * 2), indexing="ij"), axis=-1,)
                # grid = torch.FloatTensor(uv).cuda().view(1, -1, 2)
                mesh = trimesh.creation.box(extents=[2, 2, 0.0])
                mesh = mesh.subdivide_loop(iterations=8)
                grid = torch.tensor(mesh.vertices[:, :2]).to(self.device).float()
                grid = grid.view(1, grid.shape[0], 2)
                # print(mesh_.vertices.shape, mesh_.vertices)
                meshes = []
                mesh = trimesh.creation.box(extents=[2, 2, 0.0])
                mesh = mesh.subdivide_loop(iterations=7)

            vertices = self.NeuTex.module.inverse_gauge.map(grid)

            mesh.vertices = vertices.squeeze().data.cpu().numpy()
            meshes.append(mesh)
            # weights = (torch.zeros(grid.shape[:3]).float().to(self.device))
            # weights[0, :, :] = 1

            viewdir = (torch.tensor(viewdir).float().to(grid.device).expand(grid.shape[:-2] + (3,)))
            texture = self.NeuTex.module.net_texture(grid, viewdir)[0]

            return meshes, [texture.squeeze()]