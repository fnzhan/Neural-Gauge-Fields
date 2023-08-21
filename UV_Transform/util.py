import numpy as np
import os
import time
from PIL import Image
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn import init
# import open3d

def generate_grid(dim, resolution):
    grid = np.stack(np.meshgrid(*([np.arange(resolution)] * dim), indexing="ij"), axis=-1)
    grid = (2 * grid + 1) / resolution - 1
    return grid

def save_image(img_array, filepath):
    assert len(img_array.shape) == 2 or (
        len(img_array.shape) == 3 and img_array.shape[2] in [3, 4]
    )
    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    Image.fromarray(img_array).save(filepath)


def depth_to_pointcloud(depth_array, camera_position, ray_directions, mask=None):
    """
    Args:
        depth_array: :math:`[M]`
        camera_position: :math:`[M, 3]` or :math: `[3]`
        ray_directions: :math:`[M, 3]`
        mask: :math:`[M]` or None
    Return:
        points: :math:`[M', 3]` valid 3d points
    """
    assert len(depth_array.shape) == 1
    M = depth_array.shape[0]
    assert camera_position.shape in [(3,), (M, 3)]
    assert ray_directions.shape == (M, 3)
    assert mask is None or mask.shape == (M,)

    if mask is None:
        mask = np.ones_like(depth_array)

    ray_dir = ray_directions / np.linalg.norm(ray_directions, axis=-1, keepdims=True)
    points = camera_position + ray_dir * depth_array.reshape((M, 1))
    points = points[mask]
    return points


# TODO: implement a custom module for saving full PCD
def save_pointcloud(points, filename):
    assert len(points.shape) == 2 and points.shape[1] == 3

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        + "VERSION 0.7\n"
        + "FIELDS x y z\n"
        + "SIZE 4 4 4\n"
        + "TYPE F F F\n"
        + "COUNT 1 1 1\n"
        + "WIDTH {}\n".format(len(points))
        + "HEIGHT 1\n"
        + "VIEWPOINT 0 0 0 1 0 0 0\n"
        + "POINTS {}\n".format(len(points))
        + "DATA binary\n"
    )

    with open(filename, "wb") as f:
        f.write(bytearray(header, "ascii"))
        f.write(points.astype(np.float32).tobytes())


class Visualizer:
    def __init__(self, opt):
        self.opt = opt
        self.log_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.image_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.output_dir)

    def display_current_results(
        self, visuals, total_steps, camera_position=None, ray_directions=None):
        for name, img in visuals.items():
            img = np.array(img)
            filename = "{:08d}-{}.png".format(total_steps, name)
            filepath = os.path.join(self.image_dir, filename)
            save_image(img, filepath)

        if camera_position is not None and ray_directions is not None:
            camera_position = np.array(camera_position)
            ray_directions = np.array(ray_directions)
            for name, img in visuals.items():
                if len(img.shape) == 2 and "depth" in name:
                    depth = np.array(img).reshape(-1)
                    filename = "step-{:08d}-{}.pcd".format(total_steps, name)
                    filepath = os.path.join(self.image_dir, filename)
                    pcd = depth_to_pointcloud(
                        depth, camera_position, ray_directions, depth != 0
                    )
                    save_pointcloud(pcd, filepath)

    def reset(self):
        self.start_time = time.time()
        self.acc_iterations = 0
        self.acc_losses = OrderedDict()

    def accumulate_losses(self, losses):
        self.acc_iterations += 1
        for k, v in losses.items():
            if k not in self.acc_losses:
                self.acc_losses[k] = 0
            self.acc_losses[k] += v

    def print_losses(self, total_steps):
        m = "End of iteration {} \t Number of batches {} \t Time taken: {:.2f}s\n".format(
            total_steps, self.acc_iterations, (time.time() - self.start_time)
        )
        m += "[Average Loss] "
        for k, v in self.acc_losses.items():
            m += "{}: {:.10f}   ".format(k, v / self.acc_iterations)

        filepath = os.path.join(self.log_dir, "log.txt")
        with open(filepath, "a") as f:
            f.write(m + "\n")
        print(m)







def convert_cube_uv_to_xyz(index, uvc):
    assert uvc.shape[-1] == 2
    vc, uc = uvc.unbind(-1)
    if index == 0:
        x = torch.ones_like(uc).to(uc.device)
        y = vc
        z = -uc
    elif index == 1:
        x = -torch.ones_like(uc).to(uc.device)
        y = vc
        z = uc
    elif index == 2:
        x = uc
        y = torch.ones_like(uc).to(uc.device)
        z = -vc
    elif index == 3:
        x = uc
        y = -torch.ones_like(uc).to(uc.device)
        z = vc
    elif index == 4:
        x = uc
        y = vc
        z = torch.ones_like(uc).to(uc.device)
    elif index == 5:
        x = -uc
        y = vc
        z = -torch.ones_like(uc).to(uc.device)
    else:
        raise ValueError(f"invalid index {index}")

    return F.normalize(torch.stack([x, y, z], axis=-1), dim=-1)


def load_cubemap(imgs):
    assert len(imgs) == 6
    return np.array([np.array(Image.open(img))[::-1] / 255.0 for img in imgs])


def sample_cubemap(cubemap, xyz):
    assert len(cubemap.shape) == 4
    assert cubemap.shape[0] == 6
    assert cubemap.shape[1] == cubemap.shape[2]
    assert xyz.shape[-1] == 3

    result = torch.zeros(xyz.shape[:-1] + (cubemap.shape[-1],)).float().to(xyz.device)

    x, y, z = xyz.unbind(-1)

    absX = x.abs()
    absY = y.abs()
    absZ = z.abs()

    isXPositive = x > 0
    isYPositive = y > 0
    isZPositive = z > 0

    maps = cubemap.unbind(0)
    masks = [
        isXPositive * (absX >= absY) * (absX >= absZ),
        isXPositive.logical_not() * (absX >= absY) * (absX >= absZ),
        isYPositive * (absY >= absX) * (absY >= absZ),
        isYPositive.logical_not() * (absY >= absX) * (absY >= absZ),
        isZPositive * (absZ >= absX) * (absZ >= absY),
        isZPositive.logical_not() * (absZ >= absX) * (absZ >= absY),
    ]

    uvs = []

    uc = -z[masks[0]] / absX[masks[0]]
    vc = y[masks[0]] / absX[masks[0]]
    uvs.append(torch.stack([uc, vc], dim=-1))

    uc = z[masks[1]] / absX[masks[1]]
    vc = y[masks[1]] / absX[masks[1]]
    uvs.append(torch.stack([uc, vc], dim=-1))

    uc = x[masks[2]] / absY[masks[2]]
    vc = -z[masks[2]] / absY[masks[2]]
    uvs.append(torch.stack([uc, vc], dim=-1))

    uc = x[masks[3]] / absY[masks[3]]
    vc = z[masks[3]] / absY[masks[3]]
    uvs.append(torch.stack([uc, vc], dim=-1))

    uc = x[masks[4]] / absZ[masks[4]]
    vc = y[masks[4]] / absZ[masks[4]]
    uvs.append(torch.stack([uc, vc], dim=-1))

    uc = -x[masks[5]] / absZ[masks[5]]
    vc = y[masks[5]] / absZ[masks[5]]
    uvs.append(torch.stack([uc, vc], dim=-1))

    for texture, mask, uv in zip(maps, masks, uvs):
        result[mask] = (
            F.grid_sample(
                texture.permute(2, 0, 1)[None],
                uv.view((1, -1, 1, 2)),
                padding_mode="border",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .view(uv.shape[:-1] + (texture.shape[-1],))
        )

    return result


def load_cube_from_single_texture(filename, rotate=True):
    img = Image.open(filename)
    img = img.resize((2048, 1536))
    img = np.array(img) / 255.0
    assert img.shape[0] * 4 == img.shape[1] * 3
    res = img.shape[0] // 3
    if rotate:
        cube = [
            img[res : 2 * res, :res][::-1],
            img[res : 2 * res, 2 * res : 3 * res][::-1],
            img[:res, res : 2 * res][:, ::-1],
            img[2 * res : 3 * res, res : 2 * res][:, ::-1],
            img[res : 2 * res, 3 * res :][::-1],
            img[res : 2 * res, res : 2 * res][::-1],
        ]
    else:
        cube = [
            img[res : 2 * res, 2 * res : 3 * res][::-1],
            img[res : 2 * res, :res][::-1],
            img[:res, res : 2 * res][::-1],
            img[2 * res : 3 * res, res : 2 * res][::-1],
            img[res : 2 * res, res : 2 * res][::-1],
            img[res : 2 * res, 3 * res :][::-1],
        ]

    return cube



def load_square(filename):
    img = Image.open(filename)
    # img = img.resize((2048, 1536))
    img = np.array(img)[::-1] / 255.0
    return img


def sample_square(square, uv):

    # for texture, mask, uv in zip(maps, masks, uvs):
    result =  F.grid_sample(square.permute(2, 0, 1)[None], uv.view((1, -1, 1, 2)), padding_mode="border",
                            align_corners=False,).permute(0, 2, 3, 1).view(uv.shape[:-1] + (square.shape[-1],))
    return result



def merge_cube_to_single_texture(cube, flip=True, rotate=True):
    """
    cube: (6,res,res,c)
    """
    assert cube.shape[0] == 6
    assert cube.shape[1] == cube.shape[2]
    res = cube.shape[1]
    result = torch.ones((3 * res, 4 * res, cube.shape[-1]))

    if flip:
        cube = cube.flip(1)
    if rotate:
        result[res : 2 * res, :res] = cube[0]
        result[res : 2 * res, res : 2 * res] = cube[5]
        result[res : 2 * res, 2 * res : 3 * res] = cube[1]
        result[res : 2 * res, 3 * res :] = cube[4]
        result[:res, res : 2 * res] = cube[2].flip(0, 1)
        result[2 * res : 3 * res, res : 2 * res] = cube[3].flip(0, 1)
    else:
        result[res : 2 * res, :res] = cube[1]
        result[res : 2 * res, res : 2 * res] = cube[4]
        result[res : 2 * res, 2 * res : 3 * res] = cube[0]
        result[res : 2 * res, 3 * res :] = cube[5]
        result[:res, res : 2 * res] = cube[2]
        result[2 * res : 3 * res, res : 2 * res] = cube[3]

    return result


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':

        def lambda_rule(it):
            lr_l = 1.0 - max(0, it - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_xavier_multiplier(m, gain):
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // m.stride[0] // m.stride[
            1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * np.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std


def xavier_uniform_(m, gain):
    std = get_xavier_multiplier(m, gain)
    m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))


def init_weights(net, init_type='xavier_uniform', gain=1):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'xavier_uniform':
                xavier_uniform_(m, gain)
            elif init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [{}] is not implemented'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_seq(s, init_type='xavier_uniform'):
    '''initialize sequential model'''
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            init_weights(a, init_type, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            init_weights(a, init_type, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        else:
            init_weights(a, init_type)
    init_weights(s[-1])


def positional_encoding(positions, freqs):
    '''encode positions with positional encoding
        positions: :math:`(...,D)`
        freqs: int
    Return:
        pts: :math:`(...,2DF)`
    '''
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] +
                                                      (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts
