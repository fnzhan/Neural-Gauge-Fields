import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from util import init_weights, positional_encoding


class GaugeNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, mid_size=64, hidden_size=128, num_layers=2):
        """
        template_size: input size
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mid_size = mid_size
        self.hidden_neurons = hidden_size
        self.num_layers = num_layers

        self.linear1 = nn.Linear(self.input_dim + 2 * self.input_dim * 10, self.mid_size)
        self.linear2 = nn.Linear(self.mid_size, self.hidden_neurons)

        init_weights(self.linear1)
        init_weights(self.linear2)

        self.linear_list = nn.ModuleList(
            [nn.Linear(self.hidden_neurons, self.hidden_neurons)for i in range(self.num_layers)])

        for l in self.linear_list:
            init_weights(l)

        self.last_linear = nn.Linear(self.hidden_neurons, self.output_dim)
        init_weights(self.last_linear)

        self.activation = F.relu

    def forward(self, x):

        x = self.linear1(torch.cat([x, positional_encoding(x, 10)], dim=-1))
        # x = self.linear1(x) + latent[:, None]

        x = self.activation(x)
        x = self.activation(self.linear2(x))
        for i in range(self.num_layers):
            x = self.activation(self.linear_list[i](x))
        return self.last_linear(x)


class GaugeTransform(nn.Module):
    def __init__(self, primitive_type):
        super().__init__()

        if primitive_type == 'square':
            self.output_dim = 2
        else:
            self.output_dim = 3

        self.encoder = GaugeNetwork(3, self.output_dim)

    def forward(self, points):
        """
        Args:
            points: :math:`(N,*,3)`
        """
        input_shape = points.shape
        points = points.view(points.shape[0], -1, 3)
        output = self.encoder(points)  # (N, *, 3)[primitives]
        output = output.view(input_shape[:-1] + output.shape[-1:])

        if self.output_dim == 2:
            uv = torch.tanh(output)
        else:
            uv = F.normalize(output, dim=-1)
        return uv



class InverseNetwork(nn.Module):
    """
    Modified AtlasNet core function
    """

    def __init__(self, input_point_dim, mid_size=64, hidden_size=512, num_layers=2,):
        """
        template_size: input size
        """
        super().__init__()

        self.input_size = input_point_dim
        self.dim_output = 3
        self.mid_size = mid_size
        self.hidden_neurons = hidden_size
        self.num_layers = num_layers

        self.linear1 = nn.Linear(self.input_size, self.mid_size)
        self.linear2 = nn.Linear(self.mid_size, self.hidden_neurons)

        init_weights(self.linear1)
        init_weights(self.linear2)

        self.linear_list = nn.ModuleList([nn.Linear(self.hidden_neurons, self.hidden_neurons) for i in range(self.num_layers)])

        for l in self.linear_list:
            init_weights(l)

        self.last_linear = nn.Linear(self.hidden_neurons, self.dim_output)
        init_weights(self.last_linear)
        self.activation = F.relu

    def forward(self, x):
        assert x.shape[-1] == self.input_size
        self.uv_ = x

        x = self.linear1(x)
        x = self.activation(x)
        x = self.activation(self.linear2(x))
        for i in range(self.num_layers):
            x = self.activation(self.linear_list[i](x))
        xyz = self.last_linear(x)
        return xyz


class SquareTemplate:
    def __init__(self):
        self.regular_num_points = 0

    def get_random_points(self, npoints):
        with torch.no_grad():
            rand_grid = (torch.rand((npoints, 2)) * 2 - 1).cuda().float()
            return rand_grid

    def get_regular_points(self, npoints=2500):
        """
        Get regular points on a Square
        """
        assert int(npoints ** 0.5) ** 2 == npoints
        # assert device is not None, "device needs to be provided for get_regular_points"

        side_length = int(npoints ** 0.5)

        uv = np.stack(np.meshgrid(*([np.linspace(-1, 1, side_length)] * 2), indexing="ij"), axis=-1,).reshape((-1, 2))

        points = torch.FloatTensor(uv).cuda()
        return points.requires_grad_()


class SphereTemplate:
    def get_random_points(self, npoints):
        """
        Get random points on a Sphere
        Return Tensor of Size [x, 3, x ... x]
        """
        with torch.no_grad():
            points = torch.randn((npoints, 3)).cuda().float() * 2 - 1
            points = F.normalize(points, dim=-1)
        return points

    def get_regular_points(self):
        """
        Get regular points on a Sphere
        Return Tensor of Size [x, 3]
        """
        import trimesh
        mesh = trimesh.creation.icosphere(6)
        return torch.tensor(mesh.vertices).cuda().float()

# Atlasnet
class InverseGauge(nn.Module):
    def __init__(self, num_points_per_primitive, primitive_type="square",):
        super().__init__()

        if primitive_type == "square":
            self.template = SquareTemplate()
            self.input_point_dim = 2
        elif primitive_type == "sphere":
            self.template = SphereTemplate()
            self.input_point_dim = 3
        else:
            raise Exception("Unknown primitive type {}".format(primitive_type))
        self.num_points_per_primitive = num_points_per_primitive

        # Initialize deformation networks
        self.inverse_network = InverseNetwork(self.input_point_dim)
        with torch.no_grad():
            self.label = torch.zeros(num_points_per_primitive).long()

    def forward(self, regular_point_count=None):
        """
        Deform points from self.template using the embedding latent_vector
        :return: A deformed pointcloud os size : batch, nb_prim, num_point, 3
        """
        if regular_point_count is None:
            input_points = self.template.get_random_points(self.num_points_per_primitive,)
        else:
            input_points =self.template.get_regular_points()
        output_points = self.inverse_network(input_points.unsqueeze(0)).unsqueeze(1)
        return input_points, output_points.contiguous()

    def map(self, uv):
        """
        uvs: (N,...,P,2/3) ([1, 576, 64, 3])
        """
        assert uv.shape[-1] == self.input_point_dim
        input_shape = uv.shape
        output = self.inverse_network(uv.view(input_shape, -1, self.input_point_dim))

        return output.view(input_shape[:-1] + (3,))
