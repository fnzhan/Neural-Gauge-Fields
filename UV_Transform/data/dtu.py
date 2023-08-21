import numpy as np
import os
import h5py
import torch
import torch.utils.data as data
import importlib
import trimesh

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return self.__class__.__name__

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


def get_rays_dir(pixelcoords, height, width, focal, rot, princpt):
    # pixelcoords: H x W x 2
    x = (pixelcoords[..., 0] - princpt[0]) / focal[0]
    y = (pixelcoords[..., 1] - princpt[1]) / focal[1]
    z = np.ones_like(x)
    dirs = np.stack([x, y, z], axis=-1)

    dirs = np.sum(rot[None, None, :, :] * dirs[..., None], axis=-2)
    dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5)

    return dirs


class DtuDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            "--random_sample",
            type=str,
            default="no_crop",
            choices=["no_crop", "random", "balanced", "patch"],
            help="method for sampling from the image",
        )
        parser.add_argument(
            "--random_sample_size",
            type=int,
            default=64,
            help="square root of the number of random samples",
        )
        parser.add_argument(
            "--use_test_data", type=int, default=-1, help="train or test dataset",
        )
        parser.add_argument(
            "--test_views", type=str, default="6,13,35,30", help="held out views",
        )

        return parser

    def initialize(self, opt):
        self.opt = opt
        self.data_dir = opt.data_root

        self.campos = np.load(self.data_dir + "/in_camOrgs.npy")
        self.camat = np.load(self.data_dir + "/in_camAts.npy")
        self.focal = np.load(self.data_dir + "/in_camFocal.npy")
        self.princpt = np.load(self.data_dir + "/in_camPrincpt.npy")
        self.extrinsics = np.load(self.data_dir + "/in_camExtrinsics.npy")
        self.point_cloud = trimesh.load(self.data_dir + "/pcd_down_unit.ply")
        self.point_cloud = np.array(self.point_cloud.vertices)

        self.total = self.campos.shape[0]

        if os.path.isfile(self.data_dir + '/exclude.txt'):
            with open(self.data_dir + '/exclude.txt', 'r') as f:
                exclude_views = [int(x) for x in f.readline().strip().split(',')]
        else:
            exclude_views = []

        if os.path.isfile(self.data_dir + '/test_views.txt'):
            with open(self.data_dir + '/test_views.txt', 'r') as f:
                test_views = [int(x) for x in f.readline().strip().split(',')]
        else:
            test_views = [int(x) for x in opt.test_views.split(',')]

        if self.opt.use_test_data > 0:
            self.indexes = test_views
            assert len(self.indexes) > 0
        else:
            self.indexes = [i for i in range(self.total) if i not in test_views and i not in exclude_views]
            assert len(self.indexes) == self.campos.shape[0] - len(test_views) - len(exclude_views)

        print("Total views:", self.total)

        print("Loading data in memory")
        imgData = h5py.File(self.data_dir + "/data.hdf5", "r")
        self.gt_image = imgData["in"][0 : self.total]

        if "in_masks" in imgData:
            self.gt_mask = imgData["in_masks"][0 : self.total]
        else:
            print("No gt masks.")
        imgData.close()
        print("Finish loading")

        self.height = self.gt_image[0].shape[0]
        self.width = self.gt_image[0].shape[1]
        print("center cam pos: ", self.campos[33])
        self.center_cam_pos = self.campos[33]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        idx = self.indexes[idx]

        item = {}
        gt_image = self.gt_image[idx]
        gt_mask = self.gt_mask[idx]

        gt_mask = gt_mask[:, :, None] / 255.0
        item["gt_mask"] = torch.from_numpy(gt_mask).permute(2, 0, 1).float()

        gt_image = gt_image / 255.0
        height = gt_image.shape[0]
        width = gt_image.shape[1]

        camrot = self.extrinsics[idx][0:3, 0:3]
        focal = self.focal[idx]
        princpt = self.princpt[idx]
        item["campos"] = torch.from_numpy(self.campos[idx]).float()

        dist = np.linalg.norm(self.campos[idx])
        near = dist - 1.0
        far = dist + 1.0
        item["far"] = torch.FloatTensor([far])
        item["near"] = torch.FloatTensor([near])

        subsamplesize = self.opt.random_sample_size
        if self.opt.random_sample == "patch":
            indx = np.random.randint(0, width - subsamplesize + 1)
            indy = np.random.randint(0, height - subsamplesize + 1)
            px, py = np.meshgrid(
                np.arange(indx, indx + subsamplesize).astype(np.float32),
                np.arange(indy, indy + subsamplesize).astype(np.float32),
            )
        elif self.opt.random_sample == "random":
            px = np.random.randint(
                0, width, size=(subsamplesize, subsamplesize)
            ).astype(np.float32)
            py = np.random.randint(
                0, height, size=(subsamplesize, subsamplesize)
            ).astype(np.float32)
        elif self.opt.random_sample == "balanced":
            px, py, trans = self.proportional_select(gt_mask)
            item["transmittance"] = torch.from_numpy(trans).float().contiguous()
        else:
            px, py = np.meshgrid(
                np.arange(width).astype(np.float32),
                np.arange(height).astype(np.float32),
            )

        pixelcoords = np.stack((px, py), axis=-1).astype(np.float32)  # H x W x 2
        raydir = get_rays_dir(
            pixelcoords, self.height, self.width, focal, camrot, princpt
        )

        raydir = np.reshape(raydir, (-1, 3))
        item["raydir"] = torch.from_numpy(raydir).float()
        gt_image = gt_image[py.astype(np.int32), px.astype(np.int32), :]
        gt_image = np.reshape(gt_image, (-1, 3))
        item["gt_image"] = torch.from_numpy(gt_image).float().contiguous()

        item["point_cloud"] = torch.tensor(self.point_cloud).float().contiguous()
        item["background_color"] = torch.from_numpy(np.zeros(3)).float().contiguous()

        return item

    def proportional_select(self, mask):
        # random select 2 / 3 pixels from foreground
        # random select 1 / 3 pixels from background
        subsamplesize = self.opt.random_sample_size

        fg_index = np.where(mask > 0)
        #  print(fg_index)
        fg_yx = np.stack(fg_index, axis=1)  # n x 2
        fg_num = fg_yx.shape[0]

        bg_index = np.where(mask == 0)
        bg_yx = np.stack(bg_index, axis=1)
        bg_num = bg_yx.shape[0]

        select_fg_num = int(
            self.opt.random_sample_size * self.opt.random_sample_size * 2.0 / 3.0
        )

        if select_fg_num > fg_num:
            select_fg_num = fg_num

        select_bg_num = subsamplesize * subsamplesize - select_fg_num

        fg_index = np.random.choice(range(fg_num), select_fg_num)
        bg_index = np.random.choice(range(bg_num), select_bg_num)

        px = np.concatenate((fg_yx[fg_index, 1], bg_yx[bg_index, 1]))
        py = np.concatenate((fg_yx[fg_index, 0], bg_yx[bg_index, 0]))

        px = px.astype(np.float32)  # + 0.5 * np.random.uniform(-1, 1)
        py = py.astype(np.float32)  # + 0.5 * np.random.uniform(-1, 1)

        px = np.clip(px, 0, mask.shape[1] - 1)
        py = np.clip(py, 0, mask.shape[0] - 1)

        px = np.reshape(px, (subsamplesize, subsamplesize))
        py = np.reshape(py, (subsamplesize, subsamplesize))

        trans = np.zeros(len(fg_index) + len(bg_index))
        trans[len(fg_index) :] = 1

        return px, py, trans

    def get_item(self, idx):
        item = self.__getitem__(idx)

        for key, value in item.items():
            if isinstance(value, np.int64):
                item[key] = torch.LongTensor([value])
            else:
                item[key] = value.unsqueeze(0)

        return item



def find_dataset_class_by_name(name):
    '''
    Input
    name: string with underscore representation

    Output
    dataset: a dataset class with class name {camelcase(name)}Dataset

    Searches for a dataset module with name {name}_dataset in current
    directory, returns the class with name {camelcase(name)}Dataset found in
    the module.
    '''
    cls_name = underscore2camelcase(name) + 'Dataset'
    filename = "data.{}".format(name)
    module = importlib.import_module(filename)

    assert cls_name in module.__dict__, 'Cannot find dataset class name "{}" in "{}"'.format(
        cls_name, filename)
    cls = module.__dict__[cls_name]
    assert issubclass(cls, BaseDataset), 'Dataset class "{}" must inherit from BaseDataset'.format(cls_name)

    return cls


def get_option_setter(dataset_name):
    dataset_class = find_dataset_class_by_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    dataset = find_dataset_class_by_name(opt.dataset_name)
    instance = dataset()
    instance.initialize(opt)
    print("dataset [{}] was created".format(instance.name()))
    return instance


def create_data_loader(opt):
    data_loader = DefaultDataLoader()
    data_loader.initialize(opt)
    return data_loader


class DefaultDataLoader:
    def name(self):
        return self.__class__.name

    def initialize(self, opt):
        assert opt.batch_size >= 1
        assert opt.n_threads >= 0
        assert opt.max_dataset_size >= 1

        self.opt = opt
        self.dataset = create_dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=opt.batch_size,
                                                      shuffle=not opt.serial_batches,
                                                      num_workers=int(opt.n_threads))

    def load_data(self):
        return self.dataset

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

    def get_item(self, index):
        return self.dataset.get_item(index)

def underscore2camelcase(s):
    assert s == s.lower(), 'Invalid underscore string: no upper case character is allowed, in "{}"'.format(s)
    assert all([x.isdigit() or x.isalpha() or x == '_' for x in s]),\
        'Invalid underscore, all character must be letters or numbers or underscore'
    terms = s.split('_')
    for x in terms:
        assert x, 'Invalid underscore string: no consecutive _ is allowed, in "{}"'.format(s)
        assert x[0].upper() != x[0], \
            'Invalid underscore string: phrases must start with a character, in "{}'.format(s)

    return ''.join([x[0].upper() + x[1:] for x in terms if x])


def camelcase2underscore(s):
    assert s[0].isupper(), 'Invalid camel case, first character must be upper case, in "{}"'.format(s)
    assert all([x.isdigit() or x.isalpha() for x in s]),\
        'Invalid camel case, all character must be letters or numbers'

    out = s[0].lower()
    for x in s[1:]:
        if x.lower() != x:
            out += '_' + x.lower()
        else:
            out += x
    return out