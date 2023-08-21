import os
import copy
import torch
import numpy as np
from PIL import Image
import open3d
from options import TrainOptions
from model.model import create_model
from data.dtu import create_dataset
from util import Visualizer
from tqdm import tqdm
from util import merge_cube_to_single_texture


def main():
    torch.backends.cudnn.benchmark = True
    opt = TrainOptions().parse()
    opt.is_train = False
    opt.output_dir = 'test'

    assert opt.resume_dir is not None

    # resume_dir = opt.resume_dir
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Resume from {} epoch".format(opt.resume_epoch))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    dataset = create_dataset(opt)
    pos = dataset.center_cam_pos
    viewdir = -pos / np.linalg.norm(pos)

    # load model
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    rootdir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(rootdir, exist_ok=True)

    # meshes, textures = model.coordinate_deformation(opt.primitive_type, icosphere_division=7)
    # for i, (mesh, texture) in enumerate(zip(meshes, textures)):
    #     color = (255 * texture.data.cpu().numpy().clip(0, 1)).astype(np.uint8)
    #     c = np.ones((len(color), 4)) * 255
    #     c[:, :3] = color
    #     import trimesh
    #
    #     mesh.visual.vertex_colors = np.ones_like(c)
    #     trimesh.repair.fix_inversion(mesh)
    #     trimesh.repair.fix_normals(mesh)
    #     # mesh.show(viewer="gl", smooth=True)
    #
    #     mesh.visual.vertex_colors = c
    #     trimesh.repair.fix_inversion(mesh)
    #     trimesh.repair.fix_normals(mesh)
    #     # mesh.show(viewer="gl", smooth=True)
    #     mesh.export(os.path.join(rootdir, "mesh_{}.ply".format(i)))

    # visualize texture (sphere or square)
    net_texture = model.NeuTex.module.net_texture
    if opt.primitive_type == "sphere":
        texture = net_texture.export_textures(512, viewdir) ** (1 / 2.2)
        texture = merge_cube_to_single_texture(texture)
        texture = texture.clamp(0, 1).data.cpu().numpy()
        Image.fromarray((texture * 255).astype(np.uint8)).save(os.path.join(rootdir, opt.output_dir, f"cube_view.png"))

        texture = net_texture._export_sphere(512, viewdir) ** (1 / 2.2)
        texture = texture.clamp(0, 1).data.cpu().numpy()
        Image.fromarray((texture * 255).astype(np.uint8)).save(os.path.join(rootdir, opt.output_dir, f"sphere_view.png"))

        # for i, pos in enumerate(tqdm(dataset.campos)):
            # viewdir = -pos / np.linalg.norm(pos)
            # texture = net_texture.export_textures(512, viewdir) ** (1 / 2.2)
            # texture = merge_cube_to_single_texture(texture)
            # texture = texture.clamp(0, 1).data.cpu().numpy()
            # Image.fromarray((texture * 255).astype(np.uint8)).save(os.path.join(rootdir, f"images/cube_view_{i}.png"))
            #
            # texture = net_texture.textures[0]._export_sphere(512, viewdir) ** (1 / 2.2)
            # texture = texture.clamp(0, 1).data.cpu().numpy()
            # Image.fromarray((texture * 255).astype(np.uint8)).save(os.path.join(rootdir, f"images/sphere_view_{i}.png"))
    else:
        texture = net_texture.export_textures(512, viewdir) ** (1 / 2.2)
        texture = texture.clamp(0, 1).data.cpu().numpy()
        Image.fromarray((texture * 255).astype(np.uint8)).save(os.path.join(rootdir, opt.output_dir, "square_view.png"))


    # visualize rendered images
    test_opt = copy.deepcopy(opt)
    test_opt.random_sample = "no_crop"
    visualizer = Visualizer(test_opt)
    dataset2 = create_dataset(test_opt)
    height = dataset2.height
    width = dataset2.width
    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size
    for i in range(len(dataset2)):
        data = dataset2.get_item(i)

        gt_image = data["gt_image"].clone()
        campos = data["campos"]
        raydir = data["raydir"]

        visuals = None
        for k in range(0, height * width, chunk_size):
            start = k
            end = min([k + chunk_size, height * width])
            data["raydir"] = raydir[:, start:end, :]
            data["gt_image"] = gt_image[:, start:end, :]
            model.set_input(data)
            model.test()
            curr_visuals = model.get_current_visuals()

            if visuals is None:
                visuals = {}
                for key, value in curr_visuals.items():
                    # if key == 'ray_color':
                        chunk = value.cpu().numpy()
                        assert len(chunk.shape) == 3
                        assert chunk.shape[-1] == 3
                        visuals[key] = np.zeros((height * width, 3)).astype(chunk.dtype)
                        visuals[key][start:end, :] = chunk
            else:
                for key, value in curr_visuals.items():
                    # if key == 'ray_color':
                        visuals[key][start:end] = value.cpu().numpy()

        for key, value in visuals.items():
            visuals[key] = visuals[key].reshape(height, width, -1).squeeze()
        visualizer.display_current_results(visuals, i, campos.squeeze(), raydir.squeeze())
        print("Finished: {}/{}".format(i, len(dataset2)))






if __name__ == "__main__":
    main()
