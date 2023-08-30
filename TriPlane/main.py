import os
from tqdm.auto import tqdm
import datetime
import numpy as np
import sys
import imageio
import matplotlib.pyplot as plt
import timeit
import torch

from dataLoader import dataset_dict
from dataLoader.ray_utils import get_rays
from models.Field_mlp import *
from utils import *
from opt import config_parser


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

@torch.no_grad()
def mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    field = eval(args.model_name)(**kwargs)
    field.load(ckpt)

    alpha,_ = field.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=field.aabb.cpu(), level=0.005)


@torch.no_grad()
def test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    field = eval(args.model_name)(**kwargs)
    field.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset, field, args, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset, field, args, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset, field, c2ws, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, device=device)


def renderer(rays, field, chunk=1024, N_samples=-1, white_bg=True, is_train=False, device='cuda'):
    rgbs, depth_maps = [], []
    N_rays_all = rays.shape[0]
    # print('ray', rays.shape)# torch.Size([4096, 6]).
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        # print('********', rays_chunk.shape)
        output = field(rays_chunk, is_train=is_train, white_bg=white_bg, N_samples=N_samples, iteration=30001)
        rgbs.append(output['rgb_map'])
        depth_maps.append(output['depth_map'])
    # print('stop')
    return torch.cat(rgbs), torch.cat(depth_maps)

@torch.no_grad()
def evaluation(test_dataset, field, args, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])

        rgb_map, depth_map = renderer(rays, field, chunk=4096, N_samples=N_samples, white_bg=white_bg, device=device)
        # output = field(rays.to(device), is_train=False, white_bg=white_bg, N_samples=N_samples)
        # rgb_map = output['rgb_map']
        # depth_map = output['depth_map']

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', field.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', field.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

    return PSNRs


@torch.no_grad()
def evaluation_path(test_dataset, field, c2ws, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims, l_alex, l_vgg = [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):
        W, H = test_dataset.img_wh
        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, depth_map = renderer(rays, field, chunk=8192, N_samples=N_samples, white_bg=white_bg, device=device)
        # output = field(rays.to(device), is_train=False, white_bg=white_bg, N_samples=N_samples)
        # rgb_map = output['rgb_map']
        # depth_map = output['depth_map']

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))
    return PSNRs


def train(args):

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    log_txt = open(os.path.join(logfolder, 'log.txt'), 'w')

    aabb = train_dataset.scene_bbox.to(device)
    # reso_cur = N_to_reso(args.N_voxel_init, aabb)
    reso_cur = N_to_reso(256*256*256, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))


    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        field = eval(args.model_name)(**kwargs)
        field.load(ckpt)
        print('*****continue_training*****')
    else:
        field = eval(args.model_name)(aabb, reso_cur, device, near_far=near_far, alphaMask_thres=args.alpha_mask_thre,
                    distance_scale=args.distance_scale, step_ratio=args.step_ratio)


    grad_vars = field.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))

    # N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init),
    #                 np.log(args.N_voxel_final), len(upsamp_list) + 1))).long()).tolist()[1:]
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init),
                    np.log(args.N_voxel_final), len(upsamp_list)))).long()).tolist()

    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    allrays, allrgbs = field.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    Ortho_reg_weight = args.Ortho_weight
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")
    L1_reg_weight = 8e-5

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:
        start = timeit.default_timer()

        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx].to(device), allrgbs[ray_idx].to(device)
        #rgb_map, alphas_map, depth_map, weights, uncertainty.
        # rgb_map, _, depth_map, reg_loss, _ = renderer(rays_train, field, chunk=args.batch_size.
        #                         N_samples=nSamples, white_bg = white_bg, device=device, is_train=True).
        output = field(rays_train, is_train=True, white_bg=white_bg, N_samples=nSamples, iteration=iteration)
        rgb_map = output['rgb_map']

        # print('******', field)
        # 1/0

        rgb_loss = torch.mean((rgb_map - rgb_train) ** 2)
        total_loss = rgb_loss

        # loss
        if Ortho_reg_weight > 0:
            loss_reg = field.vector_comp_diffs()
            total_loss += Ortho_reg_weight*loss_reg
        if L1_reg_weight > 0:
            loss_reg_L1 = field.density_L1()
            total_loss += L1_reg_weight*loss_reg_L1

        # if TV_weight_density>0:
        #     TV_weight_density *= lr_factor
        #     loss_tv = field.TV_loss_density(tvreg) * TV_weight_density
        #     total_loss = total_loss + loss_tv
        # if TV_weight_app>0:
        #     TV_weight_app *= lr_factor
        #     loss_tv = field.TV_loss_app(tvreg)*TV_weight_app
        #     total_loss = total_loss + loss_tv



        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        rgb_loss = rgb_loss.detach().item()
        PSNRs.append(-10.0 * np.log(rgb_loss) / np.log(10.0))

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor
            # if 'rex' in param_group and iteration > 10000:


        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            log_txt.write(f'Iteration {iteration:05d}:'
                          + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                          + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                          + f' mse = {rgb_loss:.6f}'
                          + '\n')
            PSNRs = []


        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test = evaluation(test_dataset, field, args, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, compute_extra_metrics=False)
            # summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
            log_txt.write(f'Iteration {iteration:05d}:' + f' test/psnr = {float(np.mean(PSNRs_test)):.2f}' + '\n')
        log_txt.flush()

        if iteration in update_AlphaMask_list:

            # if reso_cur[0] * reso_cur[1] * reso_cur[2] <= 256 ** 3:  # update volume resolution
            #     reso_mask = reso_cur
            reso_mask = [256, 256, 256] #reso_cur
            new_aabb = field.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                field.shrink(new_aabb)
                L1_reg_weight = 4e-5
                print("continuing L1_reg_weight", L1_reg_weight)

            # if iteration == update_AlphaMask_list[1]:
                allrays, allrgbs = field.filtering_rays(allrays, allrgbs)
                trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)


        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, field.aabb)
            # reso_cur = N_to_reso(300*300*300, field.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
            field.up_sampling(reso_cur)
            # if args.lr_upsample_reset:
            print("reset lr to initial")
            lr_scale = 1
            # else:
            #     lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = field.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))


        if iteration % 10000 == 0:
            field.save(f'{logfolder}/model.th')

        # stop = timeit.default_timer()
        # print('Time: ', stop - start)

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset, field, args, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset, field, args, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, device=device)
        # summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        log_txt.write('test/psnr_all' + f'{float(np.mean(PSNRs_test)):.2f}' + '\n')
        log_txt.flush()
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses.
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset, field, c2ws, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, device=device)


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    if args.export_mesh:
        mesh(args)
    if args.render_only and (args.render_test or args.render_path):
        test(args)
    else:
        train(args)

