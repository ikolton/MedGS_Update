#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint

import torch.nn.functional as F

from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene
from models import (
    optimizationParamTypeCallbacks,
    gaussianModel
)

from utils.general_utils import safe_state
from utils.loss_utils import penalize_outside_range
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams
import matplotlib.pyplot as plt
import json
import time
import numpy as np


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def norm_gauss(m, sigma, t):
    log = ((m - t)**2 / sigma**2) / -2
    return torch.exp(log)

def interpolate(A, B, alpha):
    I = (1 - alpha) * A + alpha * B
    return I

def training(gs_type, dataset: ModelParams, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, save_xyz, use_dff):
    time_start = time.process_time()
    init_time = time.time()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    frames = len(os.listdir(f'{dataset.source_path}/original'))
    print("frames", frames)
    gaussians = gaussianModel[gs_type](dataset.sh_degree, dataset.poly_degree, frames, use_dff = use_dff)

    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    bg = torch.rand((3), device="cuda") if opt.random_background else background

    viewpoint_stack = None

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0 
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    viewpoint_cameras = scene.getTrainCameras()
    num_frames = len(viewpoint_cameras)

    gts = []
    for camera in viewpoint_cameras:
        gt = camera.get_image(bg, opt.random_background).cuda()
        gts.append(gt)

    prev_next_overlap = 2 if dataset.camera == "mirror" else 1
    print("prev_next_overlap", prev_next_overlap)

    interpolation = 1

    for iteration in range(first_iter, opt.iterations + 1):
        
        os.makedirs(f"{scene.model_path}/xyz", exist_ok=True)
        if save_xyz and (iteration % 5000 == 1 or iteration == opt.iterations):
            torch.save(gaussians.get_xyz, f"{scene.model_path}/xyz/{iteration}.pt")

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        idx = randint(0, len(viewpoint_cameras) - 1)

        camera_prev = viewpoint_cameras[idx - prev_next_overlap] if idx - prev_next_overlap >= 0 else None
        camera = viewpoint_cameras[idx]
        camera_next = viewpoint_cameras[idx + prev_next_overlap] if idx + prev_next_overlap <= num_frames - 1 else None

        cameras = {"camera_prev": camera_prev ,"camera": camera, "camera_next": camera_next}

        if (iteration - 1) == debug_from:
            pipe.debug = True

        outputs = {}
        outputs_inter = {}
        if iteration == 10000:
            interpolation = 2
            print("changed interpolation")

        for name, camera in cameras.items():
            alpha = np.random.uniform(0.2, 0.8)
            if not camera:
                continue

            if name == "camera_prev":
                A = gts[idx-prev_next_overlap]
                B = gts[idx]
                interpolated_img = interpolate(A, B, alpha)
            elif name == "camera":
                A = gts[idx]
                if idx + prev_next_overlap < len(gts):
                    B = gts[idx + prev_next_overlap]
                    interpolated_img = interpolate(A, B, alpha)
                else:
                    render_pkg = render(camera, gaussians, pipe, bg, train=True, iter=iteration)
                    render_pkg["gt"] = gts[idx]
                    outputs[name] = render_pkg
                    continue
            elif name == "camera_next":
                render_pkg = render(camera, gaussians, pipe, bg, train=True, iter=iteration)
                render_pkg["gt"] = gts[idx + prev_next_overlap]
                outputs[name] = render_pkg
                continue

            for i in range(interpolation):
                if i == 0:
                    render_pkg = render(camera, gaussians, pipe, bg, train=True, iter=iteration)
                    render_pkg["gt"] = A
                    outputs[name] = render_pkg
                else:
                    render_pkg = render(camera, gaussians, pipe, bg, train=True, iter=iteration, alpha=alpha)
                    render_pkg["gt"] = interpolated_img
                    outputs_inter[f"{name}_interpolate_{i}"] = render_pkg


        data = {}
        data_inter = {}

        for k in outputs["camera"].keys():
            if k == "viewspace_points":
                data[k] = [output[k] for output in outputs.values()]
                data_inter[k] = [output[k] for output in outputs_inter.values()]
            elif k in ["visibility_filter", "radii"]:
                data[k] = [output[k] for output in outputs.values()]
                data_inter[k] = [output[k] for output in outputs_inter.values()]
            elif k in ["render", "gt", "mask"]:
                data[k] = torch.stack([output[k] for output in outputs.values()], dim=0)
                if interpolation > 1:
                    data_inter[k] = torch.stack([output[k] for output in outputs_inter.values()], dim=0)


        sigma = gaussians.get_sigma
        sigma_loss = penalize_outside_range(sigma, 2/(num_frames), 1)

        render_curr = outputs["camera"]["render"]
        gt_curr = outputs["camera"]["gt"]
        Ll1 = l1_loss(data["render"], data["gt"])
        if interpolation > 1:
            Ll1_inter = l1_loss(data_inter["render"], data_inter["gt"])
        else:
            Ll1_inter = 0

        data['mask_t'] = torch.stack(data['visibility_filter'], dim=-1).any(dim=1)
        ssim_loss = 1.0 - ssim(data['render'], data['gt'])
        L_flow_temporal = torch.tensor(0.0, device='cuda', dtype=data['render'].dtype)
        # #
        gt_prev_warped = None
        gt_next_warped = None

        loss = 2.0 * Ll1 + 0.5 * Ll1_inter +  0.25 * sigma_loss + 1 * ssim_loss

        psnr_ = psnr(render_curr, gt_curr).mean().double()
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background), L_flow_temporal, render_curr, gt_curr, gt_prev_warped, gt_next_warped, gaussians.get_sigma)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            if iteration < 20000:
                radii_batch = torch.stack(data['radii'], dim=-1).max(dim=-1)[0]
                visibility_filter_batch = data["mask_t"]
                gaussians.max_radii2D[visibility_filter_batch] = torch.max(
                    gaussians.max_radii2D[visibility_filter_batch],
                    radii_batch[visibility_filter_batch]
                )
                xyscreen = data['viewspace_points']
                gaussians.add_densification_stats(xyscreen, visibility_filter_batch)
                    
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    print("\n[ITER {}] Densifying Gaussians".format(iteration))
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent,
                                                size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    print("\n[ITER {}] Reset opacity ".format(iteration))
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    time_elapsed = time.process_time() - time_start
    time_dict = {}
    time_dict["time"] = time_elapsed
    time_dict["elapsed"] = time.time() - init_time
    with open(scene.model_path + f"/time.json", 'w') as fp:
        json.dump(time_dict, fp, indent=True)

def training_binary_segmentation(gs_type, dataset: ModelParams, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, save_xyz, use_dff):
    time_start = time.process_time()
    init_time = time.time()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    frames = len(os.listdir(f'{dataset.source_path}/original'))
    print("frames", frames)
    gaussians = gaussianModel[gs_type](dataset.sh_degree, dataset.poly_degree, frames, use_dff = use_dff)

    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    bg = torch.rand((3), device="cuda") if opt.random_background else background

    viewpoint_stack = None

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0 
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    viewpoint_cameras = scene.getTrainCameras()
    num_frames = len(viewpoint_cameras)

    gts = []
    for camera in viewpoint_cameras:
        gt = camera.get_image(bg, opt.random_background).cuda()
        gts.append(gt)

    prev_next_overlap = 2 if dataset.camera == "mirror" else 1
    print("prev_next_overlap", prev_next_overlap)

    for iteration in range(first_iter, opt.iterations + 1):
        os.makedirs(f"{scene.model_path}/xyz", exist_ok=True)
        if save_xyz and (iteration % 5000 == 1 or iteration == opt.iterations):
            torch.save(gaussians.get_xyz, f"{scene.model_path}/xyz/{iteration}.pt")

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        idx = randint(0, len(viewpoint_cameras) - 1)

        camera_prev = viewpoint_cameras[idx - prev_next_overlap] if idx - prev_next_overlap >= 0 else None
        camera = viewpoint_cameras[idx]
        camera_next = viewpoint_cameras[idx + prev_next_overlap] if idx + prev_next_overlap <= num_frames - 1 else None


        cameras = {"camera_prev": camera_prev ,"camera": camera, "camera_next": camera_next}

        if (iteration - 1) == debug_from:
            pipe.debug = True

        outputs = []

        interpolation = 1
        outputs_inter = {}

        for name, camera in cameras.items():
            if not camera:
                continue    
            render_pkg = render(camera, gaussians, pipe, bg, train=True, iter=iteration, seg=True)
            render_pkg["gt"] = camera.get_image(bg, opt.random_background).cuda()
            outputs.append(render_pkg)

                
        data = {}
    
        for k in outputs[0].keys():
            if k == "viewspace_points":
                data[k] = [output[k] for output in outputs]
            elif k in ["visibility_filter", "radii"]:
                data[k] = [output[k] for output in outputs]
            elif k in ["render", "gt", "mask"]:
                data[k] = torch.stack([output[k] for output in outputs], dim=0)
                

        sigma = gaussians.get_sigma
        if iteration < 20000:
            sigma_loss = penalize_outside_range(sigma, 2./num_frames, 1)
        else:
            sigma_loss = 0.0
        render_curr = outputs[0]["render"]
        gt_curr = outputs[0]["gt"]
        Ll1 = l1_loss(data["render"], data["gt"])

        if interpolation > 1:
            Ll1_inter = l1_loss(data_inter["render"], data_inter["gt"])
        else:
            Ll1_inter = 0

        data['mask_t'] = torch.stack(data['visibility_filter'], dim=-1).any(dim=1)
        ssim_loss = 1.0 - ssim(data['render'], data['gt'])
        L_flow_temporal = torch.tensor(0.0, device='cuda', dtype=data['render'].dtype)
        # #
        gt_prev_warped = None
        gt_next_warped = None
    
        loss = 2.0 * Ll1 + 0.5 * Ll1_inter + 0.0 * ssim_loss + 0.5 * sigma_loss #+ 0 * L_flow_temporal #+ 0.1 * sigma_loss ##
    
        psnr_ = psnr(data["render"], data["gt"]).mean().double()
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background), L_flow_temporal, render_curr, gt_curr, gt_prev_warped, gt_next_warped, gaussians.get_sigma, seg=True)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            if iteration < 20000:
                radii_batch = torch.stack(data['radii'], dim=-1).max(dim=-1)[0]
                visibility_filter_batch = data["mask_t"]
                gaussians.max_radii2D[visibility_filter_batch] = torch.max(
                    gaussians.max_radii2D[visibility_filter_batch],
                    radii_batch[visibility_filter_batch]
                )
                xyscreen = data['viewspace_points']
                gaussians.add_densification_stats(xyscreen, visibility_filter_batch)
                    
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    print("\n[ITER {}] Densifying Gaussians".format(iteration))
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent,
                                                size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    print("\n[ITER {}] Reset opacity ".format(iteration))
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    time_elapsed = time.process_time() - time_start
    time_dict = {}
    time_dict["time"] = time_elapsed
    time_dict["elapsed"] = time.time() - init_time
    with open(scene.model_path + f"/time.json", 'w') as fp:
        json.dump(time_dict, fp, indent=True)

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l2_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, temporal_loss, render_curr, gt_curr, gt_prev_warped, gt_next_warped, sigma, seg=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('train_loss_patches/temporal_loss', temporal_loss.item(), iteration)

    # Report test and samples of training set

    if iteration % 1000 ==0 :
        tb_writer.add_images("temporal/render_curr", render_curr, global_step=iteration, dataformats="CHW")
        tb_writer.add_images("temporal/gt_curr", gt_curr, global_step=iteration, dataformats="CHW")
        if gt_prev_warped is not None:
            tb_writer.add_images("temporal/prev_warped", gt_prev_warped, global_step=iteration, dataformats="CHW")
        if gt_next_warped is not None:
            tb_writer.add_images("temporal/next_warped", gt_next_warped, global_step=iteration, dataformats="CHW")
        tb_writer.add_histogram('sigma', sigma, global_step=iteration, bins='auto')


    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        config = {'name': 'test', 'cameras': scene.getTestCameras()}


        l1_test = 0.0
        psnr_test = 0.0
        psnrs = []
        times = []
        for idx, viewpoint in enumerate(config['cameras']):
            image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0, seg=seg)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            if tb_writer and (idx < 5):
                tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                        image[None], global_step=iteration)
                if iteration == testing_iterations[0]:
                    tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                            gt_image[None], global_step=iteration)

            l1_test += l1_loss(image, gt_image).mean().double()
            psnrs.append(psnr(image, gt_image).mean().double().cpu())
            times.append(viewpoint.time)
            psnr_test += psnrs[-1]
        psnr_test /= len(config['cameras'])
        l1_test /= len(config['cameras'])
        plt.plot(times, psnrs, 'o')
        plt.ylabel("PSNR")
        plt.xlabel("Frame")
        if not os.path.isdir(f"{scene.model_path}/plots/"):
            os.makedirs(f"{scene.model_path}/plots/")
        plt.savefig(f"{scene.model_path}/plots/{str(iteration)}.png")
        plt.clf()

        num_gaussians = scene.gaussians.get_xyz.shape[0]
        poly_degree = scene.gaussians._w1.shape[-1] // 2
        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} Num Points {} Poly Degree {}".format(iteration, config['name'], l1_test, psnr_test, num_gaussians, poly_degree))
        if tb_writer:
            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--gs_type', type=str, default="gs")
    parser.add_argument('--camera', type=str, default="mirror")
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 40_000, 100_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--save_xyz", action='store_true')
    parser.add_argument("--poly_degree", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--use_dff", type=bool, default=False)
    parser.add_argument("--pipeline", type=str, choices=["img", "seg"], default="img")

    lp = ModelParams(parser)
    args, _ = parser.parse_known_args(sys.argv[1:])
    lp.gs_type = args.gs_type
    lp.camera = args.camera
    lp.distance = args.distance
    lp.num_pts = args.num_pts
    lp.poly_degree = args.poly_degree
    
    op = optimizationParamTypeCallbacks[args.gs_type](parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    op.batch_size = args.batch_size

    args.save_iterations.append(args.iterations)

    print("torch cuda: ", torch.cuda.is_available())
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if args.pipeline == "seg":
        training_binary_segmentation(
            args.gs_type,
            lp.extract(args), op.extract(args), pp.extract(args),
            args.test_iterations, args.save_iterations, args.checkpoint_iterations,
            args.start_checkpoint, args.debug_from, args.save_xyz, args.use_dff
        )
    else:
        training(
            args.gs_type,
            lp.extract(args), op.extract(args), pp.extract(args),
            args.test_iterations, args.save_iterations, args.checkpoint_iterations,
            args.start_checkpoint, args.debug_from, args.save_xyz, args.use_dff
        )

    print("\nTraining complete.")
    print()
