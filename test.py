#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------
import argparse
import gradio
import os
import torch
import numpy as np
import tempfile
import functools
import trimesh
import copy
import glob
from scipy.spatial.transform import Rotation

from dust3r.inference import inference, load_model
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import matplotlib.pyplot as pl

pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1


def _convert_scene_output_to_glb(
    outdir,
    imgs,
    pts3d,
    mask,
    focals,
    cams2world,
    cam_size=0.05,
    cam_color=None,
    as_pointcloud=False,
    transparent_cams=False,
):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(
            scene,
            pose_c2w,
            camera_edge_color,
            None if transparent_cams else imgs[i],
            focals[i],
            imsize=imgs[i].shape[1::-1],
            screen_width=cam_size,
        )

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, "scene.glb")
    print("(exporting 3D scene to", outfile, ")")
    scene.export(file_obj=outfile)

    return outfile


def get_3D_model_from_scene(
    outdir,
    scene,
    min_conf_thr=3,
    as_pointcloud=False,
    mask_sky=False,
    clean_depth=False,
    transparent_cams=False,
    cam_size=0.05,
):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None

    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()

    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())

    return _convert_scene_output_to_glb(
        outdir,
        rgbimg,
        pts3d,
        msk,
        focals,
        cams2world,
        as_pointcloud=as_pointcloud,
        transparent_cams=transparent_cams,
        cam_size=cam_size,
    )


def get_reconstructed_scene(
    outdir,
    model,
    device,
    image_size,
    filelist,
    schedule,
    niter,
    min_conf_thr,
    as_pointcloud,
    mask_sky,
    clean_depth,
    transparent_cams,
    cam_size,
    scenegraph_type,
    winsize,
    refid,
):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size)

    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1

    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)

    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(
        imgs, 
        scene_graph=scenegraph_type, 
        prefilter=None, 
        symmetrize=True
    )
    output = inference(pairs, model, device, batch_size=batch_size)

    mode = (
        GlobalAlignerMode.PointCloudOptimizer
        if len(imgs) > 2
        else GlobalAlignerMode.PairViewer
    )
    scene = global_aligner(output, device=device, mode=mode)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(
            init="mst", niter=niter, schedule=schedule, lr=lr
        )

    outfile = get_3D_model_from_scene(
        outdir,
        scene,
        min_conf_thr,
        as_pointcloud,
        mask_sky,
        clean_depth,
        transparent_cams,
        cam_size,
    )

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap("jet")
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d / confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

    return scene, outfile, imgs


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_size", type=int, default=512, choices=[512, 224], help="image size"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=False,
        default="checkpoints/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        help="path to the model weights",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="pytorch device")
    parser.add_argument(
        "--tmp_dir", type=str, default="tmp", help="value for tempfile.tempdir"
    )
    parser.add_argument("--input_dir", type=str, default="imgs", help="input files")
    parser.add_argument("--min_conf_thr", type=float, default=3.0, help="min_conf_thr")
    return parser


# if __name__ == "__main__":
parser = get_args_parser()
args = parser.parse_args()

if args.tmp_dir is not None:
    tmp_path = args.tmp_dir
    os.makedirs(tmp_path, exist_ok=True)
    tempfile.tempdir = tmp_path

model = load_model(args.weights, args.device)

# find files ending with ".jpg", ".png", ".jpeg" in input_dir
input_files = []
for ext in ["jpg", "png", "jpeg"]:
    input_files.extend(glob.glob(os.path.join(args.input_dir, f"*.{ext}")))

scene, outfile, imgs = get_reconstructed_scene(
    tmp_path,
    model,
    args.device,
    args.image_size,
    input_files,
    "cosine",
    300,
    args.min_conf_thr,
    True,
    True,
    True,
    False,
    0.005,
    "complete",
    1,
    0,
)

# inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, scenegraph_type, winsize, refid]
