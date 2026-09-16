"""Standardized run logging for cross-method comparison.

Writes <save_dir>/eval_run/{run.json,frames.csv,poses.csv} next to this repo's
own outputs. Loading, plotting and comparison live outside this repo.
"""

import json
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Persistent per-Gaussian map attributes. Optimizer state, gradients, render
# buffers and network weights are deliberately excluded.
MAP_TENSORS = ("_xyz", "_features_dc", "_features_rest",
               "_scaling", "_rotation", "_opacity")


def gaussian_stats(gaussians):
    """Active Gaussian count and the megabytes held by the map tensors."""
    if gaussians is None or gaussians._xyz.numel() == 0:
        return 0, 0.0
    nbytes = 0
    for name in MAP_TENSORS:
        tensor = getattr(gaussians, name, None)
        if tensor is not None:
            nbytes += tensor.numel() * tensor.element_size()
    return gaussians._xyz.shape[0], nbytes / 1024 ** 2


def gpu_allocated_mb():
    return torch.cuda.memory_allocated() / 1024 ** 2


def eval_dir(save_dir):
    path = os.path.join(save_dir, "eval_run")
    os.makedirs(path, exist_ok=True)
    return path


def open_frame_log(save_dir):
    log = open(os.path.join(eval_dir(save_dir), "frames.csv"), "w")
    log.write("frame,frame_time_ms,gpu_allocated_mb,num_gaussians,map_size_mb\n")
    return log


def log_frame(log, frame, frame_time_ms, gpu_mb, num_gaussians, map_size_mb):
    log.write(f"{int(frame)},{frame_time_ms:.3f},{gpu_mb:.2f},"
              f"{int(num_gaussians)},{map_size_mb:.3f}\n")
    log.flush()  # a crashed run keeps every frame logged so far


def save_poses(save_dir, poses_c2w, frames=None):
    """Save one pose per input frame as translation + quaternion and as the
    full row-major 4x4, so the native convention survives round-tripping."""
    poses_c2w = np.asarray(poses_c2w, dtype=np.float64)
    if frames is None:
        frames = np.arange(len(poses_c2w))
    rows = []
    for frame, pose in zip(frames, poses_c2w):
        qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat()
        rows.append([int(frame), *pose[:3, 3], qx, qy, qz, qw, *pose.reshape(-1)])
    header = "frame,tx,ty,tz,qx,qy,qz,qw," + ",".join(
        f"m{r}{c}" for r in range(4) for c in range(4))
    np.savetxt(os.path.join(eval_dir(save_dir), "poses.csv"), np.array(rows),
               delimiter=",", header=header, comments="",
               fmt=["%d"] + ["%.9g"] * 23)


def read_render_metrics(path):
    """Pick up mean PSNR/SSIM/LPIPS from this repo's own rendering evaluation."""
    if not os.path.isfile(path):
        return {}
    with open(path) as result_file:
        result = json.load(result_file)
    return {key: result[key] for key in ("mean_psnr", "mean_ssim", "mean_lpips")
            if key in result}


def save_run(save_dir, method, scene, status, last_frame, metrics=None, meta=None):
    """Write eval_run/run.json. Called once when the run starts and again when
    it ends, so an interrupted run still leaves a readable record."""
    info = {"method": method,
            "scene": scene,
            "status": status,
            "last_frame": int(last_frame),
            "metrics": metrics or {},
            "meta": meta or {}}
    with open(os.path.join(eval_dir(save_dir), "run.json"), "w") as run_file:
        json.dump(info, run_file, indent=4)
