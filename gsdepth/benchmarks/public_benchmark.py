"""Evaluate the published DX.GL chair splat against the rendered source geometry."""

from io import BytesIO
import json
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from PIL import Image

from gsdepth.benchmarks.experiment import save_detail, save_results, score_frame
from gsdepth.core.renderer import Camera, load_gaussian_ply, read_binary_ply_vertices, render


CALIBRATION = (29, 94, 157)
VIEWS = (14, 36, 58, 80, 102, 124, 146, 168)
BAD_THRESHOLD = 0.05


def normalized_poses(poses):
    """Replicate Nerfstudio's default up alignment, pose centering and scaling."""
    up = poses[:, :3, 1].mean(axis=0)
    up /= np.linalg.norm(up)
    target = np.array([0., 0., 1.])
    axis = np.cross(up, target)
    dot = np.dot(up, target)
    if dot <= -1 + 1e-8:
        rotation = np.diag([1, -1, -1])
    else:
        x, y, z = axis
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        rotation = np.eye(3) + cross + cross @ cross / (1 + dot)
    transformed = poses.copy()
    transformed[:, :3, :3] = rotation @ poses[:, :3, :3]
    positions = poses[:, :3, 3]
    translated = (positions - positions.mean(axis=0)) @ rotation.T
    scale = 1 / np.max(np.abs(translated))
    transformed[:, :3, 3] = translated * scale
    return transformed, scale


def read_image(archive, filename, size, resampling=Image.Resampling.NEAREST):
    with Image.open(BytesIO(archive.read(filename))) as image:
        return np.asarray(image.resize((size, size), resampling))


def camera_at(pose, intrinsics, size):
    # DX.GL supplies OpenGL matrices. The CPU renderer uses x right, y down, z forward.
    pose = pose @ np.diag([1, -1, -1, 1])
    return Camera(pose, intrinsics, size, size)


def read_truth(archive, metadata, index, size):
    view = metadata["frames"][index]
    depth16 = view["depth_file_path"].replace("depth/", "depth_16bit/", 1)
    pixel = read_image(archive, depth16, size)
    if pixel.ndim == 2:
        encoded = pixel.astype(float)
    elif pixel.ndim == 3 and pixel.shape[-1] >= 2:
        encoded = 256*pixel[..., 0].astype(float) + pixel[..., 1]
    else:
        raise ValueError(f"Unrecognized DX.GL depth encoding: {pixel.shape}")
    near, far = metadata["depth_near"], metadata["depth_far"]
    depth = near + encoded / 65535 * (far - near)
    mask = read_image(archive, view["mask_file_path"], size)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return np.where(mask > 127, depth, np.nan)


def check_depth_definition(archive, metadata, intrinsics, size):
    with archive.open(metadata["ply_file_path"]) as stream:
        vertices = read_binary_ply_vertices(stream)
    points = np.column_stack([vertices[field] for field in ("x", "y", "z")])
    checks = {}
    for index in CALIBRATION:
        pose = np.array(metadata["frames"][index]["transform_matrix"])
        camera = camera_at(pose, intrinsics, size)
        camera_points = (points - camera.camera_to_world[:3, 3]) @ camera.camera_to_world[:3, :3]
        camera_points = camera_points[camera_points[:, 2] > 0]
        image_points = camera_points @ intrinsics.T
        xy = np.floor(image_points[:, :2] / image_points[:, 2:]).astype(int)
        inside = (xy >= 0).all(axis=1) & (xy < size).all(axis=1)
        x, y = xy[inside].T
        projected = np.full((size, size), np.inf)
        np.minimum.at(projected, (y, x), camera_points[inside, 2])
        truth = read_truth(archive, metadata, index, size)
        valid = np.isfinite(projected) & np.isfinite(truth)
        if not np.any(valid):
            raise ValueError(f"Camera {index} does not overlap the source point cloud")
        z = projected[valid]
        ray_norm = np.linalg.norm(camera.rays()[valid], axis=1)
        checks[index] = {
            "overlap_pixels": int(valid.sum()),
            "median_camera_z_error": float(np.median(abs(z - truth[valid]))),
            "median_along_ray_error": float(np.median(abs(z*ray_norm - truth[valid]))),
        }
    if any(row["median_camera_z_error"] >= row["median_along_ray_error"] for row in checks.values()):
        raise ValueError(f"Camera-z depth convention failed the source point cloud check: {checks}")
    return checks


def calibrate(archive, metadata, gaussians, intrinsics, size):
    poses = np.array([item["transform_matrix"] for item in metadata["frames"]])
    adjusted, scale = normalized_poses(poses)
    calibration = {}
    candidates = (("original", poses, 1.0), ("nerfstudio_default", adjusted, scale))
    for name, pose_set, _ in candidates:
        ious = []
        for index in CALIBRATION:
            camera = camera_at(pose_set[index], intrinsics, size)
            mask = np.isfinite(read_truth(archive, metadata, index, size))
            silhouette = render(gaussians, camera)["opacity"] > 0.5
            ious.append(float(np.sum(mask & silhouette) / np.sum(mask | silhouette)))
        calibration[name] = ious
    name, pose_set, scale = max(candidates, key=lambda value: np.mean(calibration[value[0]]))
    if min(calibration[name]) < 0.65:
        raise ValueError(f"Reference alignment check failed: {calibration}")
    print(f"Silhouette calibration: {name}, IoUs {calibration}", flush=True)
    return name, pose_set, scale, calibration


def run_chair(data, output, size):
    gaussians = load_gaussian_ply(data / "scene.ply")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with ZipFile(data / "reference.zip") as archive:
        metadata = json.loads(archive.read("transforms.json"))
        sx = size / metadata["w"]
        sy = size / metadata["h"]
        intrinsics = np.array([
            [sx*metadata["fl_x"], 0, sx*metadata["cx"]],
            [0, sy*metadata["fl_y"], sy*metadata["cy"]],
            [0, 0, 1],
        ])
        depth_checks = check_depth_definition(archive, metadata, intrinsics, size)
        normalization, poses, scale, alignment = calibrate(archive, metadata, gaussians, intrinsics, size)
        for index in VIEWS:
            start = perf_counter()
            camera = camera_at(poses[index], intrinsics, size)
            frame = render(gaussians, camera)
            for key in ("depth", "spread", "std"):
                frame[key] /= scale
            truth = read_truth(archive, metadata, index, size)
            metrics, retained, error, valid, edge = score_frame(frame, truth, BAD_THRESHOLD)
            results.extend({"scene": f"chair_{index:05d}", **row} for row in metrics)
            if index == VIEWS[0]:
                # Convert camera translation as well so the point clouds retain a consistent scale.
                export_camera = camera_at(poses[index].copy(), intrinsics, size)
                export_camera.camera_to_world[:3, 3] /= scale
                save_detail(output, frame, truth, export_camera, retained, error, valid, edge, BAD_THRESHOLD)
            print(f"chair view {index:03}: {valid.sum()} pixels, {perf_counter() - start:.2f} s", flush=True)
    save_results(output, results, {
        "kind": "DX.GL Modern Arm Chair pretrained Splatfacto benchmark",
        "units": "scene units", "bad_threshold": BAD_THRESHOLD, "width": size, "height": size,
        "gaussians": len(gaussians.centers), "depth_definition": "camera z",
        "reference_views": VIEWS, "example_view": VIEWS[0],
        "calibration_views": CALIBRATION, "calibration_ious": alignment,
        "reference_point_cloud_checks": depth_checks,
        "pose_normalization": normalization, "pose_scale": scale,
        "source": "https://dx.gl/datasets/KLBJAuie9JaB",
    })
