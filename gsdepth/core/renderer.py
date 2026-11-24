"""CPU projection and alpha compositing for standard 3D Gaussian PLYs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Gaussians:
    centers: np.ndarray
    covariances: np.ndarray
    opacities: np.ndarray
    colors: np.ndarray


@dataclass
class Camera:
    camera_to_world: np.ndarray
    intrinsics: np.ndarray
    width: int
    height: int

    def rays(self):
        y, x = np.indices((self.height, self.width), dtype=float)
        pixels = np.stack((x + 0.5, y + 0.5, np.ones_like(x)), axis=-1)
        return pixels @ np.linalg.inv(self.intrinsics).T


PLY_TYPES = {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
    "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
    "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
    "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8",
}


def read_binary_ply_vertices(stream):
    if stream.readline().strip() != b"ply":
        raise ValueError("Expected a PLY file")
    header_format, count, element = None, None, None
    fields = []
    for line in stream:
        tokens = line.decode("ascii").split()
        match tokens:
            case ["format", kind, "1.0"]:
                header_format = kind
            case ["element", name, size]:
                if count is None and name != "vertex":
                    raise ValueError("The vertex element must be first")
                element = name
                if name == "vertex":
                    count = int(size)
            case ["property", kind, name] if element == "vertex":
                fields.append((name, "<" + PLY_TYPES[kind]))
            case ["property", *_] if element == "vertex":
                raise ValueError("PLY vertex properties must be scalar")
            case ["end_header"]:
                break
    else:
        raise ValueError("Incomplete PLY header")
    if header_format != "binary_little_endian" or not count:
        raise ValueError("Expected a binary little-endian PLY with vertices")
    dtype = np.dtype(fields)
    byte_count = dtype.itemsize * count
    payload = stream.read(byte_count)
    if len(payload) != byte_count:
        raise ValueError("Truncated PLY vertex data")
    return np.frombuffer(payload, dtype=dtype, count=count)


def load_gaussian_ply(path: Path) -> Gaussians:
    with path.open("rb") as stream:
        vertices = read_binary_ply_vertices(stream)

    def columns(*names):
        return np.column_stack([vertices[name] for name in names]).astype(np.float64)

    centers = columns("x", "y", "z")
    scales = np.exp(columns("scale_0", "scale_1", "scale_2"))
    quaternion = columns("rot_0", "rot_1", "rot_2", "rot_3")
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    w, x, y, z = quaternion.T
    rotation = np.stack((
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ), axis=1).reshape(-1, 3, 3)
    ellipsoid = rotation * scales[:, None, :]
    covariances = ellipsoid @ ellipsoid.transpose(0, 2, 1)
    logits = np.clip(vertices["opacity"].astype(np.float64), -60, 60)
    opacities = 1 / (1 + np.exp(-logits))
    colors = np.clip(0.5 + 0.28209479177387814 * columns("f_dc_0", "f_dc_1", "f_dc_2"), 0, 1)
    if not all(np.isfinite(values).all() for values in (centers, covariances, opacities, colors)):
        raise ValueError("The PLY contains invalid Gaussian parameters")
    return Gaussians(centers, covariances, opacities, colors)


def projected_fragments(gaussians: Gaussians, camera: Camera):
    """Return one fragment per contributing Gaussian-pixel pair."""
    alpha_min = 1 / 255
    rotation = camera.camera_to_world[:3, :3]
    centers = (gaussians.centers - camera.camera_to_world[:3, 3]) @ rotation
    good = (centers[:, 2] > 1e-4) & (gaussians.opacities > alpha_min)
    indices = np.flatnonzero(good)
    centers = centers[indices]
    x, y, z = centers.T
    fx, fy = camera.intrinsics[0, 0], camera.intrinsics[1, 1]
    cx, cy = camera.intrinsics[0, 2], camera.intrinsics[1, 2]
    screen = np.column_stack((fx*x/z + cx, fy*y/z + cy))
    jacobian = np.zeros((len(z), 2, 3))
    jacobian[:, 0, 0] = fx / z
    jacobian[:, 1, 1] = fy / z
    jacobian[:, 0, 2] = -fx*x/z**2
    jacobian[:, 1, 2] = -fy*y/z**2
    world_jacobian = jacobian @ rotation.T
    covariance = world_jacobian @ gaussians.covariances[indices] @ world_jacobian.transpose(0, 2, 1)
    covariance += 0.3 * np.eye(2)
    support = np.sqrt(2 * np.log(gaussians.opacities[indices] / alpha_min))
    extent = support[:, None] * np.sqrt(covariance.diagonal(axis1=1, axis2=2))
    size = np.array([camera.width, camera.height])
    left = np.ceil(np.clip(screen - extent - 0.5, 0, size)).astype(np.int32)
    right = np.floor(np.clip(screen + extent - 0.5, -1, size - 1)).astype(np.int32)
    dimensions = np.maximum(right - left + 1, 0)
    counts = dimensions[:, 0] * dimensions[:, 1]
    active = np.flatnonzero(counts)
    pixels, gaussians_seen, alpha = [], [], []
    prefix = np.r_[0, np.cumsum(counts[active], dtype=np.int64)]
    start = 0
    while start < active.size:
        limit = prefix[start] + 500_000
        end = max(start + 1, np.searchsorted(prefix, limit, side="right") - 1)
        batch = active[start:end]
        count = counts[batch]
        fragments = np.repeat(batch, count)
        offsets = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
        px = left[fragments, 0] + offsets % dimensions[fragments, 0]
        py = left[fragments, 1] + offsets // dimensions[fragments, 0]
        dx, dy = (np.column_stack((px, py)) + 0.5 - screen[fragments]).T
        cxx = covariance[fragments, 0, 0]
        cxy = covariance[fragments, 0, 1]
        cyy = covariance[fragments, 1, 1]
        quadratic = (cyy*dx*dx - 2*cxy*dx*dy + cxx*dy*dy) / (cxx*cyy - cxy*cxy)
        values = gaussians.opacities[indices[fragments]] * np.exp(-0.5*quadratic)
        keep = values >= alpha_min
        pixels.append((py*camera.width + px)[keep])
        gaussians_seen.append(fragments[keep])
        alpha.append(np.minimum(values[keep], 0.99))
        start = end
    if not pixels:
        return np.empty(0, int), np.empty(0), np.empty(0), np.empty((0, 3))
    splat = np.concatenate(gaussians_seen)
    return np.concatenate(pixels), z[splat], np.concatenate(alpha), gaussians.colors[indices[splat]]


def add_layer_depths(result, p, z, ids, starts, counts, opacity, cumulative, minimum_gap, shape):
    """Split rays at a wide gap with appreciable contribution on both sides."""
    size = opacity.size
    targets = cumulative[starts] + 0.5 * opacity[ids]
    median_hits = np.searchsorted(cumulative, targets, side="left") - 1
    median = np.full(size, np.nan)
    median[ids] = z[np.clip(median_hits, starts, starts + counts - 1)]
    from_start = cumulative[1:] - np.repeat(cumulative[starts], counts)
    mass = from_start / opacity[p]
    gaps = np.r_[np.diff(z), 0.]
    same_ray = np.r_[p[1:] == p[:-1], False]
    passes = same_ray & (gaps >= minimum_gap) & (mass >= 0.10) & (mass <= 0.90)
    widest = np.zeros(size)
    np.maximum.at(widest, p[passes], gaps[passes])
    winners = np.flatnonzero(passes & (gaps == widest[p]))
    split = np.full(size, p.size)
    np.minimum.at(split, p[winners], winners)
    eligible = split[ids] < p.size
    rays, begin, end = ids[eligible], starts[eligible], (starts + counts)[eligible]
    split = split[rays]
    front, back = np.full(size, np.nan), np.full(size, np.nan)
    boundary = cumulative[split + 1]
    for output, start, stop, lower, upper in (
        (front, begin, split, cumulative[begin], boundary),
        (back, split + 1, end - 1, boundary, cumulative[end]),
    ):
        hits = np.searchsorted(cumulative, (lower + upper) / 2, side="left") - 1
        output[rays] = z[np.clip(hits, start, stop)]
    result.update(
        median=median.reshape(shape), front=front.reshape(shape), back=back.reshape(shape),
        layer_gap=widest.reshape(shape),
    )


def composite(pixels, depths, alpha, colors, shape, minimum_layer_gap=None):
    """Return expected depth, contribution statistics and optional layer depths."""
    size = int(np.prod(shape))
    if not len(pixels):
        empty = {
            "depth": np.full(shape, np.nan), "opacity": np.zeros(shape),
            "spread": np.full(shape, np.nan), "std": np.full(shape, np.nan),
            "rgb": np.ones((*shape, 3)),
        }
        if minimum_layer_gap is not None:
            empty.update({key: np.full(shape, np.nan) for key in ("median", "front", "back")})
            empty["layer_gap"] = np.zeros(shape)
        return empty
    order = np.lexsort((depths, pixels))
    p, z, a = pixels[order], depths[order], alpha[order]
    ids, starts, counts = np.unique(p, return_index=True, return_counts=True)
    log_transmission = np.log1p(-a)
    prefix = np.r_[0, np.cumsum(log_transmission)]
    transmittance = np.exp(np.minimum(prefix[:-1] - np.repeat(prefix[starts], counts), 0))
    weights = a * transmittance
    opacity = np.bincount(p, weights=weights, minlength=size)
    depth = np.full(size, np.nan)
    depth[ids] = np.bincount(p, weights=weights*z, minlength=size)[ids] / opacity[ids]
    std = np.full(size, np.nan)
    squared_diff = weights * (z - depth[p])**2
    std[ids] = np.sqrt(np.bincount(p, weights=squared_diff, minlength=size)[ids] / opacity[ids])
    contribution_prefix = np.r_[0, np.cumsum(weights)]
    quantiles = []
    for fraction in (0.1, 0.9):
        targets = contribution_prefix[starts] + fraction * opacity[ids]
        hits = np.searchsorted(contribution_prefix, targets, side="left") - 1
        hits = np.clip(hits, starts, starts + counts - 1)
        quantiles.append(z[hits])
    spread = np.full(size, np.nan)
    spread[ids] = quantiles[1] - quantiles[0]
    rgb = np.column_stack([
        1 - opacity + np.bincount(p, weights=weights*colors[order, channel], minlength=size)
        for channel in range(3)
    ])
    result = {
        "depth": depth.reshape(shape), "opacity": opacity.reshape(shape),
        "spread": spread.reshape(shape), "std": std.reshape(shape),
        "rgb": np.clip(rgb.reshape(*shape, 3), 0, 1),
    }
    if minimum_layer_gap is not None:
        add_layer_depths(result, p, z, ids, starts, counts, opacity, contribution_prefix, minimum_layer_gap, shape)
    return result


def render(gaussians: Gaussians, camera: Camera, minimum_layer_gap=None):
    return composite(
        *projected_fragments(gaussians, camera), (camera.height, camera.width), minimum_layer_gap
    )
