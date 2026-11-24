"""Visibility and image evidence for fixed-model 3DGS depth hypotheses."""

from dataclasses import dataclass

import numpy as np

from gsdepth.core.renderer import Camera


@dataclass(frozen=True)
class Rules:
    minimum_layer_gap: float = 0.012
    target_opacity: float = 0.35
    neighbor_opacity: float = 0.50
    maximum_neighbor_spread: float = 0.045
    geometry_tolerance: float = 0.012
    minimum_texture: float = 0.02
    maximum_photo_cost: float = 0.25
    minimum_photo_gain: float = 0.06
    minimum_depth_change: float = 0.006
    minimum_views: int = 2


@dataclass
class ViewEvidence:
    camera: Camera
    frame: dict
    luminance: np.ndarray


def project(world, camera):
    camera_points = (world - camera.camera_to_world[:3, 3]) @ camera.camera_to_world[:3, :3]
    z = camera_points[..., 2]
    in_front = np.isfinite(z) & (z > 0)
    points = camera_points @ camera.intrinsics.T
    pixels = points[..., :2] / np.where(in_front, points[..., 2], 1)[..., None]
    return np.where(in_front[..., None], pixels, np.nan), z


def world_from_rays(camera, rays, depths):
    points = depths[..., None] * rays
    return points @ camera.camera_to_world[:3, :3].T + camera.camera_to_world[:3, 3]


def geometry_state(world, neighbor, rules):
    """Classify candidate centers as unknown, supported, occluded or contradicted."""
    pixels, z = project(world, neighbor.camera)
    x, y = pixels[..., 0], pixels[..., 1]
    size = neighbor.camera
    visible_image = (x >= 0) & (x < size.width) & (y >= 0) & (y < size.height)
    ix = np.where(visible_image, x, 0).astype(int)
    iy = np.where(visible_image, y, 0).astype(int)
    ref = neighbor.frame
    depth = ref["median"][iy, ix]
    reliable = (
        visible_image & np.isfinite(depth)
        & (ref["opacity"][iy, ix] >= rules.neighbor_opacity)
        & (ref["spread"][iy, ix] <= rules.maximum_neighbor_spread)
    )
    residual = z - depth
    state = np.zeros(z.shape, dtype=np.int8)
    state[reliable & (np.abs(residual) <= rules.geometry_tolerance)] = 1
    state[reliable & (residual > rules.geometry_tolerance)] = 2
    state[reliable & (residual < -rules.geometry_tolerance)] = -1
    return state


def sample_bilinear(image, pixels):
    """Sample an image using coordinates whose first pixel center is (0.5, 0.5)."""
    x, y = np.moveaxis(pixels - 0.5, -1, 0)
    height, width = image.shape
    inside = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    x, y = np.where(inside, x, 0), np.where(inside, y, 0)
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    ux, uy = x - ix, y - iy
    right, down = np.minimum(ix + 1, width - 1), np.minimum(iy + 1, height - 1)
    top = image[iy, ix]*(1 - ux) + image[iy, right]*ux
    bottom = image[down, ix]*(1 - ux) + image[down, right]*ux
    return np.where(inside, top*(1 - uy) + bottom*uy, np.nan)


def photo_cost(anchor_patches, other_patches, minimum_texture):
    a = anchor_patches - anchor_patches.mean(axis=-1, keepdims=True)
    b = other_patches - other_patches.mean(axis=-1, keepdims=True)
    a_norm = np.linalg.norm(a, axis=-1)
    b_norm = np.linalg.norm(b, axis=-1)
    minimum_norm = minimum_texture * np.sqrt(anchor_patches.shape[-1])
    valid = np.isfinite(b_norm) & (a_norm >= minimum_norm) & (b_norm >= minimum_norm)
    correlation = (a*b).sum(axis=-1) / np.where(valid, a_norm*b_norm, 1)
    return np.where(valid, (1 - np.clip(correlation, -1, 1))/2, np.nan)


def choose_depths(candidates, geometry, costs, rules):
    """Return the selected candidate index for each of the three evidence rules."""
    support = geometry == 1
    contradiction = (geometry == -1).any(axis=0)
    gain = costs[:, :1] - costs
    photo = np.isfinite(gain) & (costs <= rules.maximum_photo_cost) & (gain >= rules.minimum_photo_gain)
    change = np.abs(candidates - candidates[:1])
    eligible = np.isfinite(candidates) & (change >= rules.minimum_depth_change)

    def select(evidence, base_condition, use_photo_gains):
        votes = evidence.sum(axis=0)
        allowed = eligible & base_condition & (votes >= rules.minimum_views)
        gains = np.where(evidence, gain, 0).sum(axis=0) / np.maximum(votes, 1) if use_photo_gains else np.zeros_like(change)
        votes = np.where(allowed, votes, -1)
        order = np.lexsort((candidates, change, -gains, -votes), axis=0)
        winner = order[0]
        best = votes[winner, np.arange(winner.size)]
        return np.where(best >= rules.minimum_views, winner, 0)

    return {
        "geometry": select(support, ~contradiction & (support.sum(axis=0) > support.sum(axis=0)[:1]), False),
        "photo": select(photo, True, True),
        "joint": select(support & photo, ~contradiction, True),
    }


def refine(anchor, neighbors, rules):
    frame, camera = anchor.frame, anchor.camera
    depths = np.stack((frame["median"], frame["front"], frame["back"]))
    target = (
        np.isfinite(depths).all(axis=0)
        & (frame["opacity"] >= rules.target_opacity)
        & (frame["layer_gap"] >= rules.minimum_layer_gap)
    )
    # A complete anchor patch is required for a common target set in the ablations.
    target[[0, -1], :] = False
    target[:, [0, -1]] = False
    y, x = np.where(target)
    offsets = np.array([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    grid = np.column_stack((x, y))[:, None, :] + offsets
    pixels = np.concatenate((grid + 0.5, np.ones((*grid.shape[:-1], 1))), axis=-1)
    rays = pixels @ np.linalg.inv(camera.intrinsics).T
    proposals = depths[:, y, x]
    patches = world_from_rays(camera, rays[None], proposals[..., None])
    anchor_patches = anchor.luminance[grid[..., 1], grid[..., 0]]
    states, costs = [], []
    for neighbor in neighbors:
        states.append(geometry_state(patches[..., 4, :], neighbor, rules))
        projected, _ = project(patches, neighbor.camera)
        other_patches = sample_bilinear(neighbor.luminance, projected)
        costs.append(photo_cost(anchor_patches[None], other_patches, rules.minimum_texture))
    state, costs = np.stack(states), np.stack(costs)
    picks = choose_depths(proposals, state, costs, rules)
    output = {"mean": frame["depth"], "median": frame["median"]}
    choices = {}
    for name, choice in picks.items():
        depth = frame["median"].copy()
        depth[y, x] = proposals[choice, np.arange(x.size)]
        choice_image = np.zeros(target.shape, dtype=np.uint8)
        choice_image[y, x] = choice
        choices[name] = choice_image
        output[name] = depth
    trace = {
        "coordinates_yx": np.column_stack((y, x)),
        "proposals": proposals,
        "geometry_states": state,
        "photo_costs": costs,
    }
    return output, target, choices, trace
