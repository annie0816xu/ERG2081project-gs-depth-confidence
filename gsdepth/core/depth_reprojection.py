"""Propose anchor depths from neighboring views of a fixed Gaussian model."""

import numpy as np

from gsdepth.core.depth_refinement import geometry_state, photo_cost, project, sample_bilinear, world_from_rays


def reproject_neighbor_depth(neighbor, anchor_camera, rules):
    """Splat reliable depth samples into the anchor with a nearest-depth buffer."""
    frame = neighbor.frame
    depth = frame["median"]
    reliable = (
        np.isfinite(depth)
        & (frame["opacity"] >= rules.neighbor_opacity)
        & (frame["spread"] <= rules.maximum_neighbor_spread)
    )
    points = world_from_rays(neighbor.camera, neighbor.camera.rays()[reliable], depth[reliable])
    pixels, z = project(points, anchor_camera)
    x, y = pixels.T
    inside = (x >= 0) & (x < anchor_camera.width) & (y >= 0) & (y < anchor_camera.height)
    ix, iy = x[inside].astype(int), y[inside].astype(int)
    buffer = np.full((anchor_camera.height, anchor_camera.width), np.inf)
    np.minimum.at(buffer, (iy, ix), z[inside])
    buffer[~np.isfinite(buffer)] = np.nan
    return buffer


def refine_with_reprojection(anchor, neighbors, geometry_depth, rules):
    """Compare three checks of depths reprojected from neighboring views."""
    median = anchor.frame["median"]
    proposals = np.stack([reproject_neighbor_depth(view, anchor.camera, rules) for view in neighbors])
    target = (anchor.frame["opacity"] >= rules.target_opacity) & (geometry_depth == median)
    target &= np.isfinite(proposals).any(axis=0)
    target[[0, -1], :] = False
    target[:, [0, -1]] = False
    rays = anchor.camera.rays()[target]
    baseline = median[target]
    candidates = proposals[:, target]
    candidate_world = world_from_rays(anchor.camera, rays, candidates)
    baseline_world = world_from_rays(anchor.camera, rays, baseline)
    states = np.stack([geometry_state(candidate_world, view, rules) for view in neighbors])
    baseline_states = np.stack([geometry_state(baseline_world, view, rules) for view in neighbors])

    other_views = ~np.eye(len(neighbors), dtype=bool)
    support = (states == 1) & other_views[:, :, None]
    votes = support.sum(axis=0)
    baseline_votes = ((baseline_states == 1)[:, None, :] & other_views[:, :, None]).sum(axis=0)
    change = np.abs(candidates - baseline)
    allowed = (
        np.isfinite(candidates)
        & (change >= rules.minimum_depth_change)
        & (votes >= rules.minimum_views)
        & (votes > baseline_votes)
        & (~(states == -1).any(axis=0))
    )

    def select(accepted):
        scores = np.where(accepted, votes, -1)
        selected = np.lexsort((candidates, change, -scores), axis=0)[0]
        pixels = np.arange(baseline.size)
        depth = geometry_depth.copy()
        depth[target] = np.where(accepted[selected, pixels], candidates[selected, pixels], baseline)
        return depth

    # Reuse the original 3x3 image match. A validating view must supply both types of evidence.
    proof = np.zeros(allowed.shape, dtype=bool)
    active = allowed.any(axis=0)
    if active.any():
        y, x = np.nonzero(target)
        offsets = np.array([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        grid = np.column_stack((x[active], y[active]))[:, None, :] + offsets
        pixels = np.concatenate((grid + 0.5, np.ones((*grid.shape[:-1], 1))), axis=-1)
        patch_rays = pixels @ np.linalg.inv(anchor.camera.intrinsics).T
        patch_depths = np.vstack((baseline[active], candidates[:, active]))
        patches = world_from_rays(anchor.camera, patch_rays[None], patch_depths[..., None])
        anchor_patches = anchor.luminance[grid[..., 1], grid[..., 0]]
        for k, view in enumerate(neighbors):
            projected, _ = project(patches, view.camera)
            sampled = sample_bilinear(view.luminance, projected)
            costs = photo_cost(anchor_patches[None], sampled, rules.minimum_texture)
            gain = costs[:1] - costs[1:]
            proof[:, active] |= (
                support[k][:, active]
                & (costs[1:] <= rules.maximum_photo_cost)
                & (gain >= rules.minimum_photo_gain)
            )
    guarded = select(allowed & proof)

    # Use small consensus updates when at least three views reproject a nearby depth.
    nearby = change <= rules.geometry_tolerance
    stable = (nearby.sum(axis=0) >= rules.minimum_views + 1) & (guarded[target] == baseline)
    consensus = guarded.copy()
    if stable.any():
        depths = np.nanmedian(np.where(nearby[:, stable], candidates[:, stable], np.nan), axis=0)
        points = world_from_rays(anchor.camera, rays[stable], depths)
        checks = np.stack([geometry_state(points, view, rules) for view in neighbors])
        accepted = ((checks == 1).sum(axis=0) >= rules.minimum_views + 1) & ~(checks == -1).any(axis=0)
        coordinates = np.flatnonzero(target)[stable][accepted]
        consensus.flat[coordinates] = depths[accepted]

    proposals[:, ~target] = np.nan
    return {"reproject": select(allowed), "reproject_photo": guarded, "reproject_consensus": consensus}, proposals
