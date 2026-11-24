"""Small independent checks for the depth correction stages."""

import unittest

import numpy as np

from gsdepth.core.depth_refinement import Rules, ViewEvidence, choose_depths, geometry_state, photo_cost, project, sample_bilinear, world_from_rays
from gsdepth.core.renderer import Camera, composite


class RefinementChecks(unittest.TestCase):
    def setUp(self):
        self.rules = Rules()
        self.camera = Camera(np.eye(4), np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]]), 2, 2)

    def test_layers_require_mass_on_both_sides_and_preserve_expected_depth(self):
        fragments = (
            np.array([0, 1, 0, 1]), np.array([5., 1., 2., 8.]),
            np.array([.8, .98, .3, .4]), np.ones((4, 3)), (1, 2),
        )
        original = composite(*fragments)
        frame = composite(*fragments, self.rules.minimum_layer_gap)
        for key in original:
            np.testing.assert_array_equal(original[key], frame[key])
        np.testing.assert_allclose(frame["median"], [[5, 1]])
        np.testing.assert_allclose(frame["front"], [[2, np.nan]], equal_nan=True)
        np.testing.assert_allclose(frame["back"], [[5, np.nan]], equal_nan=True)

    def test_layered_renderer_matches_independent_scalar_calculation(self):
        rng = np.random.default_rng(42)
        pixels = rng.integers(0, 12, 130)
        depth = rng.uniform(.2, 3, len(pixels))
        alpha = rng.uniform(.02, .6, len(pixels))
        frame = composite(pixels, depth, alpha, np.ones((len(pixels), 3)), (3, 4), .12)

        def weighted_median(values, weights):
            return values[np.searchsorted(weights.cumsum(), weights.sum()/2)]

        for pixel in range(12):
            hits = np.flatnonzero(pixels == pixel)
            hits = hits[np.argsort(depth[hits])]
            a, z = alpha[hits], depth[hits]
            weight = a * np.r_[1., np.cumprod(1 - a[:-1])]
            self.assertAlmostEqual(frame["median"].flat[pixel], weighted_median(z, weight))
            split_candidates = [
                i for i in range(1, len(z))
                if z[i] - z[i-1] >= .12 and min(weight[:i].sum(), weight[i:].sum()) >= .1*weight.sum()
            ]
            if split_candidates:
                split = max(split_candidates, key=lambda i: z[i] - z[i-1])
                self.assertAlmostEqual(frame["front"].flat[pixel], weighted_median(z[:split], weight[:split]))
                self.assertAlmostEqual(frame["back"].flat[pixel], weighted_median(z[split:], weight[split:]))
            else:
                self.assertTrue(np.isnan(frame["front"].flat[pixel]))
                self.assertTrue(np.isnan(frame["back"].flat[pixel]))

    def test_backprojection_round_trips_with_rotated_camera(self):
        transform = np.array([[0., 0, 1, 3], [0, 1, 0, -1], [-1, 0, 0, .5], [0, 0, 0, 1]])
        camera = Camera(transform, self.camera.intrinsics, 2, 2)
        z = np.array([[1., 2.], [4., 8.]])
        pixels, depth = project(world_from_rays(camera, camera.rays(), z), camera)
        y, x = np.indices((2, 2))
        np.testing.assert_allclose(pixels, np.stack((x + .5, y + .5), axis=-1))
        np.testing.assert_allclose(depth, z)

    def test_geometry_distinguishes_occlusion_and_free_space(self):
        frame = {"median": np.ones((2, 2)), "opacity": np.ones((2, 2)), "spread": np.zeros((2, 2))}
        neighbor = ViewEvidence(self.camera, frame, np.zeros((2, 2)))
        points = np.array([[0, 0, .95], [0, 0, 1.005], [0, 0, 1.05], [1, 0, 1], [0, 0, -1]])
        np.testing.assert_array_equal(geometry_state(points, neighbor, self.rules), [-1, 1, 2, 0, 0])
        frame["spread"][:] = 0.10
        np.testing.assert_array_equal(geometry_state(points, neighbor, self.rules), 0)

    def test_bilinear_coordinates_and_photometric_texture_threshold(self):
        image = np.array([[0., 1.], [2., 3.]])
        locations = np.array([[.5, .5], [1., 1.], [1.5, 1.5], [0., 1.]])
        np.testing.assert_allclose(sample_bilinear(image, locations), [0, 1.5, 3, np.nan], equal_nan=True)
        patch = np.array([.1, .3, .4, .5, .8])
        matches = np.stack((patch + .10, 1 - patch, np.ones_like(patch)))
        np.testing.assert_allclose(photo_cost(patch, matches, .02), [0, 1, np.nan], atol=1e-12, equal_nan=True)

    def test_joint_requires_two_votes_from_the_same_views_and_no_contradiction(self):
        candidates = np.array([[1.1, 1.1, 1.1], [1.0, 1.0, 1.0], [1.2, 1.2, 1.2]])
        geometry = np.zeros((4, 3, 3), dtype=int)
        geometry[:2, 1, :] = 1
        geometry[3, 1, 2] = -1
        costs = np.full(geometry.shape, .4)
        costs[2:, 1, 0] = .15  # Geometric and appearance evidence are from disjoint views.
        costs[:2, 1, 1:] = .15
        choices = choose_depths(candidates, geometry, costs, self.rules)
        np.testing.assert_array_equal(choices["geometry"], [1, 1, 0])
        np.testing.assert_array_equal(choices["photo"], [1, 1, 1])
        np.testing.assert_array_equal(choices["joint"], [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
