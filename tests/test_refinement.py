"""Small independent checks for the depth correction stages."""

import unittest

import numpy as np

from gsdepth.core.depth_refinement import Rules, ViewEvidence, choose_depths, geometry_state, photo_cost, project, refine_weak_layers, sample_bilinear, world_from_rays
from gsdepth.core.depth_reprojection import refine_with_reprojection
from gsdepth.core.renderer import Camera, Gaussians, composite, projected_fragments, ray_gaussian_depth
from gsdepth.benchmarks.surface_validation import describe_surface, visible_reference


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

    def test_relaxed_layer_mass_keeps_a_weak_depth_layer(self):
        fragments = (np.array([0, 0]), np.array([2., 5.]), np.array([.94, .9]), np.ones((2, 3)), (1, 1))
        default = composite(*fragments, .12)
        relaxed = composite(*fragments, .12, minimum_layer_mass=.05)
        dual = composite(*fragments, .12, weak_layer_mass=.05)
        np.testing.assert_array_equal(relaxed["median"], default["median"])
        self.assertTrue(np.isnan(default["back"][0, 0]))
        self.assertAlmostEqual(relaxed["back"][0, 0], 5.0)
        np.testing.assert_array_equal(dual["back"], default["back"])
        np.testing.assert_array_equal(dual["weak_layers"]["back"], relaxed["back"])

    def test_backprojection_round_trips_with_rotated_camera(self):
        transform = np.array([[0., 0, 1, 3], [0, 1, 0, -1], [-1, 0, 0, .5], [0, 0, 0, 1]])
        camera = Camera(transform, self.camera.intrinsics, 2, 2)
        z = np.array([[1., 2.], [4., 8.]])
        pixels, depth = project(world_from_rays(camera, camera.rays(), z), camera)
        y, x = np.indices((2, 2))
        np.testing.assert_allclose(pixels, np.stack((x + .5, y + .5), axis=-1))
        np.testing.assert_allclose(depth, z)

    def test_ray_depth_minimizes_the_gaussian_quadratic_in_camera_z(self):
        rays = np.array([[.5, 0., 1.], [0., 0., 1.], [-3., 0., 1.]])
        centers = np.array([[0., 0., 3.], [0., 0., 3.], [1., 0., .1]])
        precision = np.diag([4., 1., 2.])
        np.testing.assert_allclose(ray_gaussian_depth(rays, centers, precision), [2., 3., .1])

    def test_ray_depth_changes_geometry_without_changing_fragment_coverage_or_alpha(self):
        camera = Camera(np.eye(4), self.camera.intrinsics, 2, 2)
        covariance = np.array([[[.4, 0., .3], [0., .6, 0.], [.3, 0., .4]]])
        model = Gaussians(np.array([[0., 0., 2.]]), covariance, np.array([.8]), np.ones((1, 3)))
        pixels, center_z, alpha, colors = projected_fragments(model, camera)
        ray_pixels, ray_z, ray_alpha, ray_colors = projected_fragments(model, camera, ray_depth=True)
        np.testing.assert_array_equal(ray_pixels, pixels)
        np.testing.assert_array_equal(ray_alpha, alpha)
        np.testing.assert_array_equal(ray_colors, colors)
        self.assertTrue(np.all(center_z == 2))
        self.assertGreater(np.ptp(ray_z), 0)

    def test_depth_percentiles_use_cumulative_contribution_and_trim_the_tails(self):
        weight = .9 * np.array([.07, .43, .43, .07])
        alpha = weight / (1 - np.r_[0., np.cumsum(weight[:-1])])
        inputs = (np.zeros(4, dtype=int), np.array([1., 2., 3., 10.]), alpha, np.ones((4, 3)), (1, 1))
        result = composite(*inputs, spread_intervals=((.05, .95), (.15, .85)))
        np.testing.assert_array_equal(result["spread"], [[1.]])
        np.testing.assert_array_equal(result["alternative_spreads"][(.05, .95)], [[9.]])
        np.testing.assert_array_equal(result["alternative_spreads"][(.15, .85)], [[1.]])

    def test_surface_visibility_requires_reference_depth_and_joint_pixel_coverage(self):
        points = np.array([[-.5, -.5, 2.], [-.60, -.60, 2.4], [.5, -.5, 2.], [10., 0., 2.], [0., 0., -1.]])
        native_depth = np.array([[2., 2.], [np.nan, np.nan]])
        valid = np.array([[True, False], [True, True]])
        np.testing.assert_array_equal(visible_reference(points, self.camera, native_depth, valid), [0])

    def test_surface_metrics_keep_accuracy_and_completeness_denominators_separate(self):
        result = describe_surface(np.array([.01, .05, .10]), np.array([.02, .11]))
        self.assertEqual(result["points"], 3)
        self.assertEqual(result["visible_reference_points"], 2)
        self.assertAlmostEqual(result["precision_0.05"], 2 / 3)
        self.assertAlmostEqual(result["recall_0.05"], .5)
        self.assertAlmostEqual(result["f1_0.05"], 4 / 7)
        self.assertEqual(result["outliers_0.05"], 1)

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

    def test_reprojected_depth_needs_two_other_views_and_keeps_existing_edits(self):
        camera = Camera(np.eye(4), self.camera.intrinsics, 5, 5)

        def view(depth):
            frame = {"median": np.full((5, 5), depth), "opacity": np.ones((5, 5)), "spread": np.zeros((5, 5))}
            return ViewEvidence(camera, frame, np.ones((5, 5)))

        anchor = view(1.0)
        geometry = anchor.frame["median"].copy()
        geometry[2, 2] = 1.1
        first, second, third = (view(1.2) for _ in range(3))
        unsupported, _ = refine_with_reprojection(anchor, [first, second, view(np.nan)], geometry, self.rules)
        np.testing.assert_array_equal(unsupported["reproject"], geometry)

        supported, _ = refine_with_reprojection(anchor, [first, second, third], geometry, self.rules)
        self.assertAlmostEqual(supported["reproject"][2, 3], 1.2)
        self.assertAlmostEqual(supported["reproject"][2, 2], 1.1)
        np.testing.assert_array_equal(supported["reproject_photo"], geometry)

        nearby, _ = refine_with_reprojection(anchor, [view(1.005), view(1.008), view(1.011)], geometry, self.rules)
        self.assertAlmostEqual(nearby["reproject_consensus"][2, 3], 1.008)
        self.assertAlmostEqual(nearby["reproject_consensus"][2, 2], 1.1)

    def test_weak_layers_require_an_extra_vote_and_leave_baseline_edits_intact(self):
        camera = Camera(np.eye(4), self.camera.intrinsics, 5, 5)
        shape = (5, 5)

        def view(depth):
            frame = {"median": np.full(shape, depth), "opacity": np.ones(shape), "spread": np.zeros(shape)}
            return ViewEvidence(camera, frame, None)

        anchor = view(1.0)
        anchor.frame["front"] = np.full(shape, np.nan)
        anchor.frame["back"] = np.full(shape, np.nan)
        anchor.frame["back"][1, 2] = 1.2
        anchor.frame["weak_layers"] = {"front": np.ones(shape), "back": np.full(shape, 1.2),
                                       "layer_gap": np.full(shape, .2)}
        original = anchor.frame["median"].copy()
        original[2, 2] = 1.1
        voters = [view(1.2) for _ in range(3)]
        two_votes = refine_weak_layers(anchor, voters[:2] + [view(np.nan), view(np.nan)], original, self.rules, 3)
        self.assertAlmostEqual(two_votes["geometry_mass05"][2, 3], 1.2)
        np.testing.assert_array_equal(two_votes["geometry_weak_guard"], original)

        three_votes = refine_weak_layers(anchor, voters + [view(np.nan)], original, self.rules, 3)["geometry_weak_guard"]
        self.assertAlmostEqual(three_votes[2, 3], 1.2)
        self.assertAlmostEqual(three_votes[2, 2], 1.1)
        self.assertAlmostEqual(three_votes[1, 2], 1.0)

        vetoed = refine_weak_layers(anchor, voters + [view(1.4)], original, self.rules, 3)
        np.testing.assert_array_equal(vetoed["geometry_weak_guard"], original)


if __name__ == "__main__":
    unittest.main()
