import math
import unittest

from src.pathing.MovementWindow import MovementWindow


class TestMovementWindow(unittest.TestCase):
    @staticmethod
    def feed(window, points, interval=1.0, start=0.0):
        for index, point in enumerate(points):
            now = start + index * interval
            window.observe(point, now)

    def test_stationary_position_becomes_stuck_only_after_full_window(self):
        movement = MovementWindow()
        for now in range(7):
            movement.observe((10, 20), now)
        self.assertFalse(movement.is_stuck(7.0))
        movement.observe((10, 20), 8.0)
        movement.observe((10, 20), 9.0)
        self.assertTrue(movement.is_stuck(9.0))
        self.assertEqual(0, movement.displacement)

    def test_small_localization_jitter_is_treated_as_stationary(self):
        movement = MovementWindow()
        jitter = [(0.2, -0.1), (-0.3, 0.2), (0.1, 0.3), (0, 0)] * 3
        self.feed(movement, jitter)
        self.assertLessEqual(movement.displacement, 3.0)
        self.assertTrue(movement.is_stuck(11.0))

    def test_slow_progress_above_threshold_is_not_stuck(self):
        movement = MovementWindow()
        self.feed(movement, [(index * 0.4, 0) for index in range(11)])
        self.assertGreater(movement.displacement, 3.0)
        self.assertFalse(movement.is_stuck(10.0))

    def test_circling_is_detected_as_movement_even_when_returning_to_start(self):
        movement = MovementWindow()
        circle = [(0, 0), (4, 0), (0, 4), (-4, 0), (0, -4), (0, 0),
                  (4, 0), (0, 4), (-4, 0), (0, -4), (0, 0)]
        self.feed(movement, circle)
        self.assertEqual((0, 0), movement._samples[-1][1])
        self.assertGreater(movement.displacement, 3.0)
        self.assertFalse(movement.is_stuck(10.0))

    def test_temporary_increase_in_target_distance_does_not_affect_movement(self):
        movement = MovementWindow()
        target = (5, 0)
        points = [(0, 0), (1, 0), (2, 0), (3, 0), (2, 0),
                  (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
        distances = [math.dist(point, target) for point in points]
        self.assertTrue(any(later > earlier for earlier, later in zip(distances, distances[1:])))
        self.feed(movement, points)
        self.assertFalse(movement.is_stuck(9.0))

    def test_irregular_intervals_keep_one_boundary_sample(self):
        movement = MovementWindow()
        times = [0.0, 1.3, 2.1, 3.4, 4.2, 5.5, 6.3, 7.6, 8.4]
        for now in times:
            movement.observe((2, 3), now)
        self.assertLessEqual(len(movement._samples), 19)
        self.assertTrue(movement.is_stuck(8.4))

    def test_fast_calls_are_downsampled_and_gap_restarts_window(self):
        movement = MovementWindow()
        for index in range(50):
            movement.observe((0, 0), index * 0.2)
        self.assertLessEqual(len(movement._samples), 19)
        self.assertTrue(movement.is_stuck(9.8))

        movement.observe((0, 0), 12.0)
        self.assertFalse(movement.is_stuck(12.0))
        self.assertEqual(1, len(movement._samples))

    def test_stale_check_and_reset_clear_old_evidence(self):
        movement = MovementWindow()
        self.feed(movement, [(0, 0)] * 10)
        self.assertFalse(movement.is_stuck(12.0))
        self.assertEqual(0, len(movement._samples))

        movement.observe((1, 1), 13.0)
        movement.reset()
        self.assertEqual(0, len(movement._samples))
        self.assertEqual(0, movement.displacement)
        self.assertFalse(movement.is_stuck(20.0))

    def test_non_finite_values_and_reversed_time_are_rejected(self):
        movement = MovementWindow()
        movement.observe((0, 0), 1)
        for point, now in (((math.nan, 0), 2), ((0, math.inf), 2), ((0, 0), math.inf)):
            with self.subTest(point=point, now=now), self.assertRaises(ValueError):
                movement.observe(point, now)
        with self.assertRaisesRegex(ValueError, "backwards"):
            movement.observe((0, 0), 0.5)
        with self.assertRaisesRegex(ValueError, "backwards"):
            movement.is_stuck(0.5)
        self.assertEqual(1, len(movement._samples))


if __name__ == '__main__':
    unittest.main()
