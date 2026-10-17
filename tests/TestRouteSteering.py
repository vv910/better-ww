import math
import unittest

from src.pathing.RouteSteering import DIRECTIONS, RouteSteering


class TestRouteSteering(unittest.TestCase):
    def test_all_camera_relative_directions_and_wrap(self):
        for index, keys in enumerate(DIRECTIONS):
            steering = RouteSteering(350)
            self.assertEqual(keys, steering.update((350 + index * 45) % 360, 350, 0))

    def test_hysteresis_keeps_common_forward_key_near_boundary(self):
        steering = RouteSteering(0)
        self.assertEqual(('w',), steering.update(20, 0, 0))
        self.assertEqual(('w',), steering.update(27, 0, 0.1))
        self.assertEqual(('w', 'd'), steering.update(35, 0, 0.2))
        self.assertEqual(('w', 'd'), steering.update(20, 45, 0.3))

    def test_character_turn_is_not_mistaken_for_camera_turn(self):
        steering = RouteSteering(90)
        self.assertEqual(('a',), steering.update(0, 90, 0))
        steering.update(0, 45, 0.1)  # Character is still turning towards west relative to camera.
        self.assertEqual(90, steering.camera_heading)
        steering.update(0, 0, 0.4)
        self.assertEqual(90, steering.camera_heading)

    def test_closed_loop_reaches_goal_while_camera_drifts(self):
        steering = RouteSteering(90)
        x = y = 0
        player_heading = 90
        for tick in range(100):
            target = math.degrees(math.atan2(65 - y, 100 - x)) % 360
            keys = steering.update(target, player_heading, tick * 0.2)
            camera = 90 + tick * 0.08
            player_heading = (camera + DIRECTIONS.index(keys) * 45) % 360
            x += math.cos(math.radians(player_heading)) * 6
            y += math.sin(math.radians(player_heading)) * 6
            if math.hypot(100 - x, 65 - y) <= 8:
                break
        self.assertLess(math.hypot(100 - x, 65 - y), 8)
        self.assertLess(tick, 30)


if __name__ == '__main__':
    unittest.main()
