import unittest

from src.pathing.FallGuard import FallGuard


class TestFallGuard(unittest.TestCase):
    def setUp(self):
        self.guard = FallGuard()

    def test_short_airborne_flicker_does_not_open_glider(self):
        self.assertFalse(self.guard.observe(True, 0.0))
        self.assertTrue(self.guard.active)
        self.assertFalse(self.guard.airborne_confirmed)
        self.assertFalse(self.guard.observe(False, 0.05))
        self.assertFalse(self.guard.observe(False, 0.30))
        self.assertFalse(self.guard.active)
        self.assertFalse(self.guard.gliding)

    def test_confirmed_fall_requests_one_jump(self):
        self.assertFalse(self.guard.observe(True, 1.0))
        self.assertTrue(self.guard.observe(True, 1.12))
        self.assertTrue(self.guard.active)
        self.assertTrue(self.guard.airborne_confirmed)
        self.assertTrue(self.guard.gliding)
        for now in (1.2, 1.4, 1.8, 2.0):
            self.assertFalse(self.guard.observe(True, now))

    def test_unknown_does_not_close_glide_latch(self):
        self.guard.observe(True, 0.0)
        self.assertTrue(self.guard.observe(True, 0.12))
        self.assertFalse(self.guard.observe(None, 0.2))
        self.assertTrue(self.guard.active)
        self.assertTrue(self.guard.airborne_or_unknown)
        self.assertTrue(self.guard.gliding)
        self.assertTrue(self.guard.airborne_confirmed)
        self.assertFalse(self.guard.observe(False, 0.3))
        self.assertTrue(self.guard.active)  # One ground frame cannot land/reset it.

    def test_observation_gap_resets_pending_confirmation_only(self):
        self.assertFalse(self.guard.observe(True, 0.0))
        self.assertFalse(self.guard.observe(True, 0.61))
        self.assertTrue(self.guard.active)
        self.assertFalse(self.guard.airborne_confirmed)
        self.assertFalse(self.guard.observe(True, 0.70))
        self.assertTrue(self.guard.observe(True, 0.82))

    def test_gap_does_not_clear_existing_glide_latch(self):
        self.guard.observe(True, 0.0)
        self.assertTrue(self.guard.observe(True, 0.12))
        self.assertFalse(self.guard.observe(True, 1.0))
        self.assertTrue(self.guard.gliding)
        self.assertTrue(self.guard.active)
        self.assertFalse(self.guard.observe(True, 1.12))

    def test_stable_landing_rearms_for_a_later_fall(self):
        self.guard.observe(True, 0.0)
        self.assertTrue(self.guard.observe(True, 0.12))
        self.assertFalse(self.guard.observe(False, 0.2))
        self.assertFalse(self.guard.observe(False, 0.39))  # Not quite 0.2 s.
        self.assertTrue(self.guard.active)
        self.assertFalse(self.guard.observe(False, 0.41))
        self.assertFalse(self.guard.active)
        self.assertFalse(self.guard.gliding)
        self.assertFalse(self.guard.observe(True, 0.5))
        self.assertTrue(self.guard.observe(True, 0.62))

    def test_reset_clears_state_for_known_grounded_start(self):
        self.guard.observe(True, 0.0)
        self.assertTrue(self.guard.observe(True, 0.12))
        self.guard.reset()
        self.assertFalse(self.guard.active)
        self.assertFalse(self.guard.airborne_or_unknown)
        self.assertFalse(self.guard.gliding)
        self.assertFalse(self.guard.airborne_confirmed)
        self.assertFalse(self.guard.observe(True, 2.0))


if __name__ == '__main__':
    unittest.main()
