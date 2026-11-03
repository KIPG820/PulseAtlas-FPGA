import unittest
from collections import deque

from patch_presence_checker_candidate import CalibratedPatchPresenceChecker


def checker():
    value = object.__new__(CalibratedPatchPresenceChecker)
    value.temporal_window = 3
    value.temporal_votes_required = 2
    value.temporal_max_gap_s = 2.5
    value._presence_history = {}
    return value


class TemporalConfirmationTests(unittest.TestCase):
    def test_single_positive_is_pending_and_second_of_three_confirms(self):
        c = checker()
        a = c._temporal_confirmation("RA", "pad_present", .83, {}, 1.0)
        b = c._temporal_confirmation("RA", "pad_present", .81, {}, 2.0)
        self.assertEqual(a[0], "pad_pending")
        self.assertEqual(b[0], "pad_present")
        self.assertEqual(b[3], 2)

    def test_one_frame_false_trigger_never_turns_green(self):
        c = checker()
        statuses = [c._temporal_confirmation("V1", raw, score, {}, t)[0]
                    for raw, score, t in (("pad_present", .8, 1.0),
                                          ("pad_not_found", .2, 2.0),
                                          ("pad_not_found", .1, 3.0))]
        self.assertEqual(statuses, ["pad_pending", "pad_not_found", "pad_not_found"])

    def test_confirmed_presence_survives_one_miss_then_clears(self):
        c = checker()
        states = [
            c._temporal_confirmation("LA", raw, score, {}, t)[0]
            for raw, score, t in (("pad_present", .8, 1.0),
                                  ("pad_present", .8, 2.0),
                                  ("pad_not_found", .2, 3.0),
                                  ("pad_not_found", .2, 4.0))
        ]
        self.assertEqual(states, ["pad_pending", "pad_present", "pad_present", "pad_not_found"])

    def test_color_mismatch_requires_repeated_same_group(self):
        c = checker()
        details = {"detected_color_group": "yellow", "color_confidence": .95}
        first = c._temporal_confirmation("RA", "pad_color_mismatch", .9, details, 1.0)
        second = c._temporal_confirmation("RA", "pad_color_mismatch", .9, details, 2.0)
        self.assertEqual(first[0], "pad_pending")
        self.assertEqual(second[0], "pad_color_mismatch")
        self.assertEqual(second[2]["detected_color_group"], "yellow")

    def test_stale_history_resets(self):
        c = checker()
        c._temporal_confirmation("RL", "pad_present", .9, {}, 1.0)
        next_state = c._temporal_confirmation("RL", "pad_present", .9, {}, 5.0)
        self.assertEqual(next_state[0], "pad_pending")
        self.assertEqual(next_state[3], 1)


if __name__ == "__main__":
    unittest.main()
