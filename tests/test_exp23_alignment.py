import unittest


class Exp23AlignmentTest(unittest.TestCase):
    def test_typical_crop_uses_shared_normalized_interval(self):
        from speechtokenizer.speaker_identity.cache import aligned_crop_bounds

        bounds = aligned_crop_bounds(160000, 500, 32000, 48000)

        self.assertEqual((bounds.audio_start, bounds.audio_end), (32000, 80000))
        self.assertEqual((bounds.token_start, bounds.token_end), (100, 250))

    def test_crop_clamps_at_audio_end_and_keeps_a_token(self):
        from speechtokenizer.speaker_identity.cache import aligned_crop_bounds

        bounds = aligned_crop_bounds(160000, 500, 159999, 48000)

        self.assertEqual((bounds.audio_start, bounds.audio_end), (159999, 160000))
        self.assertEqual((bounds.token_start, bounds.token_end), (499, 500))

    def test_crop_larger_than_item_clamps_to_full_item(self):
        from speechtokenizer.speaker_identity.cache import aligned_crop_bounds

        bounds = aligned_crop_bounds(100, 3, 0, 1000)

        self.assertEqual((bounds.audio_start, bounds.audio_end), (0, 100))
        self.assertEqual((bounds.token_start, bounds.token_end), (0, 3))

    def test_start_beyond_item_clamps_to_last_nonempty_interval(self):
        from speechtokenizer.speaker_identity.cache import aligned_crop_bounds

        bounds = aligned_crop_bounds(100, 3, 1000, 10)

        self.assertEqual((bounds.audio_start, bounds.audio_end), (99, 100))
        self.assertEqual((bounds.token_start, bounds.token_end), (2, 3))

    def test_invalid_integer_arguments_are_rejected(self):
        from speechtokenizer.speaker_identity.cache import aligned_crop_bounds

        invalid_calls = [
            (True, 10, 0, 1),
            (10, False, 0, 1),
            (10, 10, True, 1),
            (10, 10, 0, False),
            (0, 10, 0, 1),
            (10, 0, 0, 1),
            (10, 10, -1, 1),
            (10, 10, 0, 0),
            (10.0, 10, 0, 1),
        ]
        for args in invalid_calls:
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                aligned_crop_bounds(*args)


if __name__ == '__main__':
    unittest.main()
