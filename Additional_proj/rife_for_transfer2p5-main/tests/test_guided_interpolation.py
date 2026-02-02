import unittest

from guided_interpolation import guided_non_uniform_interpolation


class DummyModel:
    version = 3.9

    def inference(self, frame_start, frame_end, t, scale=1.0):
        return {"start": frame_start, "end": frame_end, "t": t, "scale": scale}


class GuidedInterpolationTests(unittest.TestCase):
    def test_time_steps_length(self):
        model = DummyModel()
        frames = ["f0", "f1"]
        output = guided_non_uniform_interpolation(
            model=model,
            frames=frames,
            segment_frames=[5],
            time_steps=[[0.1, 0.5, 0.9]],
        )
        self.assertEqual(len(output), 5)

    def test_time_steps_strict_increasing(self):
        model = DummyModel()
        frames = ["f0", "f1"]
        with self.assertRaises(ValueError):
            guided_non_uniform_interpolation(
                model=model,
                frames=frames,
                segment_frames=[5],
                time_steps=[[0.5, 0.5, 0.9]],
            )


if __name__ == "__main__":
    unittest.main()
