from typing import List, Optional, Sequence


def _make_inference(model, frame_start, frame_end, middle_count: int, scale: float):
    if middle_count <= 0:
        return []
    if getattr(model, "version", 0) >= 3.9:
        return [
            model.inference(frame_start, frame_end, (i + 1) * 1.0 / (middle_count + 1), scale)
            for i in range(middle_count)
        ]
    middle = model.inference(frame_start, frame_end, scale)
    if middle_count == 1:
        return [middle]
    first_half = _make_inference(model, frame_start, middle, middle_count=middle_count // 2, scale=scale)
    second_half = _make_inference(model, middle, frame_end, middle_count=middle_count // 2, scale=scale)
    if middle_count % 2:
        return [*first_half, middle, *second_half]
    return [*first_half, *second_half]


def _inference_at_t(model, frame_start, frame_end, t: float, scale: float, max_depth: int = 10):
    if getattr(model, "version", 0) >= 3.9:
        return model.inference(frame_start, frame_end, t, scale)
    middle = model.inference(frame_start, frame_end, scale)
    if max_depth <= 0 or abs(t - 0.5) < 1e-8:
        return middle
    if t < 0.5:
        return _inference_at_t(model, frame_start, middle, t * 2, scale, max_depth=max_depth - 1)
    return _inference_at_t(model, middle, frame_end, (t - 0.5) * 2, scale, max_depth=max_depth - 1)


def validate_time_steps(
    time_steps: Optional[Sequence[Sequence[float]]],
    segment_frames: Sequence[int],
) -> Optional[List[List[float]]]:
    if time_steps is None:
        return None
    if len(time_steps) != len(segment_frames):
        raise ValueError("time_steps length must be len(segment_frames).")
    validated: List[List[float]] = []
    for idx, (segment_count, steps) in enumerate(zip(segment_frames, time_steps)):
        middle_count = segment_count - 2
        if len(steps) != middle_count:
            raise ValueError(
                "time_steps[{idx}] length must be {expected} for segment_frames[{idx}]={segment}.".format(
                    idx=idx, expected=middle_count, segment=segment_count
                )
            )
        last_t: Optional[float] = None
        for jdx, t in enumerate(steps):
            if not 0 < t < 1:
                raise ValueError(f"time_steps[{idx}][{jdx}] must satisfy 0 < t < 1.")
            if last_t is not None and t <= last_t:
                raise ValueError(f"time_steps[{idx}] must be strictly increasing.")
            last_t = t
        validated.append(list(steps))
    return validated


def guided_non_uniform_interpolation(
    model,
    frames: Sequence,
    frame_map: Optional[Sequence[int]] = None,
    segment_frames: Optional[Sequence[int]] = None,
    scale: float = 1.0,
    time_steps: Optional[Sequence[Sequence[float]]] = None,
) -> List:
    """Generate non-uniform interpolated frames in a single call.

    Args:
        model: RIFE model instance with `inference` method.
        frames: sampled frames ordered by time.
        frame_map: indices of sampled frames in the original timeline.
        segment_frames: output length (inclusive) between adjacent sampled frames.
        scale: scale argument forwarded to `model.inference`.
        time_steps: optional per-segment middle-frame timestamps (0 < t < 1).
    """
    if len(frames) < 2:
        raise ValueError("frames must contain at least 2 items.")
    if segment_frames is None:
        if frame_map is None:
            raise ValueError("Either frame_map or segment_frames must be provided.")
        if len(frame_map) != len(frames):
            raise ValueError("frame_map length must match frames length.")
        segment_frames = [
            frame_map[i + 1] - frame_map[i] + 1 for i in range(len(frame_map) - 1)
        ]
    if len(segment_frames) != len(frames) - 1:
        raise ValueError("segment_frames length must be len(frames) - 1.")
    for idx, segment_count in enumerate(segment_frames):
        if segment_count < 2:
            raise ValueError(f"segment_frames[{idx}] must be >= 2.")
    validated_time_steps = validate_time_steps(time_steps, segment_frames)
    output_frames: List = []
    for idx, (frame_start, frame_end, total_count) in enumerate(
        zip(frames[:-1], frames[1:], segment_frames)
    ):
        middle_count = max(total_count - 2, 0)
        if validated_time_steps is None:
            middle_frames = _make_inference(
                model, frame_start, frame_end, middle_count=middle_count, scale=scale
            )
        else:
            middle_frames = [
                _inference_at_t(model, frame_start, frame_end, t, scale)
                for t in validated_time_steps[idx]
            ]
        segment = [frame_start, *middle_frames, frame_end]
        if output_frames:
            output_frames.extend(segment[1:])
        else:
            output_frames.extend(segment)
    return output_frames
