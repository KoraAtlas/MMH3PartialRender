"""Size math and resizing for supersampling H3 video latents. No ComfyUI imports."""

import torch

DOWNSCALE_METHODS = ["area", "bicubic"]
PATCH = 2  # H3 tokens cover 2x2 latent cells


def supersample_size(height, width, scale):
    """Target latent size, snapped to whole H3 tokens and never below the input."""
    def snap(n):
        return max(n, int(round(n * scale / PATCH)) * PATCH)
    return snap(height), snap(width)


def resize_video(video, size, mode):
    """[B, C, T, H, W] -> [B, C, T, *size], resizing each frame in 2D so time is untouched."""
    b, c, t, h, w = video.shape
    frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    kwargs = {"align_corners": False, "antialias": True} if mode == "bicubic" else {}
    frames = torch.nn.functional.interpolate(frames.float(), size=tuple(size), mode=mode, **kwargs).to(video.dtype)
    return frames.reshape(b, t, c, size[0], size[1]).permute(0, 2, 1, 3, 4)


def downscale_mid_sampling(video, size, sigma, seed):
    """Shrink a latent between sampler stages, keeping its noise level correct.

    A stage hands over x0 + sigma / (1 - sigma) * noise. Area averaging shrinks that
    noise's variance by the ratio of the areas, and the next stage would read the
    shortfall as lost signal, so fresh noise tops it back up."""
    sigma = float(sigma)
    if sigma >= 0.999:
        raise ValueError("The switch happens at sigma %.3f, before any step has run. Switch at a later step." % sigma)
    small = resize_video(video, size, "area")
    if sigma <= 0.0:
        return small
    area_ratio = (size[0] * size[1]) / float(video.shape[3] * video.shape[4])
    missing = (sigma / (1.0 - sigma)) * max(0.0, 1.0 - area_ratio) ** 0.5
    generator = torch.Generator("cpu").manual_seed(int(seed))
    noise = torch.randn(tuple(small.shape), generator=generator, dtype=torch.float32).to(small.device).to(small.dtype)
    return small + noise * missing
