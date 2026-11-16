"""Checkerboard and interlaced token skipping for MiniMax H3. No ComfyUI imports in this file.

How a sparse step works
-----------------------
1. The latent is patchified into H3's 2x2 tokens and only active tokens are kept.
2. They are handed to the model as a smaller but regular token grid: T x H x W/2
   for checkerboard and vertical lines, T x H/2 x W for horizontal lines,
   with a layout whose RoPE position ids are the tokens' REAL frame coordinates.
   Patches inside the model, such as H3-Optimizations' cube ordering and sparse
   routing, see a normal H3 call, and tokens that share a grid cell neighborhood
   are still spatial neighbours. Each token keeps its true (t, h, w).
3. The prediction is converted to x0, and the skipped cells are interpolated in
   x0 space. Each skipped cell's velocity is then rebuilt from its own noisy
   latent: v = (x - x0) / sigma. Interpolating velocity directly would average
   independent noise between tiles and shrink its variance.
"""

import logging

import numpy as np
import torch

try:
    from .supersample_core import resize_video, supersample_size
except ImportError:  # run directly by the tests
    from supersample_core import resize_video, supersample_size

INTERPOLATION_METHODS = ["edge_directed", "detail_transfer", "temporal", "bilinear", "bicubic", "nearest",
                         "gaussian", "harmonic", "previous_step"]
EDGE_METHODS = ("edge_directed", "detail_transfer", "temporal")
GUIDE_METHODS = ("previous_step", "detail_transfer")
PATCH = 2  # H3 tokens cover 2x2 latent cells


# ---------------------------------------------------------------- geometry

def snap_grid_size(grid_size):
    """Grid size in latent cells, snapped down to H3's token grid (even, >= 2)."""
    return max(PATCH, int(grid_size) // PATCH * PATCH)


def _tile_axis(n, tile, shifted):
    offset = 0
    if shifted:
        if tile == 1:  # a whole-token shift only swaps parity, so double the tile instead
            tile = 2
        else:
            offset = tile // 2
    return (np.arange(n) + offset) // tile


def active_token_mask(h_tokens, w_tokens, grid_size, parity, shifted=False):
    """[Ht, Wt] bool mask. Tile (i, j) is active when (i + j) % 2 == parity.
    grid_size is one size, or (height, width), in latent cells. `shifted` moves the
    tile boundaries, so tokens split apart by one layout share a pass under the other."""
    grid_h, grid_w = grid_size if isinstance(grid_size, (tuple, list)) else (grid_size, grid_size)
    rows = _tile_axis(h_tokens, snap_grid_size(grid_h) // PATCH, shifted)
    cols = _tile_axis(w_tokens, snap_grid_size(grid_w) // PATCH, shifted)
    return (rows[:, None] + cols[None, :]) % 2 == parity


def interlace_token_mask(h_tokens, w_tokens, thickness, parity, shifted=False, direction="horizontal", render_percent=50):
    """[Ht, Wt] bool mask of whole lines, the requested share of them kept.
    Every kept line is complete, so the kept tokens always form a rectangle."""
    horizontal = direction == "horizontal"
    n = h_tokens if horizontal else w_tokens
    bands = _tile_axis(n, snap_grid_size(thickness) // PATCH, shifted)
    picked = _balance_line(bands % 2 == parity, int(np.clip(round(n * render_percent / 100.0), 0, n)))
    if horizontal:
        return np.repeat(picked[:, None], w_tokens, axis=1)
    return np.repeat(picked[None, :], h_tokens, axis=0)


def _balance_line(active, target):
    """Exactly `target` True values: drop or add evenly spaced ones as needed."""
    out = np.zeros_like(active)
    if target == 0:
        return out
    on, off = np.nonzero(active)[0], np.nonzero(~active)[0]
    if on.size >= target:
        out[_evenly(on, target)] = True
    else:
        out[on] = True
        out[_evenly(off, target - on.size)] = True
    return out


def _evenly(values, count):
    return values[np.linspace(0, values.size - 1, count).round().astype(np.int64)]


def balance_rows(token_mask, render_percent=50):
    """Give every row exactly the requested share of tokens, so they form a regular grid."""
    width = token_mask.shape[1]
    target = int(np.clip(round(width * render_percent / 100.0), 0, width))
    return np.stack([_balance_line(row, target) for row in token_mask])


def tokens_to_cells(token_mask):
    return np.repeat(np.repeat(token_mask, PATCH, axis=0), PATCH, axis=1)


def _nearest_active(active, axis):
    """Index of the nearest active cell before/after each cell along `axis`.
    `before` is -1 and `after` is the axis size where there is none."""
    n = active.shape[axis]
    shape = [1, 1]
    shape[axis] = n
    idx = np.arange(n).reshape(shape)
    before = np.maximum.accumulate(np.where(active, idx, -1), axis=axis)
    after = np.flip(np.minimum.accumulate(np.flip(np.where(active, idx, n), axis=axis), axis=axis), axis=axis)
    return before, after


# ---------------------------------------------------------------- fill plans

class FillPlan:
    """Each skipped cell is a weighted sum of K source cells. Weights are linear
    in the samples, so one plan serves every channel and frame."""

    def __init__(self, ys, xs, sample_y, sample_x, weights, neighbors=None, iterations=0, edge_directed=False,
                 directions=None):
        self.edge_directed = edge_directed
        self.directions = directions  # angled sample pairs for interlaced lines
        self._direction_tensors = None
        self.ys, self.xs = ys, xs                    # [n]
        self.sample_y, self.sample_x = sample_y, sample_x  # [K, n]
        self.weights = weights                       # [K, n]
        self.neighbors = neighbors                   # (ny, nx), each [4, n], for harmonic
        self.iterations = iterations
        self._tensors = None

    def _on(self, device, dtype):
        key = (str(device), str(dtype))
        if self._tensors is None or self._tensors[0] != key:
            def to_long(a):
                return torch.as_tensor(a.astype(np.int64)).to(device)
            n = None
            if self.neighbors is not None:
                n = (to_long(self.neighbors[0]), to_long(self.neighbors[1]))
            self._tensors = (key, to_long(self.ys), to_long(self.xs), to_long(self.sample_y),
                             to_long(self.sample_x),
                             torch.as_tensor(self.weights.astype(np.float32)).to(device).to(dtype), n)
        return self._tensors[1:]

    def apply(self, x0):
        """x0: [B, C, T, H, W]. Returns a copy with the skipped cells filled."""
        ys, xs, sy, sx, w, nbrs = self._on(x0.device, x0.dtype)
        out = x0.clone()
        if self.directions is not None:
            out[:, :, :, ys, xs] = self._directional_fill(x0)
            return out
        if self.edge_directed:
            out[:, :, :, ys, xs] = self._edge_fill(x0, sy, sx, w)
            return out
        filled = 0.0
        for k in range(sy.shape[0]):
            filled = filled + x0[:, :, :, sy[k], sx[k]] * w[k]
        out[:, :, :, ys, xs] = filled
        if nbrs is not None:
            ny, nx = nbrs
            for _ in range(self.iterations):  # Jacobi; rendered cells stay fixed
                avg = 0.0
                for k in range(ny.shape[0]):
                    avg = avg + out[:, :, :, ny[k], nx[k]]
                out[:, :, :, ys, xs] = avg / ny.shape[0]
        return out


    def _directional_fill(self, x0):
        """Edge-based line averaging, as used in deinterlacers. Lines have no rendered
        neighbors along their own axis, so each cell compares several angles across
        the gap and follows the ones whose two ends match best."""
        key = str(x0.device)
        if self._direction_tensors is None or self._direction_tensors[0] != key:
            d = self.directions
            def long_t(a):
                return torch.as_tensor(a.astype(np.int64)).to(x0.device)
            self._direction_tensors = (key, [long_t(a) for a in d["indices"]],
                                       torch.as_tensor(d["near"].astype(np.float32)).to(x0.device),
                                       torch.as_tensor(d["penalty"].astype(np.float32)).to(x0.device))
        _, (ay, ax, by, bx), near, penalty = self._direction_tensors
        near = near.to(x0.dtype)
        estimates, scores = [], []
        for angle in range(ay.shape[0]):
            score = 0.0
            for along in range(ay.shape[1]):  # the cell and its neighbors along the line, for a steadier match
                first = x0[:, :, :, ay[angle, along], ax[angle, along]]
                second = x0[:, :, :, by[angle, along], bx[angle, along]]
                score = score + (first - second).abs().mean(dim=1, keepdim=True)
                if along == 0:
                    estimates.append(first * near + second * (1.0 - near))
            scores.append(score * penalty[angle].to(x0.dtype))
        eps = float(scores[self.directions["straight"]].mean()) * 0.1 + 1e-4
        total, weight = 0.0, 0.0
        for estimate, score in zip(estimates, scores):
            w_angle = 1.0 / (score + eps) ** 4
            total = total + estimate * w_angle
            weight = weight + w_angle
        return total / weight

    @staticmethod
    def _edge_fill(x0, sy, sx, w):
        """Blends the vertical and horizontal estimates, trusting whichever axis
        changes least, so fills run along edges instead of across them."""
        s = [x0[:, :, :, sy[k], sx[k]] for k in range(4)]
        estimates, reach, gradients, paired = [], [], [], []
        for k in (0, 2):
            total = w[k] + w[k + 1]
            estimates.append((s[k] * w[k] + s[k + 1] * w[k + 1]) / total.clamp(min=1e-6))
            reach.append(total)
            pair = (w[k] > 0) & (w[k + 1] > 0)
            # an unpaired sample points at the skipped cell itself, which holds noise
            gradients.append((s[k] - s[k + 1]).abs().mean(dim=1, keepdim=True) * pair.to(x0.dtype))
            paired.append(pair)
        pairs = (paired[0].to(x0.dtype) + paired[1].to(x0.dtype)).clamp(min=1.0)
        typical = (gradients[0] + gradients[1]) / pairs
        eps = float(typical.mean()) * 0.1 + 1e-4
        weights = [reach[i] / (torch.where(paired[i], gradients[i], typical) + eps) ** 2 for i in (0, 1)]
        return (estimates[0] * weights[0] + estimates[1] * weights[1]) / (weights[0] + weights[1])


def _axis_samples(ys, xs, before, after, size, axis):
    """Distances and validity of the nearest active cell on each side."""
    b, a = before[ys, xs], after[ys, xs]
    pos = ys if axis == 0 else xs
    has_b, has_a = b >= 0, a < size
    d_b = np.where(has_b, pos - b, 0)
    d_a = np.where(has_a, a - pos, 0)
    return b, a, has_b, has_a, d_b, d_a


def _axis_confidence(d_b, d_a, has_b, has_a):
    """An axis counts for more when an active cell on it is close."""
    closest = np.where(has_b & has_a, np.minimum(d_b, d_a), np.where(has_b, d_b, d_a))
    return np.where(has_b | has_a, 1.0 / np.maximum(closest, 1), 0.0)


def _linear_plan(active, ys, xs):
    H, W = active.shape
    sy, sx, w = [], [], []
    conf = []
    axis_weights = []
    for axis, size in ((0, H), (1, W)):
        before, after = _nearest_active(active, axis)
        b, a, has_b, has_a, d_b, d_a = _axis_samples(ys, xs, before, after, size, axis)
        gap = np.maximum(d_b + d_a, 1)
        both = has_b & has_a
        w_b = np.where(both, d_a / gap, has_b.astype(np.float64))
        w_a = np.where(both, d_b / gap, has_a.astype(np.float64))
        for idx, valid, weight in ((b, has_b, w_b), (a, has_a, w_a)):
            on_axis = np.where(valid, idx, ys if axis == 0 else xs)
            sy.append(on_axis if axis == 0 else ys)
            sx.append(xs if axis == 0 else on_axis)
            axis_weights.append((axis, weight))
        conf.append(_axis_confidence(d_b, d_a, has_b, has_a))
    total = conf[0] + conf[1]
    if np.any(total == 0):
        raise ValueError("partial render: a skipped cell has no active cell on its row or column")
    for axis, weight in axis_weights:
        w.append(weight * conf[axis] / total)
    return np.stack(sy), np.stack(sx), np.stack(w)


def _nearest_plan(active, ys, xs):
    sy, sx, w = _linear_plan(active, ys, xs)
    H, W = active.shape
    dist = np.abs(sy - ys[None]) + np.abs(sx - xs[None])
    dist = np.where(w > 0, dist, H + W + 1)
    best = dist.argmin(axis=0)
    n = np.arange(ys.shape[0])
    return sy[best, n][None], sx[best, n][None], np.ones((1, ys.shape[0]))


def _bicubic_plan(active, ys, xs):
    """Cubic Hermite across each gap, with end slopes read from the rendered
    tiles. A side with no partner holds the edge value rather than extrapolating."""
    H, W = active.shape
    n = ys.shape[0]
    sy, sx, w, conf, owners = [], [], [], [], []
    for axis, size in ((0, H), (1, W)):
        before, after = _nearest_active(active, axis)
        b, a, has_b, has_a, d_b, d_a = _axis_samples(ys, xs, before, after, size, axis)
        pos = ys if axis == 0 else xs
        p0 = np.where(has_b, b, pos)
        p1 = np.where(has_a, a, pos)

        def is_active(line):
            inside = (line >= 0) & (line < size)
            clamped = np.clip(line, 0, size - 1)
            cell = active[clamped, xs] if axis == 0 else active[ys, clamped]
            return inside & cell

        pm = np.where(has_b & is_active(p0 - 1), p0 - 1, p0)
        pp = np.where(has_a & is_active(p1 + 1), p1 + 1, p1)

        both = has_b & has_a
        gap = np.maximum(d_b + d_a, 1).astype(np.float64)
        s = np.where(both, d_b / gap, 0.0)
        h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
        h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
        k0 = np.where(pm != p0, h10 * gap, 0.0)
        k1 = np.where(pp != p1, h11 * gap, 0.0)
        weights = [np.where(both, -k0, 0.0),
                   np.where(both, h00 + k0, has_b.astype(np.float64)),
                   np.where(both, h01 - k1, has_a.astype(np.float64)),
                   np.where(both, k1, 0.0)]
        for line, weight in zip((pm, p0, p1, pp), weights):
            sy.append(line if axis == 0 else ys)
            sx.append(xs if axis == 0 else line)
            w.append(weight)
            owners.append(axis)
        conf.append(_axis_confidence(d_b, d_a, has_b, has_a))
    total = conf[0] + conf[1]
    if np.any(total == 0):
        raise ValueError("partial render: a skipped cell has no active cell on its row or column")
    w = [weight * conf[axis] / total for weight, axis in zip(w, owners)]
    return np.stack(sy), np.stack(sx), np.stack(w).reshape(8, n)


def _gaussian_plan(active, ys, xs, grid_size, top_k=16):
    H, W = active.shape
    r = max(grid_size, 2)
    sigma = max(grid_size / 2.0, 1.0)
    dy, dx = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
    dy, dx = dy.reshape(-1, 1), dx.reshape(-1, 1)
    cand_y = ys[None] + dy
    cand_x = xs[None] + dx
    inside = (cand_y >= 0) & (cand_y < H) & (cand_x >= 0) & (cand_x < W)
    usable = inside & active[np.clip(cand_y, 0, H - 1), np.clip(cand_x, 0, W - 1)]
    g = np.exp(-(dy**2 + dx**2) / (2 * sigma * sigma)) * usable
    k = min(top_k, g.shape[0])
    order = np.argsort(-g, axis=0)[:k]
    n = np.arange(ys.shape[0])
    w = g[order, n]
    total = w.sum(axis=0, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("partial render: gaussian window found no active cells")
    return np.clip(cand_y[order, n], 0, H - 1), np.clip(cand_x[order, n], 0, W - 1), w / total


def _direction_samples(active, ys, xs, line_axis, spreads=(0, -2, 2, -4, 4)):
    """Sample pairs across the gap at several angles. `line_axis` is the axis lines
    run along: 1 for horizontal lines, 0 for vertical ones."""
    across = 1 - line_axis
    size_across, size_along = active.shape[across], active.shape[line_axis]
    before, after = _nearest_active(active, across)
    pos_across = ys if across == 0 else xs
    pos_along = xs if across == 0 else ys
    b, a = before[ys, xs], after[ys, xs]
    has_b, has_a = b >= 0, a < size_across
    if np.any(~has_b & ~has_a):
        raise ValueError("partial render: a skipped line has no rendered line near it")
    first_line = np.where(has_b, b, a)      # an edge line uses its one neighbor for both ends
    second_line = np.where(has_a, a, b)
    d_first = np.maximum(np.abs(pos_across - first_line), 1)
    d_second = np.maximum(np.abs(second_line - pos_across), 1)
    near = np.where(has_b & has_a, d_second / (d_first + d_second), 1.0)
    gap = (d_first + d_second).astype(np.float64)
    pairs = [[], [], [], []]
    for spread in spreads:
        shift_first = np.where(has_b & has_a, np.round(spread * d_first / gap), 0).astype(np.int64)
        shift_second = np.where(has_b & has_a, np.round(spread * d_second / gap), 0).astype(np.int64)
        rows = [[], [], [], []]
        for along in (0, -1, 1):
            f = np.clip(pos_along - shift_first + along, 0, size_along - 1)
            g = np.clip(pos_along + shift_second + along, 0, size_along - 1)
            coords = ((first_line, f), (second_line, g)) if across == 0 else ((f, first_line), (g, second_line))
            rows[0].append(coords[0][0]); rows[1].append(coords[0][1])
            rows[2].append(coords[1][0]); rows[3].append(coords[1][1])
        for k in range(4):
            pairs[k].append(np.stack(rows[k]))
    return {"indices": [np.stack(p) for p in pairs], "near": near,
            "penalty": np.array([1.0 + 0.05 * abs(sp) for sp in spreads]), "straight": 0}


def build_fill_plan(active, method, grid_size, line_axis=None):
    """active: [H, W] bool over latent cells."""
    if method not in INTERPOLATION_METHODS:
        raise ValueError(f"unknown interpolation method: {method}")
    ys, xs = np.nonzero(~active)
    neighbors, iterations = None, 0
    if method == "nearest":
        parts = _nearest_plan(active, ys, xs)
    elif method == "bicubic":
        parts = _bicubic_plan(active, ys, xs)
    elif method == "gaussian":
        parts = _gaussian_plan(active, ys, xs, grid_size)
    else:  # bilinear, the edge-aware modes, harmonic's starting point, previous_step's fallback
        parts = _linear_plan(active, ys, xs)
    if method == "harmonic":
        H, W = active.shape
        neighbors = (np.stack([np.clip(ys - 1, 0, H - 1), np.clip(ys + 1, 0, H - 1), ys, ys]),
                     np.stack([xs, xs, np.clip(xs - 1, 0, W - 1), np.clip(xs + 1, 0, W - 1)]))
        iterations = max(4, grid_size * 2)
    directions = None
    if line_axis is not None and method in EDGE_METHODS:
        directions = _direction_samples(active, ys, xs, line_axis)
    return FillPlan(ys, xs, *parts, neighbors=neighbors, iterations=iterations, edge_directed=method in EDGE_METHODS,
                    directions=directions)


# ---------------------------------------------------------------- tokens

def patchify(latent):
    """[1, C, T, H, W] -> [T*Ht*Wt, C*4], same row order as H3's patchify_video."""
    b, c, t, h, w = latent.shape
    x = latent.reshape(b, c, t, 1, h // PATCH, PATCH, w // PATCH, PATCH)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(b * t * (h // PATCH) * (w // PATCH), c * PATCH * PATCH)


def unpatchify(rows, t, h_tokens, w_tokens, c):
    """Inverse of patchify; returns [1, C, T, H, W]."""
    x = rows.reshape(1, t, h_tokens, w_tokens, c, 1, PATCH, PATCH)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(1, c, t, h_tokens * PATCH, w_tokens * PATCH)


class SparseLayout:
    """PackedLayout with the target video rows cut down to the active tokens.
    Text, audio, keyframe and reference rows are untouched."""

    def __init__(self, full, active_rows, text_len, t, h_cells, w_cells, audio_t):
        va, vb = next((a, b) for a, b, kind in full.segments if kind == "video")
        if vb != full.seq_len:
            raise ValueError("partial render: expected the video rows to be the last segment")
        n = int(active_rows.shape[0])
        idx = active_rows.to(full.position_ids.device)
        self.position_ids = torch.cat([full.position_ids[:va], full.position_ids[va:vb][idx]])
        keep = full.img_pos < va
        new_rows = torch.arange(va, va + n).to(full.img_pos.device).to(full.img_pos.dtype)
        self.img_pos = torch.cat([full.img_pos[keep], new_rows])
        self.img_update = torch.cat([full.img_update[keep],
                                     torch.ones(n, dtype=full.img_update.dtype).to(full.img_update.device)])
        self.audio_pos = full.audio_pos
        self.audio_update = full.audio_update
        self.segments = [(a, b, kind) for a, b, kind in full.segments if kind != "video"] + [(va, va + n, "video")]
        self.seq_len = va + n
        # the compact grid as the model will see it, which is what it validates against
        self.signature = (text_len, t, h_cells, w_cells, audio_t)


def expand_mask(mask, h, w):
    """Replicate-pad a [B, 1, T, H, W] mask out to the patch-padded size."""
    if mask.shape[-2] == h and mask.shape[-1] == w:
        return mask
    iy = torch.arange(h).clamp(max=mask.shape[-2] - 1).to(mask.device)
    ix = torch.arange(w).clamp(max=mask.shape[-1] - 1).to(mask.device)
    return mask[:, :, :, iy][:, :, :, :, ix]


# ---------------------------------------------------------------- wrapper

class _SmallCache(dict):
    """Keeps only the most recent entries. Context windows hand over a new payload
    on every call, so an unbounded cache would grow for the whole run."""

    def __init__(self, size=8):
        super().__init__()
        self.size = size

    def __setitem__(self, key, value):
        self.pop(key, None)
        super().__setitem__(key, value)
        while len(self) > self.size:
            self.pop(next(iter(self)))


class PartialRenderWrapper:
    """DIFFUSION_MODEL wrapper. `pad_fn` and `layout_factory` come from ComfyUI."""

    MIN_DENOM = 1e-6

    def __init__(self, grid_size, pattern, interpolation, alternate, start_step, full_steps_at_end,
                 pad_fn, layout_factory, shift_grid=False, render_percent=50, flip_per_frame=False, supersample=1.0, interlace=None):
        self.interlace = interlace  # None, "horizontal" or "vertical"
        self.supersample = float(supersample)
        self._latest_x0 = {}   # whole x0 at the user's size, per cond/uncond and window
        self._lifted = {}      # (step, sigma, window) -> enlarged latent, shared by cond and uncond
        self._big_payloads = _SmallCache()
        grid = grid_size if isinstance(grid_size, (tuple, list)) else (grid_size, grid_size)
        self.grid_size = (snap_grid_size(grid[0]), snap_grid_size(grid[1]))
        self.base_parity = 1 if pattern == "odd" else 0
        self.interpolation = interpolation
        self.alternate = alternate
        self.start_step = start_step
        self.full_steps_at_end = full_steps_at_end
        self.shift_grid = shift_grid
        self.render_percent = render_percent
        # temporal fill reads each skipped cell from neighboring frames, which needs a flipped pattern
        self.flip_per_frame = flip_per_frame or interpolation == "temporal"
        self.pad_fn = pad_fn
        self.layout_factory = layout_factory
        self._plans = {}
        self._layouts = _SmallCache()
        self._x0_cache = {}  # cond/uncond and window -> (x0, rendered cells per frame group; None = all)
        self._logged = set()
        self._last_step = None

    def parity_for_step(self, step, total_steps):
        """Parity for this sampler step, or None when it should render in full."""
        if step < self.start_step or step >= total_steps - self.full_steps_at_end:
            return None
        if self.alternate:
            return (self.base_parity + step - self.start_step) % 2
        return self.base_parity

    def shifted_for_step(self, step):
        return self.shift_grid and (step - self.start_step) % 2 == 1

    @staticmethod
    def _step_info(sigma, transformer_options):
        schedule = transformer_options.get("sample_sigmas")
        if schedule is None:
            return None, None
        schedule = schedule.to(sigma.device)
        return int((schedule - sigma).abs().argmin()), int(schedule.shape[0]) - 1

    def _plan(self, h, w, parity, shifted):
        """Token masks and fill plans. There are two when the pattern flips per frame:
        frame f uses entry f % len(masks)."""
        key = (h, w, parity, shifted)
        if key not in self._plans:
            parities = (parity, 1 - parity) if self.flip_per_frame else (parity,)
            ht, wt = h // PATCH, w // PATCH
            if self.interlace:
                thickness = self.grid_size[0] if self.interlace == "horizontal" else self.grid_size[1]
                fill_size, line_axis = thickness, (1 if self.interlace == "horizontal" else 0)
                raw = [interlace_token_mask(ht, wt, thickness, p, shifted, self.interlace, 50) for p in parities]
                masks = [interlace_token_mask(ht, wt, thickness, p, shifted, self.interlace, self.render_percent)
                         for p in parities]
            else:
                fill_size, line_axis = max(self.grid_size), None
                raw = [active_token_mask(ht, wt, self.grid_size, p, shifted) for p in parities]
                masks = [balance_rows(m, self.render_percent) for m in raw]
            plans = None
            # a tile covering the whole frame leaves no pattern to skip
            if all(m.any() and not m.all() for m in raw) and masks[0].any() and not masks[0].all():
                try:
                    plans = [build_fill_plan(tokens_to_cells(m), self.interpolation, fill_size, line_axis) for m in masks]
                except ValueError:  # tile as large as the frame: nothing to fill from
                    plans = None
            self._plans[key] = (masks, plans)
        return self._plans[key]

    def _layout(self, payload, text_len, t, h, w, audio_t, parity, shifted, masks, device):
        full = payload.get("layout")
        if full is None or full.signature != (text_len, t, h, w, audio_t):
            full = self.layout_factory(text_len, t, h, w, audio_t,
                                       keyframes=payload.get("keyframes"), refs=payload.get("refs"))
        key = (id(full), parity, shifted)
        cached = self._layouts.get(key)
        if cached is None or cached[0] is not full:
            per_frame = masks[0].size
            rows = torch.as_tensor(np.concatenate([
                np.nonzero(masks[f % len(masks)].reshape(-1))[0].astype(np.int64) + f * per_frame
                for f in range(t)]))
            # the kept tokens form a rectangle: rows that have any, each with the same count
            compact_h = int(masks[0].any(axis=1).sum())
            compact_w = int(masks[0].sum(axis=1).max())
            cached = (full, rows, SparseLayout(full, rows, text_len, t, compact_h * PATCH, compact_w * PATCH, audio_t),
                      compact_h, compact_w)
            self._layouts[key] = cached
        return cached[1].to(device), cached[2], cached[3], cached[4]

    @staticmethod
    def _groups(plans, t, device):
        return [(i, torch.as_tensor(np.arange(i, t, len(plans))).to(device), plan) for i, plan in enumerate(plans)
                if i < t]

    def _spatial(self, x0, plans):
        out = x0.clone()
        for _, frames, plan in self._groups(plans, x0.shape[2], x0.device):
            out[:, :, frames] = plan.apply(x0[:, :, frames])
        return out

    def _temporal(self, filled, plans):
        """Motion-adaptive: use neighboring frames where they agree with the spatial
        fill, and keep the spatial fill where they don't, because something moved."""
        t = filled.shape[2]
        if t < 2 or len(plans) < 2:
            return filled
        prev_f = torch.as_tensor(np.where(np.arange(t) == 0, 1, np.arange(t) - 1)).to(filled.device)
        next_f = torch.as_tensor(np.where(np.arange(t) == t - 1, t - 2, np.arange(t) + 1)).to(filled.device)
        across_time = (filled[:, :, prev_f] + filled[:, :, next_f]) * 0.5
        difference = (across_time - filled).abs().mean(dim=1, keepdim=True)
        scale = float(difference.mean()) * 2.0 + 1e-4
        trust = (1.0 - difference / scale).clamp(0.0, 1.0)
        blended = across_time * trust + filled * (1.0 - trust)
        out = filled.clone()
        for _, frames, plan in self._groups(plans, t, filled.device):
            ys, xs = plan._on(filled.device, filled.dtype)[:2]
            part = out[:, :, frames]
            part[:, :, :, ys, xs] = blended[:, :, frames][:, :, :, ys, xs]
            out[:, :, frames] = part
        return out

    def _remember(self, key, x0, rendered):
        if self.interpolation in GUIDE_METHODS:
            self._x0_cache[key] = (x0.detach(), rendered)

    def _fill(self, key, x0, masks, plans):
        filled = self._spatial(x0, plans)
        cached = self._x0_cache.get(key)
        usable = cached is not None and cached[0].shape == x0.shape
        rendered = [tokens_to_cells(m) for m in masks]
        if self.interpolation == "temporal":
            filled = self._temporal(filled, plans)
        elif self.interpolation == "detail_transfer" and usable:
            # interpolation loses high frequencies: take them back from the latest whole x0
            guide = cached[0].to(x0.device).to(x0.dtype)
            filled = filled + guide - self._spatial(guide, plans)
        elif self.interpolation == "previous_step" and usable:
            prior = cached[0].to(x0.device).to(x0.dtype)
            for i, frames, plan in self._groups(plans, x0.shape[2], x0.device):
                was_rendered = np.ones(plan.ys.shape, dtype=bool) if cached[1] is None else cached[1][i][plan.ys, plan.xs]
                ys = torch.as_tensor(plan.ys[was_rendered].astype(np.int64)).to(x0.device)
                xs = torch.as_tensor(plan.xs[was_rendered].astype(np.int64)).to(x0.device)
                part = filled[:, :, frames]
                part[:, :, :, ys, xs] = prior[:, :, frames][:, :, :, ys, xs]
                filled[:, :, frames] = part
                if cached[1] is None:
                    rendered[i] = np.ones_like(rendered[i])
                else:
                    rendered[i] = rendered[i] | cached[1][i]
        self._remember(key, filled, rendered)
        return filled

    def __call__(self, executor, x, timestep, context, transformer_options,
                 minimax_payload=None, denoise_mask=None, audio_denoise_mask=None, **kwargs):
        video, audio = x[0], x[1]
        sigmas = transformer_options.get("sigmas")
        if sigmas is None:  # ComfyUI always sets this; the fallback assumes the default multiplier of 1000
            sigmas = timestep / 1000.0
        sigma = sigmas.flatten()[0].float().clamp(min=1e-6)
        step, total = self._step_info(sigma, transformer_options)
        if step is not None:
            if self._last_step is not None and step < self._last_step:  # a new sampling run
                self._x0_cache.clear()
                self._layouts.clear()
                self._latest_x0.clear()
                self._lifted = {}
                self._big_payloads.clear()
            self._last_step = step
        parity = None if step is None else self.parity_for_step(step, total)
        # cached x0 must never cross between cond/uncond or between context windows of the same size
        window = transformer_options.get("context_window")
        window_id = tuple(int(i) for i in window.index_list) if getattr(window, "index_list", None) is not None else None
        key = tuple(transformer_options.get("cond_or_uncond") or (0,)) + (window_id,)

        def run_full():
            out = executor(x, timestep, context, transformer_options, minimax_payload=minimax_payload,
                           denoise_mask=denoise_mask, audio_denoise_mask=audio_denoise_mask, **kwargs)
            if self.supersample > 1.0 and denoise_mask is None:
                self._latest_x0[key] = video.float() - sigma * out[0].float()
            if self.interpolation in GUIDE_METHODS:
                v = out[0].float()
                m = 1.0 if denoise_mask is None else denoise_mask.float()
                x0 = video.float() - sigma * m * v
                if x0.shape[-2] % PATCH == 0 and x0.shape[-1] % PATCH == 0:
                    self._remember(key, x0, None)
            return out

        if parity is None:
            return run_full()
        if video.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        if kwargs.get("control") is not None:
            raise ValueError("Partial rendering does not support ControlNet. Disconnect it or turn the toggle off.")

        orig_h, orig_w = video.shape[3], video.shape[4]
        shifted = self.shifted_for_step(step)
        payload = minimax_payload or {}
        common = (executor, audio, timestep, context, transformer_options, audio_denoise_mask, kwargs,
                  parity, shifted, sigma, step)

        if self.supersample > 1.0:
            big = supersample_size(orig_h, orig_w, self.supersample)
            guide = self._latest_x0.get(key)
            if guide is None:  # uncond in a separate call can borrow cond's image
                guide = self._latest_x0.get((0, window_id))
            if big != (orig_h, orig_w) and denoise_mask is None and guide is not None and guide.shape == video.shape:
                out = self._render_supersampled(video, guide, big, payload, key, window_id, common)
                if out is not None:
                    return out
            elif "ss_skip" not in self._logged:
                self._logged.add("ss_skip")
                logging.warning("[H3 Partial Render] Supersampling needs start_step 1 or more and no noise mask, and scale "
                                "large enough to change the size. Rendering this step at normal size.")

        padded = self.pad_fn(video, (1, PATCH, PATCH))
        rendered = self._render_sparse(padded, payload, denoise_mask, key, common)
        if rendered is None:
            return run_full()
        v, audio_out, _ = rendered
        return [v[:, :, :, :orig_h, :orig_w].to(video.dtype), audio_out]

    def _lift(self, video, guide, big, sigma, seed, step, window_id):
        """Enlarge a noisy latent without changing its noise level. The image part is
        upscaled from x0. Noise is split apart: enlarging alone would leave it too weak
        and too smooth, so the new fine scales get fresh noise. The result shrinks back
        to the original noise, rescaled for the larger number of cells."""
        cache_key = (step, float(sigma), window_id)
        hit = self._lifted.get(cache_key)
        if hit is not None:
            return hit
        small = (video.shape[3], video.shape[4])
        v = video.float()
        sigma_f = float(sigma)
        noise = (v - (1.0 - sigma_f) * guide) / sigma_f
        shape = tuple(v.shape[:3]) + tuple(big)
        generator = torch.Generator("cpu").manual_seed(int(seed) + 7919 * (step + 1))
        white = torch.randn(shape, generator=generator, dtype=torch.float32).to(v.device).to(v.dtype)
        fine = white - resize_video(resize_video(white, small, "area"), big, "nearest")
        ratio = (small[0] * small[1] / float(big[0] * big[1])) ** 0.5
        lifted = (1.0 - sigma_f) * resize_video(guide, big, "bicubic") + sigma_f * (
            fine + resize_video(noise, big, "nearest") * ratio)
        self._lifted = {cache_key: lifted}
        return lifted

    def _big_payload(self, payload, big, text_len, t, audio_t):
        """Keyframes share the target's grid, so they get resized. References keep their own."""
        key = (id(payload), big, text_len, t, audio_t)
        hit = self._big_payloads.get(key)
        if hit is not None and hit[0] is payload:
            return hit[1]
        out = dict(payload)
        keyframes = []
        for kf in payload.get("keyframes") or ():
            kf = dict(kf)
            if kf.get("latent") is not None:
                kf["latent"] = resize_video(kf["latent"], big, "bicubic")
            keyframes.append(kf)
        if payload.get("keyframes") is not None:
            out["keyframes"] = keyframes
        out["cond_video_latents"] = ([kf["latent"] for kf in keyframes if kf.get("latent") is not None]
                                     + [r["latent"] for r in (payload.get("refs") or ()) if "latent" in r])
        out["layout"] = self.layout_factory(text_len, t, big[0], big[1], audio_t,
                                            keyframes=out.get("keyframes"), refs=payload.get("refs"))
        self._big_payloads[key] = (payload, out)
        return out

    def _render_supersampled(self, video, guide, big, payload, key, window_id, common):
        executor, audio, timestep, context, transformer_options, audio_denoise_mask, kwargs, parity, shifted, sigma, step = common
        lifted = self._lift(video, guide, big, sigma, payload.get("seed", 0), step, window_id)
        big_payload = self._big_payload(payload, big, context.shape[1], video.shape[2], audio.shape[-1])
        big_key = key + ("big",)
        if self.interpolation == "detail_transfer" and big_key not in self._x0_cache:
            self._x0_cache[big_key] = (resize_video(guide, big, "bicubic"), None)
        rendered = self._render_sparse(lifted.to(video.dtype), big_payload, None, big_key, common)
        if rendered is None:
            return None
        _, audio_out, x0_big = rendered
        x0 = resize_video(x0_big, (video.shape[3], video.shape[4]), "area")
        self._latest_x0[key] = x0
        v = (video.float() - x0) / sigma
        if "ss" not in self._logged:
            self._logged.add("ss")
            logging.info("[H3 Partial Render] supersampling sparse steps at %dx%d latent cells, returning to %dx%d.",
                         big[0], big[1], video.shape[3], video.shape[4])
        return [v.to(video.dtype), audio_out]

    def _render_sparse(self, padded, payload, denoise_mask, key, common):
        """One sparse render at this latent's size. Returns (velocity, audio, filled x0),
        or None when the pattern leaves nothing to skip."""
        executor, audio, timestep, context, transformer_options, audio_denoise_mask, kwargs, parity, shifted, sigma, step = common
        c, t, h, w = padded.shape[1], padded.shape[2], padded.shape[3], padded.shape[4]

        masks, plans = self._plan(h, w, parity, shifted)
        if plans is None:
            if ("no_tiles", parity) not in self._logged:
                self._logged.add(("no_tiles", parity))
                logging.warning("[H3 Partial Render] grid %s at %s%% gives nothing to skip on a %dx%d latent; rendering in full.",
                                self.grid_size, self.render_percent, h, w)
            return None

        rows_all = patchify(padded.float())
        active_rows, layout, compact_h, compact_w = self._layout(payload, context.shape[1], t, h, w, audio.shape[-1], parity, shifted, masks, padded.device)
        n_active = int(active_rows.shape[0])
        compact = unpatchify(rows_all[active_rows], t, compact_h, compact_w, c).to(padded.dtype)

        full_mask = None
        compact_mask = None
        if denoise_mask is not None:
            full_mask = expand_mask(denoise_mask.float(), h, w)
            compact_mask = unpatchify(patchify(full_mask)[active_rows], t, compact_h, compact_w, 1).to(denoise_mask.dtype)

        sparse_payload = dict(payload)
        sparse_payload["layout"] = layout
        out = executor([compact, audio], timestep, context, transformer_options, minimax_payload=sparse_payload,
                       denoise_mask=compact_mask, audio_denoise_mask=audio_denoise_mask, **kwargs)

        # put the rendered tokens back at their true locations
        v_rows = torch.zeros_like(rows_all)
        v_rows[active_rows] = patchify(out[0].float())
        v = unpatchify(v_rows, t, h // PATCH, w // PATCH, c)

        # fill in x0 space; each skipped cell keeps its own noise via its own x
        m = 1.0 if full_mask is None else full_mask
        x_full = padded.float()
        denom = sigma * m
        x0 = x_full - denom * v
        x0 = self._fill(key, x0, masks, plans)
        denom_t = torch.as_tensor(denom).to(x_full.device)
        rebuilt = torch.where(denom_t > self.MIN_DENOM,
                              (x_full - x0) / denom_t.clamp(min=self.MIN_DENOM),
                              torch.zeros_like(x_full))
        for _, frames, plan in self._groups(plans, t, x_full.device):
            ys, xs = plan._on(x_full.device, x_full.dtype)[:2]
            part = v[:, :, frames]
            part[:, :, :, ys, xs] = rebuilt[:, :, frames][:, :, :, ys, xs]
            v[:, :, frames] = part

        total_tokens = rows_all.shape[0]
        if "stats" not in self._logged:
            self._logged.add("stats")
            logging.info("[H3 Partial Render] sending %d of %d video tokens to the model (%d other tokens unchanged).",
                         n_active, total_tokens, layout.seq_len - n_active)
        return v, out[1], x0
