"""python tests/test_core.py  -- uses real torch if installed, otherwise the numpy shim."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
try:
    import torch; BACKEND = "torch " + torch.__version__
except ImportError:
    import _torch_shim, torch; BACKEND = "numpy shim"
import numpy as np
import partial_render_core as cc
import supersample_core as ss

def tensor(a): return torch.as_tensor(np.asarray(a, dtype=np.float32))
def npy(t): return np.asarray(t.detach().cpu().numpy() if hasattr(t, "cpu") else t)

class FakeLayout:
    """Same field layout as PackedLayout: text, then audio, then video last."""
    def __init__(self, text_len, t, h, w, audio_t, keyframes=None, refs=None):
        ht, wt = h // 2, w // 2
        na, nv = audio_t * 2, t * ht * wt
        tt, yy, xx = np.meshgrid(np.arange(t), np.arange(ht), np.arange(wt), indexing="ij")
        video = np.stack([tt.ravel(), yy.ravel(), xx.ravel()], 1).astype(np.float64)
        head = np.zeros((text_len + na, 3))
        self.position_ids = tensor(np.concatenate([head, video]))
        va = text_len + na
        self.seq_len = va + nv
        self.img_pos = torch.as_tensor(np.arange(va, va + nv))
        self.img_update = torch.as_tensor(np.ones(nv, dtype=bool))
        self.audio_pos = torch.as_tensor(np.arange(text_len, va))
        self.audio_update = torch.as_tensor(np.ones(na, dtype=bool))
        self.segments = [(0, text_len, "text"), (text_len, va, "audio"), (va, va + nv, "video")]
        self.signature = (text_len, t, h, w, audio_t)

def pad(x, patch):
    h, w = x.shape[3], x.shape[4]
    iy = np.arange(h + h % 2) % h; ix = np.arange(w + w % 2) % w
    return x[:, :, :, torch.as_tensor(iy)][:, :, :, :, torch.as_tensor(ix)]

class FakeExecutor:
    """Checks what _forward checks, and predicts v = (x - x0_true) / sigma for each token it is given."""
    def __init__(self, full_x0, sigma):
        self.full_x0, self.sigma, self.calls = full_x0, sigma, []
    def __call__(self, x, timestep, context, to, minimax_payload=None, denoise_mask=None, audio_denoise_mask=None, **kw):
        v, a = x
        layout = minimax_payload.get("layout")
        sig = (context.shape[1], v.shape[2], v.shape[3], v.shape[4], a.shape[-1])
        full_call = layout is None or layout.signature != sig
        self.calls.append(dict(strip=tuple(v.shape), layout=layout, mask=denoise_mask, full=full_call))
        if full_call:
            return [(v - self.full_x0) / self.sigma, a * 0]
        va, vb = next((p, q) for p, q, k in layout.segments if k == "video")
        assert vb - va == v.shape[2] * (v.shape[3] // 2) * (v.shape[4] // 2) and layout.seq_len == vb
        assert int(np.asarray(layout.img_update).sum()) == vb - va
        assert layout.position_ids.shape[0] == layout.seq_len
        rows = cc.patchify(v)
        coords = npy(layout.position_ids[va:vb]).astype(int)
        x0 = npy(self.full_x0)
        out = np.zeros_like(npy(rows))
        for i, (t, y, xx) in enumerate(coords):
            tok = x0[0, :, t, y * 2:y * 2 + 2, xx * 2:xx * 2 + 2]
            out[i] = (npy(rows)[i] - np.transpose(tok, (0, 1, 2)).reshape(-1)) / self.sigma
        return [cc.unpatchify(tensor(out), v.shape[2], v.shape[3] // 2, v.shape[4] // 2, v.shape[1]), a * 0]

def make(method="bilinear", grid=4, pattern="even", alternate=True, start=1, end=1, shift=False):
    return cc.PartialRenderWrapper(grid, pattern, method, alternate, start, end, pad, FakeLayout, shift_grid=shift)

def scene(h=24, w=32, t=3, c=4, sigma=0.5, seed=0, ramp=True):
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x0 = np.zeros((1, c, t, h, w), np.float32)
    for ci in range(c):
        for ti in range(t):
            x0[0, ci, ti] = 0.1 * yy + 0.05 * xx + ci + ti if ramp else np.sin(yy / 5 + ci) + np.cos(xx / 7 + ti)
    eps = np.random.default_rng(seed).standard_normal(x0.shape).astype(np.float32)
    return tensor(x0), tensor((1 - sigma) * x0 + sigma * eps), tensor(eps)

def run(w, x, x0, sigma=0.5, step=1, total=8, mask=None):
    ex = FakeExecutor(x0, sigma)
    audio = tensor(np.ones((1, 32, 2, 10)))
    ctx = tensor(np.ones((1, 7, 8)))
    to = {"sample_sigmas": tensor(np.linspace(1, 0, total + 1)), "cond_or_uncond": [0]}
    to["sample_sigmas"] = np.asarray(to["sample_sigmas"]).copy().view(type(to["sample_sigmas"])) if BACKEND == "numpy shim" else to["sample_sigmas"]
    to["sample_sigmas"][step] = sigma
    ts = tensor([sigma * 1000.0])
    out = w(ex, [x, audio], ts, ctx, to, minimax_payload={}, denoise_mask=mask)
    return out, ex

# ---------- tests
def test_tiles_partition_and_alternate():
    for g in (2, 4, 6, 8):
        a, b = (cc.active_token_mask(20, 30, g, p) for p in (0, 1))
        assert not (a & b).any() and (a | b).all()
        assert abs(a.mean() - 0.5) < 0.1
    assert cc.snap_grid_size(5) == 4 and cc.snap_grid_size(1) == 2
    w = make(start=2, end=1)
    assert [w.parity_for_step(i, 8) for i in range(8)] == [None, None, 0, 1, 0, 1, 0, None]
    w = make(pattern="odd", alternate=False, start=0, end=0)
    assert [w.parity_for_step(i, 4) for i in range(4)] == [1, 1, 1, 1]

def test_patchify_roundtrip_and_h3_row_order():
    a = tensor(np.arange(4 * 3 * 8 * 6).reshape(1, 4, 3, 8, 6))
    rows = cc.patchify(a)
    assert rows.shape == (3 * 4 * 3, 16)
    assert np.array_equal(npy(cc.unpatchify(rows, 3, 4, 3, 4)), npy(a))
    # reference: comfy.ldm.minimax.model.patchify_video
    x = np.asarray(npy(a)).reshape(1, 4, 3, 1, 4, 2, 3, 2)
    ref = np.einsum("nctrhpwq->nthwcrpq", x).reshape(1 * 3 * 4 * 3, 16)
    assert np.array_equal(npy(rows), ref)

def test_only_active_tokens_reach_model_at_true_positions():
    x0, x, _ = scene()
    out, ex = run(make(), x, x0)
    call = ex.calls[0]
    assert not call["full"]
    tokens = cc.balance_rows(cc.active_token_mask(12, 16, 4, 0))
    n = int(tokens.sum()) * 3
    assert call["strip"] == (1, 4, 3, 24, 2 * int(tokens[0].sum())) and n < 12 * 16 * 3 * 0.6
    va = call["layout"].segments[-1][0]
    ids = npy(call["layout"].position_ids[va:]).astype(int)
    assert all(tokens[y, xx] for _, y, xx in ids)
    assert len({tuple(i) for i in ids}) == n

def test_rendered_velocity_untouched_and_noise_kept_in_filled_cells():
    sigma = 0.5
    x0, x, eps = scene(sigma=sigma)
    out, _ = run(make(), x, x0, sigma=sigma)
    v = npy(out[0]); true_v = (npy(x) - npy(x0)) / sigma
    cells = cc.tokens_to_cells(cc.active_token_mask(12, 16, 4, 0))
    assert np.allclose(v[..., cells], true_v[..., cells], atol=1e-4)
    interior = ~cells; interior[:4] = interior[-4:] = False; interior[:, :4] = interior[:, -4:] = False
    # x0 is a ramp, so linear fill is exact and v must equal the cell's own (x - x0) / sigma
    assert np.allclose(v[..., interior], true_v[..., interior], atol=1e-3)
    # filled velocity carries each cell's own noise, at full variance
    assert abs(v[..., ~cells].std() / true_v[..., ~cells].std() - 1) < 0.02

def test_every_method_is_sane():
    x0, x, _ = scene(ramp=False)
    results = {}
    for m in cc.INTERPOLATION_METHODS:
        for g in (2, 4, 8):
            active = cc.tokens_to_cells(cc.active_token_mask(12, 16, g, 0))
            plan = cc.build_fill_plan(active, m, g)
            assert np.allclose(plan.weights.sum(0), 1, atol=1e-6), (m, g)
            assert active[plan.sample_y, plan.sample_x][plan.weights != 0].all(), (m, g)
            filled = npy(plan.apply(x0))
            assert np.isfinite(filled).all()
            assert np.array_equal(filled[..., active], npy(x0)[..., active])
            if g == 4: results[m] = float(np.abs(filled - npy(x0))[..., ~active].mean())
    print("   fill error on a smooth signal, grid 4:", {k: round(v, 4) for k, v in results.items()})
    assert results["bilinear"] < results["nearest"]

def test_previous_step_uses_last_real_render():
    sigma = 0.5
    x0, x, _ = scene(sigma=sigma, ramp=False)
    w = make("previous_step", start=0, end=0)
    run(w, x, x0, sigma=sigma, step=0)            # parity 0
    out, _ = run(w, x, x0, sigma=sigma, step=1)   # parity 1: parity 0 cells come from step 0
    v, true_v = npy(out[0]), (npy(x) - npy(x0)) / sigma
    assert np.allclose(v, true_v, atol=1e-3)      # same x0 both steps, so reuse is exact everywhere

def test_full_steps_pass_through():
    x0, x, _ = scene()
    for step, sigma in ((0, 1.0), (7, 0.125)):   # sigmas taken from the test schedule
        out, ex = run(make(), x, x0, sigma=sigma, step=step)
        assert ex.calls[0]["strip"] == tuple(x.shape)

def test_denoise_mask_follows_tokens():
    x0, x, _ = scene()
    m = np.ones((1, 1, 3, 24, 32), np.float32); m[:, :, 0] = 0
    out, ex = run(make(), x, x0, mask=tensor(m))
    n = ex.calls[0]["strip"][4]
    assert ex.calls[0]["mask"].shape == (1, 1, 3, 24, n)
    assert np.isfinite(npy(out[0])).all()
    cells = ~cc.tokens_to_cells(cc.active_token_mask(12, 16, 4, 0))
    assert np.all(npy(out[0])[:, :, 0][..., cells] == 0)   # no velocity from a zero denominator

def test_odd_latent_size_and_oversized_grid():
    x0, x, _ = scene(h=23, w=31)
    x0p = pad(x0, None)
    out, ex = run(make(), x, x0p)
    assert out[0].shape == x.shape
    x0, x, _ = scene(h=8, w=8)
    out, ex = run(make(grid=64), x, x0)   # one tile: nothing to interpolate from
    assert ex.calls[0]["strip"] == tuple(x.shape)

def test_shift_moves_boundaries_and_stays_a_partition():
    for g in (2, 4, 8):
        a, b = (cc.active_token_mask(20, 30, g, p, True) for p in (0, 1))
        assert not (a & b).any() and (a | b).all()
        assert not np.array_equal(a, cc.active_token_mask(20, 30, g, 0))
        assert not np.array_equal(a, cc.active_token_mask(20, 30, g, 1))
    w = make(start=2, shift=True)
    assert [w.shifted_for_step(i) for i in (2, 3, 4, 5)] == [False, True, False, True]

def test_shift_in_one_pattern_mode():
    x0, x, _ = scene(ramp=False)
    out, ex = run(make(shift=True), x, x0, step=2)
    assert np.isfinite(npy(out[0])).all() and len(ex.calls) == 1

def test_compact_grid_is_regular_and_keeps_neighbours_together():
    # uneven tile counts per row must still produce a rectangle for cube ordering
    for h, w, g in ((14, 15, 2), (14, 15, 4), (13, 17, 6), (20, 30, 8)):
        for parity in (0, 1):
            for shifted in (False, True):
                tokens = cc.balance_rows(cc.active_token_mask(h, w, g, parity, shifted))
                counts = tokens.sum(axis=1)
                assert counts.min() == counts.max(), (h, w, g)
                assert abs(tokens.mean() - 0.5) < 0.13
                cols = np.stack([np.nonzero(r)[0] for r in tokens])
                # a compact-grid column stays near the same real column in every row
                assert (cols.max(axis=0) - cols.min(axis=0)).max() <= 2 * (g // 2) + 2

def test_cube_reorder_inside_the_model_round_trips():
    """Mimics H3-Optimizations: reorder rows and positions by the grid it is given, restore after."""
    x0, x, _ = scene(ramp=False)
    sigma = 0.5
    inner = FakeExecutor(x0, sigma)
    def reordering(xs, ts, ctx, to, minimax_payload=None, denoise_mask=None, audio_denoise_mask=None):
        v, a = xs
        rows = cc.patchify(v)
        order = np.argsort(np.arange(rows.shape[0]) % 8, kind="stable")
        inverse = np.argsort(order)
        layout = minimax_payload["layout"]
        assert tuple(layout.signature) == (ctx.shape[1], v.shape[2], v.shape[3], v.shape[4], a.shape[-1])
        out = inner(xs, ts, ctx, to, minimax_payload=minimax_payload)
        shuffled = cc.patchify(out[0])[torch.as_tensor(order)]
        restored = shuffled[torch.as_tensor(inverse)]
        return [cc.unpatchify(restored, v.shape[2], v.shape[3] // 2, v.shape[4] // 2, v.shape[1]), out[1]]
    w = make()
    audio = tensor(np.ones((1, 32, 2, 10)))
    to = {"sample_sigmas": tensor([1.0, 0.75, 0.5, 0.25, 0.0]), "cond_or_uncond": [0]}
    out = w(reordering, [x, audio], tensor([sigma * 1000.0]), tensor(np.ones((1, 7, 8))), to, minimax_payload={})
    cells = cc.tokens_to_cells(cc.balance_rows(cc.active_token_mask(12, 16, 4, w.parity_for_step(2, 4))))
    assert np.allclose(npy(out[0])[..., cells], ((npy(x) - npy(x0)) / sigma)[..., cells], atol=1e-3)

def test_previous_step_cache_resets_between_runs():
    x0, x, _ = scene(ramp=False)
    w = make("previous_step", start=0, end=0)
    run(w, x, x0, step=1)
    assert w._x0_cache
    stale = w._x0_cache[(0, None)][0]
    run(w, x, x0, step=0)
    assert w._last_step == 0 and w._x0_cache[(0, None)][0] is not stale

def test_render_percent_and_rectangular_tiles_stay_regular():
    for pct in (25, 33, 50, 67, 75, 90):
        for grid in (2, (2, 4), (4, 2), (6, 2)):
            m = cc.balance_rows(cc.active_token_mask(14, 21, grid, 0), pct)
            assert m.sum(axis=1).min() == m.sum(axis=1).max()
            assert abs(m.mean() - pct / 100) < 0.04
    a = cc.active_token_mask(12, 16, (2, 4), 0)
    assert a[0, 0] and a[0, 1] and not a[0, 2] and not a[1, 0]    # 1 token tall, 2 tokens wide

def _edge_scene():
    yy, xx = np.meshgrid(np.arange(24), np.arange(32), indexing="ij")
    img = (xx >= 15).astype(np.float32) * 2.0                     # a hard vertical edge, like a letter stroke
    return tensor(np.broadcast_to(img, (1, 4, 3, 24, 32)).copy())

def test_edge_directed_does_not_blur_across_an_edge():
    x0 = _edge_scene()
    cells = cc.tokens_to_cells(cc.active_token_mask(12, 16, 2, 0))
    err = {m: float(np.abs(npy(cc.build_fill_plan(cells, m, 2).apply(x0)) - npy(x0)).mean()) for m in ("bilinear", "edge_directed")}
    print("   hard edge fill error:", {k: round(v, 4) for k, v in err.items()})
    assert err["edge_directed"] < err["bilinear"] * 0.5

def test_detail_transfer_restores_detail_from_a_full_step():
    sigma = 0.5
    x0, x, _ = scene(sigma=sigma, ramp=False)
    w = make("detail_transfer", grid=2, start=1, end=1)
    run(w, x, x0, sigma=1.0, step=0)
    assert w._x0_cache[(0, None)][1] is None                           # guide came from the full step
    w._x0_cache[(0, None)] = (x0, None)                                # a guide equal to the truth must fill exactly
    out, _ = run(w, x, x0, sigma=sigma, step=1)
    assert np.allclose(npy(out[0]), (npy(x) - npy(x0)) / sigma, atol=1e-2)

def test_flip_per_frame_and_temporal_fill():
    x0, x, _ = scene(sigma=0.5, ramp=False)
    for f in range(1, 3):
        x0[:, :, f] = x0[:, :, 0]                                 # a still shot
    w = make("temporal", grid=2)
    assert w.flip_per_frame
    out, ex = run(w, x, x0)
    layout = ex.calls[0]["layout"]
    ids = npy(layout.position_ids[layout.segments[-1][0]:]).astype(int)
    a = {tuple(i[1:]) for i in ids if i[0] == 0}; b = {tuple(i[1:]) for i in ids if i[0] == 1}
    assert not (a & b) and len(a | b) == 12 * 16                  # adjacent frames render opposite tokens
    got = npy(x) - 0.5 * npy(out[0])
    out2, _ = run(make("edge_directed", grid=2), x, x0)
    spatial = npy(x) - 0.5 * npy(out2[0])
    err_t, err_s = np.abs(got - npy(x0)).mean(), np.abs(spatial - npy(x0)).mean()
    print("   still shot fill error:", {"temporal": round(float(err_t), 4), "edge_directed": round(float(err_s), 4)})
    assert np.isfinite(got).all() and err_t < err_s

def test_new_modes_work_with_masks_and_every_percent():
    x0, x, _ = scene(ramp=False)
    m = np.ones((1, 1, 3, 24, 32), np.float32); m[:, :, 0] = 0
    for method in cc.INTERPOLATION_METHODS:
        for pct in (25, 67):
            w = cc.PartialRenderWrapper((2, 4), "even", method, False, 1, 1, pad, FakeLayout, render_percent=pct, flip_per_frame=True)
            out, ex = run(w, x, x0, mask=tensor(m))
            assert np.isfinite(npy(out[0])).all() and out[0].shape == x.shape

def test_supersample_size_and_frames_stay_in_order():
    assert ss.supersample_size(28, 28, 1.4) == (40, 40)
    assert ss.supersample_size(30, 54, 1.2) == (36, 64)
    for h, w, s in ((28, 28, 1.05), (30, 54, 2.0), (28, 40, 1.4)):
        y, x = ss.supersample_size(h, w, s)
        assert y % 2 == 0 and x % 2 == 0 and y >= h and x >= w
    v = np.zeros((1, 3, 4, 8, 8), np.float32)
    for c in range(3):
        for t in range(4):
            v[0, c, t] = 10 * c + t
    up = npy(ss.resize_video(tensor(v), (12, 16), "area"))
    assert up.shape == (1, 3, 4, 12, 16)
    for c in range(3):
        for t in range(4):
            assert np.allclose(up[0, c, t], 10 * c + t)     # no channel or frame got mixed

def test_stopping_mid_sampling_keeps_the_noise_level():
    sigma = 0.6
    level = sigma / (1 - sigma)                         # what a stage hands over: x0 + level * noise
    rng = np.random.default_rng(1)
    x0 = np.full((1, 4, 2, 120, 120), 0.5, np.float32)
    handed = tensor(x0 + level * rng.standard_normal(x0.shape).astype(np.float32))
    plain = npy(ss.resize_video(handed, (60, 60), "area"))
    fixed = npy(ss.downscale_mid_sampling(handed, (60, 60), sigma, 7))
    assert fixed.shape == (1, 4, 2, 60, 60)
    assert abs(plain.std() / level - 0.5) < 0.02        # averaging alone halves it
    assert abs(fixed.std() / level - 1.0) < 0.02        # the top-up restores it
    assert abs(fixed.mean() - 0.5) < 0.05               # the image content is unchanged
    assert np.array_equal(fixed, npy(ss.downscale_mid_sampling(handed, (60, 60), sigma, 7)))
    try:
        ss.downscale_mid_sampling(handed, (60, 60), 1.0, 7); assert False
    except ValueError:
        pass

def _super(scale=2.0, start=1, **kw):
    return cc.PartialRenderWrapper(2, "even", kw.pop("method", "edge_directed"), False, start, 1, pad, FakeLayout, supersample=scale, **kw)

def _big_scene(h=24, w=32, big=(48, 64), sigma=0.5):
    def field(rows, cols):
        yy, xx = np.meshgrid((np.arange(rows) + .5) / rows, (np.arange(cols) + .5) / cols, indexing="ij")
        return np.sin(yy * 6) + np.cos(xx * 5)
    x0 = tensor(np.broadcast_to(field(h, w), (1, 4, 3, h, w)).copy())
    x0b = tensor(np.broadcast_to(field(*big), (1, 4, 3) + big).copy())
    eps = np.random.default_rng(3).standard_normal(x0.shape).astype(np.float32)
    return x0, x0b, tensor((1 - sigma) * npy(x0) + sigma * eps)

def test_supersample_only_on_sparse_steps_and_returns_to_size():
    x0, x0b, x = _big_scene()
    w = _super()
    _, ex = run(w, x, x0, sigma=1.0, step=0)
    assert ex.calls[0]["strip"] == tuple(x.shape)                  # full step: normal size
    out, ex = run(w, x, x0b, sigma=0.5, step=1)
    assert ex.calls[0]["strip"] == (1, 4, 3, 48, 32)               # sparse step: larger grid, half the tokens
    assert out[0].shape == x.shape
    got = npy(x) - 0.5 * npy(out[0])                               # returns the picture at the user's size
    assert np.abs(got - npy(x0)).mean() < 0.03
    _, ex = run(w, x, x0, sigma=0.125, step=7)
    assert ex.calls[0]["strip"] == tuple(x.shape)

def test_supersample_keeps_noise_level_and_shared_latent():
    sigma = 0.6
    x0, _, _ = _big_scene(h=40, w=40, big=(80, 80))
    guide = x0 * 0
    eps = tensor(np.random.default_rng(5).standard_normal(x0.shape).astype(np.float32))
    video = sigma * eps                                            # image is zero, so all that's left is noise
    w = _super()
    lifted = npy(w._lift(video, guide, (80, 80), tensor(sigma), 11, 2, None))
    assert lifted.shape == (1, 4, 3, 80, 80)
    assert abs(lifted.std() / sigma - 1) < 0.02                    # plain enlarging would make the fine scales too smooth
    down = lifted.reshape(1, 4, 3, 40, 2, 40, 2).mean(axis=(4, 6))
    assert np.allclose(down, npy(video) * 0.5, atol=1e-4)          # shrinks back to the original noise, scaled for 4x the cells
    assert w._lift(video, guide, (80, 80), tensor(sigma), 11, 2, None) is w._lift(video, guide, (80, 80), tensor(sigma), 11, 2, None)

def test_supersample_falls_back_and_resizes_keyframes():
    x0, x0b, x = _big_scene()
    out, ex = run(_super(start=0), x, x0, sigma=1.0, step=0)       # no finished x0 yet
    assert ex.calls[0]["strip"][3] == 24
    w = _super()
    run(w, x, x0, sigma=1.0, step=0)
    m = np.ones((1, 1, 3, 24, 32), np.float32); m[:, :, 0] = 0
    _, ex = run(w, x, x0, sigma=0.5, step=1, mask=tensor(m))
    assert ex.calls[0]["strip"][3] == 24                           # a noise mask stays at normal size
    kf = {"resolved_frame_index": 0, "latent": tensor(np.ones((1, 4, 1, 24, 32)))}
    ref = {"kind": "image", "latent": tensor(np.ones((1, 4, 1, 10, 10)))}
    p = w._big_payload({"keyframes": [kf], "refs": [ref]}, (48, 64), 7, 3, 10)
    assert p["keyframes"][0]["latent"].shape[-2:] == (48, 64) and kf["latent"].shape[-2:] == (24, 32)
    assert p["cond_video_latents"][1].shape[-2:] == (10, 10)       # references keep their own size
    assert p["layout"].signature == (7, 3, 48, 64, 10)

def test_supersample_with_every_fill_mode():
    x0, x0b, x = _big_scene()
    for method in cc.INTERPOLATION_METHODS:
        w = _super(method=method)
        run(w, x, x0, sigma=1.0, step=0)
        for step, sigma in ((1, 0.875), (2, 0.75)):
            out, ex = run(w, x, x0b, sigma=sigma, step=step)
            assert ex.calls[0]["strip"][3] == 48 and np.isfinite(npy(out[0])).all() and out[0].shape == x.shape

def _lines(direction="horizontal", method="edge_directed", start=1, end=1, grid=2, pct=50, flip=False, ss=1.0, alternate=False, shift=False):
    return cc.PartialRenderWrapper(grid, "even", method, alternate, start, end, pad, FakeLayout, render_percent=pct,
                                  flip_per_frame=flip, supersample=ss, interlace=direction, shift_grid=shift)

def test_interlace_masks_are_whole_lines_and_any_percent():
    for direction in ("horizontal", "vertical"):
        for thick in (2, 4, 6):
            for pct in (25, 33, 50, 67, 90):
                for shifted in (False, True):
                    m = cc.interlace_token_mask(13, 19, thick, 0, shifted, direction, pct)
                    rows, cols = m.any(axis=1).sum(), m.any(axis=0).sum()
                    assert m.sum() == rows * cols                   # always a rectangle
                    n = 13 if direction == "horizontal" else 19
                    assert (rows if direction == "horizontal" else cols) == round(n * pct / 100)
    a = cc.interlace_token_mask(8, 8, 2, 0)
    b = cc.interlace_token_mask(8, 8, 2, 1)
    assert a[0].all() and not a[1].any() and a[2].all() and not (a & b).any() and (a | b).all()
    t = cc.interlace_token_mask(8, 8, 4, 0)
    assert t[0].all() and t[1].all() and not t[2].any()             # thickness is respected

def test_interlace_sends_half_the_lines_at_true_positions():
    x0, x, _ = scene(ramp=False)
    for direction, shape in (("horizontal", (1, 4, 3, 12, 32)), ("vertical", (1, 4, 3, 24, 16))):
        out, ex = run(_lines(direction), x, x0)
        call = ex.calls[0]
        assert call["strip"] == shape and out[0].shape == x.shape
        ids = npy(call["layout"].position_ids[call["layout"].segments[-1][0]:]).astype(int)
        axis = 1 if direction == "horizontal" else 2
        assert set(ids[:, axis]) == set(range(0, 12 if axis == 1 else 16, 2))
        cells = cc.tokens_to_cells(cc.interlace_token_mask(12, 16, 2, 0, False, direction))
        true_v = (npy(x) - npy(x0)) / 0.5
        assert np.allclose(npy(out[0])[..., cells], true_v[..., cells], atol=1e-4)

def _diagonal_scene():
    yy, xx = np.meshgrid(np.arange(48), np.arange(64), indexing="ij")
    img = ((xx - yy) % 24 < 12).astype(np.float32) * 2.0            # diagonal stripes, like slanted letters
    return tensor(np.broadcast_to(img, (1, 4, 2, 48, 64)).copy())

def test_interlace_edge_directed_follows_diagonals():
    x0 = _diagonal_scene()
    cells = cc.tokens_to_cells(cc.interlace_token_mask(24, 32, 2, 0))
    err = {m: float(np.abs(npy(cc.build_fill_plan(cells, m, 2, 1).apply(x0)) - npy(x0)).mean()) for m in ("bilinear", "edge_directed")}
    print("   diagonal fill error:", {k: round(v, 4) for k, v in err.items()})
    assert err["edge_directed"] < err["bilinear"] * 0.7

def test_interlace_with_every_setting():
    x0, x, _ = scene(ramp=False)
    x0b = tensor(np.broadcast_to(np.sin(np.arange(48)[:, None] / 9.0) + np.cos(np.arange(64)[None] / 9.0), (1, 4, 3, 48, 64)).copy())
    m = np.ones((1, 1, 3, 24, 32), np.float32); m[:, :, 0] = 0
    for direction in ("horizontal", "vertical"):
        for method in cc.INTERPOLATION_METHODS:
            for kw in ({}, {"pct": 67}, {"grid": 4}, {"flip": True}, {"alternate": True}, {"shift": True}):
                w = _lines(direction, method, **kw)
                out, _ = run(w, x, x0, sigma=1.0, step=0)
                for step, sigma in ((1, 0.875), (2, 0.75)):
                    out, ex = run(w, x, x0, sigma=sigma, step=step)
                    assert np.isfinite(npy(out[0])).all() and out[0].shape == x.shape, (direction, method, kw)
            out, _ = run(_lines(direction, method), x, x0, mask=tensor(m))
            assert np.isfinite(npy(out[0])).all()
            w = _lines(direction, method, ss=2.0)
            run(w, x, x0, sigma=1.0, step=0)
            out, ex = run(w, x, x0b, sigma=0.5, step=1)
            assert ex.calls[0]["strip"][3:] == ((24, 64) if direction == "horizontal" else (48, 32)) and out[0].shape == x.shape

def test_interlace_flip_and_alternate_swap_the_lines():
    x0, x, _ = scene(ramp=False)
    _, ex = run(_lines(flip=True), x, x0)
    layout = ex.calls[0]["layout"]
    ids = npy(layout.position_ids[layout.segments[-1][0]:]).astype(int)
    a = set(ids[ids[:, 0] == 0][:, 1]); b = set(ids[ids[:, 0] == 1][:, 1])
    assert not (a & b) and len(a | b) == 12                         # like classic interlaced video fields
    w = _lines(alternate=True, start=1)
    assert [w.parity_for_step(i, 8) for i in (1, 2, 3)] == [0, 1, 0]

def test_interlace_off_leaves_checkerboard_as_it_was():
    x0, x, _ = scene(ramp=False)
    out_a, ex = run(make(grid=2), x, x0)
    assert ex.calls[0]["strip"] == (1, 4, 3, 24, 16)
    assert make().interlace is None

def test_uses_comfy_sigma_not_timestep_scale():
    x0, x, _ = scene(ramp=False)
    w = make()
    ex = FakeExecutor(x0, 0.5)
    to = {"sample_sigmas": tensor([1.0, 0.5, 0.25, 0.0]), "cond_or_uncond": [0], "sigmas": tensor([0.5])}
    # a model with a different timestep multiplier: timestep alone would give the wrong step
    out = w(ex, [x, tensor(np.ones((1, 32, 2, 10)))], tensor([0.5 * 777.0]), tensor(np.ones((1, 7, 8))), to,
            minimax_payload={})
    assert w._last_step == 1 and not ex.calls[0]["full"]
    true_v = (npy(x) - npy(x0)) / 0.5
    cells = cc.tokens_to_cells(cc.balance_rows(cc.active_token_mask(12, 16, 4, 0)))
    assert np.allclose(npy(out[0])[..., cells], true_v[..., cells], atol=1e-4)

class _Window:
    def __init__(self, idx): self.index_list = idx

def test_context_windows_never_share_cached_images():
    x0a, x, _ = scene(ramp=False, seed=1)
    x0b = x0a * -1.0
    w = make("previous_step", start=0, end=0)
    audio, ctx = tensor(np.ones((1, 32, 2, 10))), tensor(np.ones((1, 7, 8)))
    for idx, x0 in (([0, 1, 2], x0a), ([3, 4, 5], x0b)):
        to = {"sample_sigmas": tensor([1.0, 0.5, 0.0]), "cond_or_uncond": [0], "sigmas": tensor([1.0]),
              "context_window": _Window(idx)}
        w(FakeExecutor(x0, 1.0), [x, audio], tensor([1000.0]), ctx, to, minimax_payload={})
    keys = set(w._x0_cache)
    assert keys == {(0, (0, 1, 2)), (0, (3, 4, 5))}
    assert not np.allclose(npy(w._x0_cache[(0, (0, 1, 2))][0]), npy(w._x0_cache[(0, (3, 4, 5))][0]))

def test_caches_stay_small_when_payloads_change_every_call():
    x0, x, _ = scene(ramp=False)
    w = make()
    for i in range(30):   # context windows hand over a fresh payload, and so a fresh layout, every call
        run(w, x, x0, step=1)
        w._layouts[("fresh", i)] = None
    assert len(w._layouts) <= 8 and not hasattr(w, "stats")
    c = cc._SmallCache(3)
    for i in range(10): c[i] = i
    assert list(c) == [7, 8, 9]

def test_shifted_small_grid_keeps_full_render_where_fill_cannot_reach():
    # deliberate: shifted steps double one-token tiles, and gaussian cannot fill gaps that wide well,
    # so those steps render in full. Filling them anyway caused heavy artifacts.
    w = cc.PartialRenderWrapper(2, "even", "gaussian", False, 1, 1, pad, FakeLayout, shift_grid=True)
    assert w._plan(24, 32, 0, False)[1] is not None and w._plan(24, 32, 0, True)[1] is None

if __name__ == "__main__":
    print("backend:", BACKEND)
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try: fn(); print("ok  ", name)
            except Exception as e:
                import traceback; fails += 1; print("FAIL", name); traceback.print_exc()
    print("failures:", fails); sys.exit(fails)
