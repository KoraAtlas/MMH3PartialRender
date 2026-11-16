# Handoff: ComfyUI H3 Partial Render

Last updated 2026-11-15. Custom ComfyUI node pack that speeds up MiniMax H3 video sampling by sending only part of the video tokens to the model on some steps, as a checkerboard or as interlaced lines, and filling in the rest.

## Status

- **Working on the user's setup.** The user reports checkerboard mode gives a real speedup with good quality "with the correct settings" (exact settings not recorded).
- **Not yet verified on real H3:** interlace mode, the new fill modes (`edge_directed`, `detail_transfer`, `temporal`), `render_percent`, rectangular grids and per-step `supersample`. These pass logic tests only.
- **Logic tests:** `python tests/test_core.py` passes (0 failures). The dev container had no PyTorch, so tests ran on a numpy stand-in (`tests/_torch_shim.py`) and a fake model. They use real torch automatically if it's installed.
- **Latest package:** `ComfyUI-H3-PartialRender.zip` (built 2026-11-04). The user was told to delete the old `ComfyUI-H3-Checkerboard` folder before installing.

## User's setup

- ComfyUI with MiniMax H3 Ref2VA, Pika H3 Flex turbo LoRA (6-step) at 1.0, Euler, ManualSigmas, about 0.2 MP, RTX 3090.
- Uses **H3-Optimizations** (github.com/Zironic/H3-Optimizations) memory optimization and sparse attention. Compatibility with it is a hard requirement.
- Node order: `Load Model -> Turbo LoRA -> H3 Partial Render -> H3 Memory Optimization / H3 Sparse Attention -> Guider -> SamplerCustomAdvanced`

## User requirements

- Standalone custom node, no ComfyUI files modified, easy to install and uninstall.
- Skipped tokens must never reach the model (real compute savings, not masking).
- Options for alternating the pattern per step and a start step.
- Must stay compatible with H3-Optimizations.

## Files

| File | Purpose |
|---|---|
| `nodes.py` | Node definitions. `H3CheckerboardPatch` (display name **H3 Partial Render**) plus three whole-run supersample nodes |
| `partial_render_core.py` | Core logic, no ComfyUI imports: masks, fill plans, the `PartialRenderWrapper`, per-step supersampling |
| `supersample_core.py` | Resize helpers and size math, used by the supersample nodes and per-step supersampling |
| `__init__.py` | Registers nodes via `comfy_entrypoint` (V3 API) |
| `tests/test_core.py`, `tests/_torch_shim.py` | Logic tests and numpy stand-in for torch |
| `README.md` | User-facing docs |

The node ID stays `H3CheckerboardPatch` so saved workflows still load. Don't rename it.

## How it works

Implemented as a `WrappersMP.DIFFUSION_MODEL` wrapper added to a cloned ModelPatcher, so it works with any sampler and sits outside H3-Optimizations' patches.

On a partial step:

1. Patchify the latent into H3's 2x2 tokens and pick active tokens from the pattern.
2. Balance so every row (checkerboard) or every kept line (interlace) has the same count, then pass the active tokens as a **smaller regular grid** (`T x H x W/2`, or `T x H/2 x W` for horizontal lines), with a `SparseLayout` whose RoPE position ids are the tokens' **true** coordinates. Text, audio, keyframe and reference tokens are untouched.
3. Convert the prediction to x0, fill skipped cells in **x0 space**, then rebuild each skipped cell's velocity as `(x - x0) / sigma` so it keeps its own noise.

Step index comes from `transformer_options["sample_sigmas"]`. Caches reset when the step index goes backwards (a new run).

## Node inputs (H3 Partial Render)

| Input | Default | Notes |
|---|---|---|
| `enable` | on | Off returns the model unchanged |
| `grid_height`, `grid_width` | 2, 2 | Latent cells, even only. 2 = one token (32 px), the minimum |
| `render_percent` | 50 | Share of tokens or lines rendered on partial steps |
| `flip_pattern_per_frame` | off | Swap parity between latent frames. `temporal` forces it on |
| `pattern` | even | Which tiles or lines render |
| `interpolation` | edge_directed | `edge_directed`, `detail_transfer`, `temporal`, `bilinear`, `bicubic`, `nearest`, `gaussian`, `harmonic`, `previous_step` |
| `alternate_each_step` | off | Swap parity each step |
| `shift_grid_each_step` | off | Offset tile boundaries every other step |
| `start_step` | 2 | Steps before this render in full |
| `full_steps_at_end` | 2 | Final steps that render in full |
| `supersample` | 1.0 | Render partial steps larger, shrink back. Needs `start_step` >= 1 |
| `interlace` | off | Lines instead of checkerboard |
| `interlace_direction` | horizontal | Line thickness = `grid_height` (horizontal) or `grid_width` (vertical) |

## Code review, 2026-11-15

Logic tests and a stress run (1,728 setting combinations, two 8-step runs each) pass. Fixed in that review:

- **Reverted the same day:** widening the fill reach so shifted 2-cell steps with `gaussian` or `harmonic` render partially. It caused heavy artifacts on the user's setup. Those steps render in full, as originally, and a test now locks that in. An update to the supersample guide on fallback steps was reverted too. Without context windows, output now matches the pre-review version exactly in every tested configuration.
- **Context windows:** cached x0 (`previous_step`, `detail_transfer`, supersample guide) was keyed only by cond/uncond, so windows of the same size could reuse each other's image. Keys now include the window's frame indices.
- **Unbounded caches:** context windows send a new payload every call, so layout and supersample payload caches grew for the whole run. They now keep only the latest 8 entries. A debug `stats` list that grew forever was removed.
- **Sigma:** now read from `transformer_options["sigmas"]` instead of assuming `timestep / 1000`.
- **Redundancy:** `balance_rows` reuses `_balance_line`, and the supersample guide lookup is simpler.

## Approaches tried and dropped

Avoid re-trying these without a new idea:

- **Packed tiles / 1xN strip of active tokens.** Caused misaligned tiles and broken anatomy, partly because H3-Optimizations builds 3D cubes from grid position. Replaced by the regular compact grid.
- **Two passes per step (even then odd on the same latent).** No speedup at all (cost scales with token count, so two halves equal one whole) and garbled text, because the two halves never see each other.
- **Block skipping (half the tokens sit out some transformer blocks).** Worse artifacts and incompatible with H3-Optimizations (its cube ordering reorders tokens and its routing rejects changed token counts).
- **CPU + GPU split.** Discussed only. CPU is far too slow to help. A second GPU could make two passes useful.

## Known limits

- MiniMax H3 only. ControlNet raises an error on partial steps.
- FastH3/VSA checkpoints were trained on solid token cubes; the compact grid's cubes have gaps, so expect lower quality with the H3 VSA Attention node.
- Per-step `supersample` above about 1.4 costs more than a full step at normal size, and skips steps with a noise mask or no finished x0 yet.
- Interlace fills are weakest on detail running across the lines (for example, thin horizontal strokes with horizontal lines).

## Other contents not written in this conversation

`H3SupersampleLatent`, `H3SupersampleDownscale` and `H3SupersampleStop` plus `supersample_core.py` appeared in the folder from another session. They supersample the whole run and are kept in the pack. Don't combine `H3SupersampleLatent` with per-step `supersample` unless both scales should multiply.

## Suggested next steps

1. Get the user's current best checkerboard settings and record them as recommended defaults.
2. Verify interlace mode on real H3 with H3-Optimizations, using the title-text shot as the test case.
3. Test `supersample` at 1.2 and 1.3, watching for flicker from the size switch.
4. Run the tests with real PyTorch if available.
