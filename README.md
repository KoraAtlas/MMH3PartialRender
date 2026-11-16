# ComfyUI H3 Partial Render

Speeds up MiniMax H3 by sending only part of the video to the model on some steps, as a checkerboard or as interlaced lines, and filling in the rest. Adds **H3 Partial Render** and optional whole-run supersample nodes. It changes no ComfyUI files. To uninstall, delete this folder.

## Install

Put this folder in `ComfyUI/custom_nodes/` and restart ComfyUI.

Place **H3 Partial Render** on the MODEL line, right after the model loader:

`Load Model -> H3 Partial Render -> H3 Memory Optimization / H3 Sparse Attention -> Guider -> any sampler`

Node order only matters for readability, since the patch wraps the model call from the outside.

## Settings

| Input | Default | What it does |
|---|---|---|
| `enable` | on | Off passes the model through untouched. |
| `grid_height`, `grid_width` | 2, 2 | Tile size in latent cells, even numbers only. 2 cells is one H3 token (32 px), the smallest unit the model can skip. |
| `render_percent` | 50 | Share of video tokens sent to the model on sparse steps. Higher leaves smaller gaps and runs slower. |
| `flip_pattern_per_frame` | off | Swaps even and odd between latent frames, so a skipped token is rendered in the frames next to it. |
| `pattern` | even | Which tiles to render. Even tiles have an even (row + column). |
| `interpolation` | edge_directed | How skipped tiles are filled. |
| `alternate_each_step` | off | Swaps even and odd each step. Off lets one tile set decide the image, so the sets cannot disagree. |
| `shift_grid_each_step` | off | Moves tile boundaries every other step. |
| `start_step` | 2 | Steps before this one render in full. Counts from 0. |
| `full_steps_at_end` | 2 | How many final steps render in full. |

## Interlacing

Turn on `interlace` to render alternating lines instead of a checkerboard, and choose the direction with `interlace_direction`.

- **Line thickness:** `grid_height` for horizontal lines, `grid_width` for vertical ones. 2 is one token, the thinnest line possible.
- **Same controls:** `render_percent`, `pattern`, `alternate_each_step`, `flip_pattern_per_frame`, `shift_grid_each_step`, `start_step`, `full_steps_at_end`, `supersample` and every fill mode work the same way.
- **Classic interlacing:** `flip_pattern_per_frame` swaps the lines between latent frames, like interlaced video fields. `temporal` fill turns this on by itself.
- **Fill:** `edge_directed`, `detail_transfer` and `temporal` compare several angles across each gap and follow the best match, as deinterlacers do. Detail running across the lines is the hardest to fill, such as horizontal strokes with horizontal lines.

**Interpolation methods**

- `edge_directed`: fills along edges instead of across them. A good default.
- `detail_transfer`: `edge_directed`, plus fine detail from the latest whole frame. Needs `start_step` of 1 or more.
- `temporal`: `edge_directed`, plus neighboring frames wherever they agree. Turns on `flip_pattern_per_frame`.
- `bilinear`: linear blend across each gap.
- `bicubic`: cubic blend that follows the slope at tile edges. Sharper, but can overshoot at high contrast.
- `harmonic`: bilinear, then smoothed to remove creases.
- `gaussian`: soft weighted average of nearby rendered cells.
- `nearest`: copies the closest rendered cell. Fastest, with visible tile edges.
- `previous_step`: reuses the tile's last rendered result, falling back to bilinear if it has none. Best paired with `alternate_each_step`.

## Supersampling during partial steps

Set `supersample` on **H3 Partial Render**. Sparse steps render larger and shrink each result back to your size. Full steps and the final latent stay at your size, so no other nodes are needed.

- Sparse-step cost grows with the square of the value. At `render_percent` 50: 1.2 is about 0.72x a full step, 1.3 about 0.85x, and 1.4 about 1x. Above that, peak VRAM exceeds a full step.
- It needs `start_step` of 1 or more, because the first enlargement starts from a finished x0. Steps without one render at normal size.
- Latents with a noise mask render at normal size.
- Don't combine it with H3 Supersample Latent unless you want both sizes to multiply.

## Supersampling the whole run

**H3 Supersample Latent** enlarges the video latent before sampling, so tokens cover less of the final image. Audio is untouched.

Pick one way back down:

- **Highest quality:** `H3 Supersample Latent -> Sampler -> VAE Decode -> scale images to your target size (Lanczos)`
- **Less VAE work:** `H3 Supersample Latent -> Sampler -> H3 Supersample Downscale -> VAE Decode`

Notes:

- Token count grows with the square of `scale`: 1.2 is about 1.44x, and 1.4 about 2x. Full steps and peak VRAM grow by that much.
- The same seed gives a different image, because the noise has a new shape.
- Place the node before MiniMaxH3AddGuide, which reads the latent's size. Latents from MiniMaxH3ImageToVideo carry keyframes at their original size, so set the larger size on that node instead.
- Reference images keep their own size, so MiniMaxH3ReferenceToVideo latents can be enlarged directly.
- Latents with a noise mask are not supported.

### Stopping part way through

A sampler can't change latent size during a run, so this takes two stages:

`H3 Supersample Latent -> Sampler 1 (high sigmas, RandomNoise) -> H3 Supersample Stop -> Sampler 2 (low sigmas, DisableNoise)`

- Use Split Sigmas for the two schedules, and connect its low sigmas to `next_sigmas` as well.
- Use Sampler 1's `output`, not `denoised_output`.
- Step counts restart in each sampler, so give each stage its own H3 Partial Render: `full_steps_at_end` 0 on the first and `start_step` 0 on the second.
- Keyframes must match each stage's size, so this only suits workflows without them, such as reference or text-only ones.
- The last steps run at normal size, where fine detail forms. Expect better structure, not the finer detail of supersampling to the end.

## How it works

- **Skipped tokens never reach the model.** A sparse step sends about half the video tokens.
- **The model still sees a regular grid.** Active tokens are arranged as a narrower T x H x W/2 token grid, so patches inside the model, such as cube ordering and sparse routing, get a normal H3 call. Rows are balanced to equal counts, and any extra token is filled like a skipped tile.
- **Tiles stay aligned.** Tokens keep their true (t, h, w) RoPE coordinates, so the narrower grid changes how tokens are grouped, not where the model thinks they are.
- **Conditioning stays whole.** Text, audio, keyframe and reference tokens are always sent in full. A clip with long video keyframes saves less.
- **Filling happens in x0 space.** Each skipped cell's velocity is rebuilt as `(x - x0) / sigma`, which keeps that cell's own noise.

## Limits

- MiniMax H3 only. Other models raise an error unless the toggle is off.
- ControlNet raises an error on sparse steps.
- FastH3/VSA checkpoints were trained on a specific sparse pattern over contiguous cubes. The narrower grid's cubes contain gaps, so expect lower quality there.

## Checking it

With INFO logging, the console prints the token count once per run:

`[H3 Partial Render] sending N of M video tokens to the model (K other tokens unchanged).`

`python tests/test_core.py` runs the logic tests, using real torch if it is installed.
