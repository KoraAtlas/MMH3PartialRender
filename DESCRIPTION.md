# GitHub description

> Repository "About" line: Faster MiniMax H3 video sampling in ComfyUI: renders part of the frame on some steps (checkerboard or interlaced) and fills in the rest.

# ComfyUI H3 Partial Render

Faster MiniMax H3 video generation in ComfyUI. On selected sampling steps, **H3 Partial Render** sends only part of each frame to the model, as a checkerboard of tiles or as interlaced lines, and fills in the rest. Steps that render part of the frame run noticeably faster, and full steps at the start and end keep the overall composition and fine detail intact.

The idea comes from checkerboard rendering in games: render half the pixels, reconstruct the other half, and spend the saved time elsewhere.

## Goals

- **Real speedups, not masking.** Skipped parts of the frame never reach the model, so the time saved is real.
- **Stay coherent.** Rendered tokens keep their true positions in the frame, so the image stays aligned. Text, audio and reference conditioning always go to the model in full.
- **Work with what you already use.** It's a model patch, so it works with any sampler and with [H3-Optimizations](https://github.com/Zironic/H3-Optimizations) memory optimization and sparse attention.
- **Stay out of the way.** It changes no ComfyUI files. Turn it off and the model passes through untouched. Delete the folder to uninstall.

## Features

- **Checkerboard or interlaced rendering**, with horizontal or vertical lines.
- **Adjustable pattern:** tile size down to one H3 token (32 px), rectangular tiles, and the share of the frame rendered (25–100%).
- **Nine fill modes**, including edge-directed fill that follows edges, detail transfer from earlier steps, and temporal fill from neighboring frames.
- **Step control:** choose which steps render in full at the start and end, and optionally swap or shift the pattern from step to step or frame to frame.
- **Optional supersampling during partial steps:** spend some of the saved time rendering those steps at a higher resolution, then shrink each result back to your size.
- **Whole-run supersample nodes:** H3 Supersample Latent, H3 Supersample Downscale and H3 Supersample Stop, for rendering larger than your target and scaling down.

## Install

1. Copy the `ComfyUI-H3-PartialRender` folder into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI.
3. Search for **H3 Partial Render** in the node menu.

No extra Python packages are needed.

## Example workflow

`example_workflows/H3_Partial_Render_t2v_turbo_0.4MP.json` is a bare-minimum text-to-video workflow. Drag it into ComfyUI, or find it under **Templates** after installing.

```
Load Diffusion Model -> Turbo LoRA -> H3 Partial Render -> BasicGuider + BasicScheduler
Load CLIP + Video VAE -> MiniMax H3 Image to Video (prompt, size, length)
SamplerCustomAdvanced -> VAE Decode + VAE Decode Audio -> Create Video -> Save Video
```

| Setting | Value |
|---|---|
| Size | 832 × 480 (about 0.4 MP) |
| Length | 124 frames at 24 fps (about 5 s) |
| Steps | 6, with the MiniMax H3 turbo LoRA at 1.0 |
| Sampler / scheduler | res_multistep / simple |
| Guider | BasicGuider (CFG 1, as turbo expects) |
| Seed | 42, fixed |
| H3 Partial Render | node defaults |

The models are the ones in ComfyUI's MiniMax H3 template. Swap in your own files if their names differ. To add H3-Optimizations, place its nodes between **H3 Partial Render** and the guider.

**Sample prompt**

```
Cinematic nature shot, golden hour, shallow depth of field, natural color, mood calm and quiet.

Timeline:
[0s-2s] A red fox trots through a snowy birch forest, low tracking shot at the fox's eye level, snow falling softly in the warm backlight.
[2s-4s] The fox stops, ears turning toward the camera, breath fogging in the cold air, fine detail in its fur.
[4s-5s] It leaps over a fallen log and runs out of frame, powder snow drifting behind it.

Single continuous shot, smooth handheld camera, no cuts.

Audio: soft wind through bare branches, the crunch of paws in snow, a distant crow, no music.
```

**H3 Partial Render defaults**

| Setting | Value |
|---|---|
| `enable` | on |
| `grid_height` / `grid_width` | 2 / 2 |
| `render_percent` | 50 |
| `interpolation` | edge_directed |
| `start_step` | 2 |
| `full_steps_at_end` | 2 |
| `interlace` | off |
| `supersample` | 1.0 |

With 6 steps, these settings render steps 3 and 4 partially and the rest in full. Check the console for `[H3 Partial Render] sending N of M video tokens to the model`. N should be about half of M.

To add it to an existing H3 workflow instead, put **H3 Partial Render** on the MODEL line after your LoRAs and before the guider. Nothing else changes.

## Tuning

- **Faster:** lower `full_steps_at_end` or `start_step` to 1. On 6 turbo steps, 1 and 1 render four of the six steps partially.
- **Cleaner:** raise `render_percent` to 60–67, or try `detail_transfer`.
- **Still or slow shots:** try `temporal` fill.
- **Visible tiles or lines:** keep the grid at 2, and leave `shift_grid_each_step` off unless you've compared the results.

## Limits

- MiniMax H3 only. Other models raise an error unless the node is turned off.
- ControlNet isn't supported on partial steps.
- FastH3/VSA checkpoints were trained on dense token blocks, so expect lower quality with the H3 VSA Attention node.
- Partial steps trade some fine detail for speed. Full steps at the end recover most of it, but very fine text and thin lines are the hardest to fill.

See `README.md` for every setting and how each one works.

---

# CivitAI description

**H3 Partial Render: faster MiniMax H3 video in ComfyUI**

Speeds up MiniMax H3 by rendering only part of each frame on selected steps, as a checkerboard or interlaced lines, and filling in the rest. Like checkerboard rendering in games: render half, reconstruct half, and keep the time saved.

**What it does**
- Skipped parts of the frame never reach the model, so the time saved is real.
- Full steps at the start and end keep composition and fine detail.
- Works with any sampler, LoRAs and H3-Optimizations sparse attention.
- One node on the MODEL line. No ComfyUI files are changed.

**Features**
- Checkerboard or interlaced (horizontal or vertical) patterns
- Tile size down to 32 px, adjustable share of the frame rendered
- 9 fill modes, including edge-directed, detail transfer and temporal
- Optional supersampling during partial steps for extra detail
- Whole-run supersample and downscale nodes

**Example workflow (included)**
Text to video, 832 × 480 (about 0.4 MP), 124 frames (about 5 s), 6 steps with the H3 turbo LoRA, res_multistep, BasicGuider:
`Load Diffusion Model -> Turbo LoRA -> H3 Partial Render -> Guider -> SamplerCustomAdvanced -> VAE Decode + VAE Decode Audio -> Save Video`

Sample prompt: a red fox crossing a snowy birch forest at golden hour, with a timeline and ambient audio. Full text is in the workflow.

To add it to your own H3 workflow, put H3 Partial Render on the MODEL line after your LoRAs. The defaults are a good start: grid 2×2, 50% rendered, edge_directed fill, 2 full steps at the start and 2 at the end.

**Install:** copy the folder into `ComfyUI/custom_nodes` and restart ComfyUI. The example workflow is in `example_workflows`.

**Limits:** MiniMax H3 only. No ControlNet on partial steps. Very fine text and thin lines are the hardest to fill.
