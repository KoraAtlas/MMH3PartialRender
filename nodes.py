import logging

import torch

import comfy.ldm.common_dit
import comfy.nested_tensor
import comfy.patcher_extension
from comfy_api.latest import io

from .partial_render_core import PartialRenderWrapper, INTERPOLATION_METHODS
from .supersample_core import DOWNSCALE_METHODS, downscale_mid_sampling, resize_video, supersample_size

WRAPPER_KEY = "h3_partial_render"
SIZE_KEY = "h3_supersample_size"


def _video_and_audio(latent):
    samples = latent["samples"]
    if not samples.is_nested or len(samples.tensors) != 2 or samples.tensors[0].ndim != 5:
        raise ValueError("This node needs a MiniMax H3 audio/video latent.")
    return samples.tensors[0], samples.tensors[1]


class H3CheckerboardPatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3CheckerboardPatch",
            display_name="H3 Partial Render",
            category="model/patch",
            description="Checkerboard or interlaced rendering for MiniMax H3. Skipped tokens never reach the model and are "
                        "interpolated from the rendered tiles. Works with any sampler, and with H3-Optimizations "
                        "memory and sparse attention. With the toggle off, the model passes through untouched.",
            search_aliases=["checkerboard", "h3 checkerboard", "partial render", "tile skipping", "interlace", "interlacing"],
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enable", default=True),
                io.Int.Input("grid_height", default=2, min=2, max=256, step=2,
                             tooltip="Tile height in latent cells. 2 cells is one H3 token (32 px), the smallest "
                                     "unit the model can skip."),
                io.Int.Input("grid_width", default=2, min=2, max=256, step=2,
                             tooltip="Tile width in latent cells. Different height and width give bricks or stripes."),
                io.Int.Input("render_percent", default=50, min=25, max=100, step=1,
                             tooltip="Share of video tokens sent to the model on sparse steps. Higher leaves smaller "
                                     "gaps and runs slower. 100 renders everything."),
                io.Boolean.Input("flip_pattern_per_frame", default=False,
                                 tooltip="Swaps even/odd between latent frames, so a skipped token is rendered in the "
                                         "frames next to it. temporal turns this on by itself."),
                io.Combo.Input("pattern", options=["even", "odd"], default="even",
                               tooltip="Which tiles or lines to render. Checkerboard even: (row + column) is even. "
                                       "Interlace even: the first line is rendered."),
                io.Combo.Input("interpolation", options=INTERPOLATION_METHODS, default="edge_directed",
                               tooltip="How skipped tokens are filled. edge_directed fills along edges. detail_transfer "
                                       "adds back fine detail from the latest whole frame and needs start_step 1 or "
                                       "more. temporal borrows from neighboring frames where nothing moved."),
                io.Boolean.Input("alternate_each_step", default=False,
                                 tooltip="Swap even/odd every step. Off lets one tile set decide the image, so the two "
                                         "sets cannot disagree; full steps at the end then render the rest."),
                io.Boolean.Input("shift_grid_each_step", default=False,
                                 tooltip="Moves tile boundaries every other step."),
                io.Int.Input("start_step", default=2, min=0, max=10000,
                             tooltip="Steps before this one render in full, which settles the layout. Counts from 0."),
                io.Int.Input("full_steps_at_end", default=2, min=0, max=10000,
                             tooltip="How many final steps render in full, to restore detail in filled tiles."),
                io.Float.Input("supersample", default=1.0, min=1.0, max=2.0, step=0.05,
                               tooltip="Renders sparse steps at this size per side and shrinks each result back. Full "
                                       "steps stay at normal size. Needs start_step 1 or more. Sparse-step cost grows "
                                       "with its square: at 50% rendered, 1.2 is about 0.72x a full step, 1.4 about 1x."),
                io.Boolean.Input("interlace", default=False,
                                 tooltip="Renders alternating lines instead of a checkerboard. Every other setting "
                                         "works the same way."),
                io.Combo.Input("interlace_direction", options=["horizontal", "vertical"], default="horizontal",
                               tooltip="Which way the lines run. Line thickness is grid_height for horizontal lines "
                                       "and grid_width for vertical ones. Fine detail running along the lines "
                                       "survives best, so vertical may suit tall subjects."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, enable, grid_height, grid_width, render_percent,
                flip_pattern_per_frame, pattern, interpolation, alternate_each_step,
                shift_grid_each_step, start_step, full_steps_at_end, supersample, interlace,
                interlace_direction) -> io.NodeOutput:
        if not enable:
            return io.NodeOutput(model)

        # imported here so a ComfyUI without H3 can still load this pack
        from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout
        if not isinstance(model.get_model_object("diffusion_model"), MiniMaxH3Model):
            raise ValueError("H3 Partial Render supports MiniMax H3 only.")

        wrapper = PartialRenderWrapper((grid_height, grid_width), pattern, interpolation, alternate_each_step, start_step,
                                      full_steps_at_end, comfy.ldm.common_dit.pad_to_patch_size, PackedLayout,
                                      shift_grid=shift_grid_each_step, render_percent=render_percent,
                                      flip_per_frame=flip_pattern_per_frame, supersample=supersample,
                                      interlace=interlace_direction if interlace else None)
        patched = model.clone()
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, wrapper)
        return io.NodeOutput(patched)


class H3SupersampleLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SupersampleLatent",
            display_name="H3 Supersample Latent",
            category="latent/minimax",
            description="Enlarges an H3 video latent before sampling, and remembers its size for H3 Supersample "
                        "Downscale. Audio is untouched. Place it before guide nodes, whose keyframes must match the "
                        "latent size.",
            search_aliases=["supersample", "ssaa", "h3 supersample"],
            inputs=[
                io.Latent.Input("latent"),
                io.Float.Input("scale", default=1.4, min=1.0, max=2.0, step=0.05,
                               tooltip="Size multiplier per side. Token count and full-step cost grow with its "
                                       "square: 1.2 is about 1.44x, 1.4 about 2x."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, latent, scale) -> io.NodeOutput:
        video, audio = _video_and_audio(latent)
        if latent.get("noise_mask") is not None:
            raise ValueError("H3 Supersample Latent does not support latents with a noise mask.")
        h, w = video.shape[3], video.shape[4]
        size = supersample_size(h, w, scale)
        if torch.count_nonzero(video) == 0:
            bigger = video.new_zeros(video.shape[:3] + size)
        else:
            bigger = resize_video(video, size, "bicubic")
        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((bigger, audio))
        out.setdefault(SIZE_KEY, (h, w))
        logging.info("[H3 Partial Render] supersampling %dx%d to %dx%d latent cells (%.2fx tokens).",
                     h, w, size[0], size[1], size[0] * size[1] / float(h * w))
        return io.NodeOutput(out)


class H3SupersampleDownscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SupersampleDownscale",
            display_name="H3 Supersample Downscale",
            category="latent/minimax",
            description="Returns a sampled H3 latent to the size it had before H3 Supersample Latent. Audio is "
                        "untouched. For the highest quality, skip this node, decode at full size and scale the "
                        "images down instead.",
            search_aliases=["supersample", "ssaa", "h3 supersample"],
            inputs=[
                io.Latent.Input("latent"),
                io.Combo.Input("method", options=DOWNSCALE_METHODS, default="area",
                               tooltip="area averages cells and is safest. bicubic is sharper and can ring on hard edges."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, latent, method) -> io.NodeOutput:
        size = latent.get(SIZE_KEY)
        if size is None:
            raise ValueError("No original size found. Connect this after H3 Supersample Latent and a sampler.")
        video, audio = _video_and_audio(latent)
        out = latent.copy()
        out.pop(SIZE_KEY)
        out["samples"] = comfy.nested_tensor.NestedTensor((resize_video(video, size, method), audio))
        return io.NodeOutput(out)


class H3SupersampleStop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SupersampleStop",
            display_name="H3 Supersample Stop (Mid-Sampling)",
            category="latent/minimax",
            description="Ends supersampling part way through sampling. Connect between two samplers split with Split "
                        "Sigmas: it returns the partly noisy latent to its original size and corrects its noise level "
                        "for the next stage, which should use Disable Noise.",
            search_aliases=["supersample", "stop supersampling", "h3 supersample"],
            inputs=[
                io.Latent.Input("latent", tooltip="The first sampler's output, not denoised_output."),
                io.Sigmas.Input("next_sigmas", tooltip="The sigmas of the next sampler stage."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, latent, next_sigmas, seed) -> io.NodeOutput:
        size = latent.get(SIZE_KEY)
        if size is None:
            raise ValueError("No original size found. Connect this after H3 Supersample Latent and a sampler.")
        if next_sigmas.shape[-1] < 2:
            raise ValueError("next_sigmas has no steps left to run.")
        video, audio = _video_and_audio(latent)
        out = latent.copy()
        out.pop(SIZE_KEY)
        out["samples"] = comfy.nested_tensor.NestedTensor(
            (downscale_mid_sampling(video, size, next_sigmas[0], seed), audio))
        return io.NodeOutput(out)
