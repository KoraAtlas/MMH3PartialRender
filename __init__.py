from typing_extensions import override

from comfy_api.latest import ComfyExtension

from .nodes import H3CheckerboardPatch, H3SupersampleDownscale, H3SupersampleLatent, H3SupersampleStop


class H3PartialRenderExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [H3CheckerboardPatch, H3SupersampleLatent, H3SupersampleDownscale, H3SupersampleStop]


async def comfy_entrypoint():
    return H3PartialRenderExtension()
