"""Image input for Qwen3.5: loading, preprocessing, placeholder expansion and M-RoPE.

The pipeline mirrors the Hugging Face ``Qwen2VLImageProcessorPil`` + Qwen3.5 model:

1. resize so both sides are multiples of ``patch_size * merge_size`` (32) and the pixel
   count lies in ``[shortest_edge, longest_edge]``; rescale to [0, 1] and normalize;
2. cut ``16x16`` patches in 2x2-merge-block order, repeating each image twice along time
   (``temporal_patch_size=2``) -> ``[num_patches, 3*2*16*16]``;
3. the chat template emits one ``<|image_pad|>`` per image; it is expanded to
   ``t*h*w / merge**2`` tokens, one per merged vision feature;
4. text tokens get equal (t, h, w) positions; an image's tokens get ``(t, row, col)`` on its
   merged grid; the next text resumes at ``start + max(h, w)``. Decode positions are the
   K/V index plus ``rope_delta = max_position + 1 - prompt_length``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from mysglang.config import VisionConfig


@dataclass(frozen=True)
class MultimodalInputs:
    """Everything a request needs beyond its token IDs to run its images."""

    pixel_values: torch.Tensor  # [num_patches, patch_dim] float32 (CPU)
    image_grid_thw: tuple[tuple[int, int, int], ...]
    mrope_positions: torch.Tensor  # [3, prompt_length] long (CPU)
    rope_delta: int
    image_token_id: int
    # 提示词中每个图片占位 token 的下标，按出现顺序对应视觉特征的行。
    image_token_positions: tuple[int, ...] = field(default=())

    @property
    def num_image_tokens(self) -> int:
        return len(self.image_token_positions)

    @cached_property
    def image_cache_ids(self) -> tuple[int, ...]:
        """每个图片占位 token 在前缀缓存钥匙里用的 id：同一张图共用一个由像素指纹生成的负数。

        占位 token 的 id 全一样，直接拿来当 radix 的钥匙会把不同图片当成同一个前缀；
        换成指纹后，同一张图（同一段像素、同样的网格）才会命中。负数不会和词表 id 冲突。
        """
        patches_per_token = self.pixel_values.size(0) // max(1, self.num_image_tokens)
        ids: list[int] = []
        offset = 0
        for t, h, w in self.image_grid_thw:
            patches = t * h * w
            digest = hashlib.sha256(repr((t, h, w)).encode())
            digest.update(self.pixel_values[offset : offset + patches].numpy().tobytes())
            image_id = -1 - int.from_bytes(digest.digest()[:7], "big")
            ids.extend([image_id] * (patches // patches_per_token))
            offset += patches
        if len(ids) != self.num_image_tokens:
            raise ValueError("image grids disagree with the number of image tokens")
        return tuple(ids)

    def cache_key(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        """把 token 序列里的图片占位换成指纹 id，作为 radix 前缀缓存的钥匙（不送进模型）。"""
        key = list(token_ids)
        for position, image_id in zip(self.image_token_positions, self.image_cache_ids):
            if position >= len(key):
                break  # 只截了一段前缀（快照边界可能切在图片中间）
            key[position] = image_id
        return tuple(key)


def load_image(source):
    """Open a PIL image from a PIL object, bytes, path, ``file://``, ``http(s)://`` or data URL."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("image input requires the 'pillow' package") from exc
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source))
    if not isinstance(source, str):
        raise TypeError("image source must be a PIL image, bytes or a string")
    if source.startswith("data:"):
        header, _, payload = source.partition(",")
        if ";base64" not in header:
            raise ValueError("data URL images must be base64 encoded")
        return Image.open(io.BytesIO(base64.b64decode(payload)))
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as response:
            return Image.open(io.BytesIO(response.read()))
    return Image.open(Path(source.removeprefix("file://")).expanduser())


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int):
    """Closest size with sides divisible by ``factor`` and pixels in ``[min, max]``."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("image aspect ratio must be smaller than 200")
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return resized_height, resized_width


@dataclass(frozen=True)
class ImageProcessor:
    vision: VisionConfig
    min_pixels: int = 65536
    max_pixels: int = 16777216
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    @classmethod
    def from_pretrained(cls, model_path: str | Path, vision: VisionConfig) -> ImageProcessor:
        path = Path(model_path) / "preprocessor_config.json"
        if not path.is_file():
            return cls(vision)
        raw = json.loads(path.read_text())
        size = raw.get("size") or {}
        return cls(
            vision,
            min_pixels=size.get("shortest_edge", raw.get("min_pixels", cls.min_pixels)),
            max_pixels=size.get("longest_edge", raw.get("max_pixels", cls.max_pixels)),
            image_mean=tuple(raw.get("image_mean", cls.image_mean)),
            image_std=tuple(raw.get("image_std", cls.image_std)),
        )

    def preprocess(self, image) -> tuple[torch.Tensor, tuple[int, int, int]]:
        from PIL import Image

        vision = self.vision
        image = load_image(image).convert("RGB")
        factor = vision.patch_size * vision.spatial_merge_size
        height, width = smart_resize(
            image.height, image.width, factor, self.min_pixels, self.max_pixels
        )
        resized = np.asarray(image.resize((width, height), resample=Image.BICUBIC))
        # 与 HF 一致：float64 下乘 1/255 后转 float32，再按通道归一化。
        pixels = (resized.astype(np.float64) * (1 / 255)).astype(np.float32).transpose(2, 0, 1)
        mean = np.asarray(self.image_mean, dtype=np.float32)[:, None, None]
        std = np.asarray(self.image_std, dtype=np.float32)[:, None, None]
        pixels = (pixels - mean) / std

        patch, merge = vision.patch_size, vision.spatial_merge_size
        temporal = vision.temporal_patch_size
        grid_h, grid_w = height // patch, width // patch
        patches = pixels.reshape(3, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
        # [gh, gw, mh, mw, C, ph, pw]：同一个 2x2 合并块的 4 个 patch 相邻，合并器直接 reshape。
        patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)
        patches = np.broadcast_to(
            patches[:, :, :, :, :, None], (*patches.shape[:5], temporal, patch, patch)
        )
        flat = np.ascontiguousarray(patches).reshape(grid_h * grid_w, vision.patch_dim)
        return torch.from_numpy(flat.copy()), (1, grid_h, grid_w)


def image_token_count(grid_thw: Sequence[int], merge_size: int) -> int:
    t, h, w = grid_thw
    return t * h * w // (merge_size * merge_size)


def expand_image_tokens(
    token_ids: Sequence[int],
    grids: Sequence[tuple[int, int, int]],
    image_token_id: int,
    merge_size: int,
) -> list[int]:
    """Repeat each template ``<|image_pad|>`` once per merged vision feature."""
    result: list[int] = []
    images = iter(grids)
    for token in token_ids:
        if token == image_token_id:
            try:
                grid = next(images)
            except StopIteration as exc:
                raise ValueError("more image placeholders than images") from exc
            result.extend([image_token_id] * image_token_count(grid, merge_size))
        else:
            result.append(token)
    if next(images, None) is not None:
        raise ValueError("more images than image placeholders")
    return result


def mrope_positions(
    token_ids: Sequence[int],
    grids: Sequence[tuple[int, int, int]],
    image_token_id: int,
    merge_size: int,
) -> tuple[torch.Tensor, int]:
    """3-axis positions for a prompt with expanded image tokens, and its decode offset."""
    columns: list[torch.Tensor] = []
    images = iter(grids)
    position = 0
    index = 0
    while index < len(token_ids):
        if token_ids[index] != image_token_id:
            end = index
            while end < len(token_ids) and token_ids[end] != image_token_id:
                end += 1
            columns.append(torch.arange(position, position + end - index).expand(3, -1))
            position += end - index
            index = end
            continue
        t, h, w = next(images)
        h, w = h // merge_size, w // merge_size
        grid_t, grid_h, grid_w = torch.meshgrid(
            torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij"
        )
        columns.append(torch.stack((grid_t, grid_h, grid_w)).reshape(3, -1) + position)
        count = t * h * w
        if any(token != image_token_id for token in token_ids[index : index + count]):
            raise ValueError("image placeholder run does not match its image size")
        position += max(h, w)
        index += count
    positions = torch.cat(columns, dim=1).long()
    return positions, int(positions.max()) + 1 - len(token_ids)


def build_multimodal_inputs(
    token_ids: Sequence[int],
    images: Sequence,
    processor: ImageProcessor,
) -> tuple[list[int], MultimodalInputs]:
    """Preprocess images and expand their placeholders in a rendered chat prompt."""
    vision = processor.vision
    processed = [processor.preprocess(image) for image in images]
    grids = tuple(grid for _, grid in processed)
    expanded = expand_image_tokens(
        token_ids, grids, vision.image_token_id, vision.spatial_merge_size
    )
    positions, delta = mrope_positions(
        expanded, grids, vision.image_token_id, vision.spatial_merge_size
    )
    return expanded, MultimodalInputs(
        pixel_values=torch.cat([pixels for pixels, _ in processed]),
        image_grid_thw=grids,
        mrope_positions=positions,
        rope_delta=delta,
        image_token_id=vision.image_token_id,
        image_token_positions=tuple(
            index for index, token in enumerate(expanded) if token == vision.image_token_id
        ),
    )
