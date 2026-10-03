"""Qwen3.5 image input: preprocessing, vision tower, M-RoPE and scheduling."""

from __future__ import annotations

import importlib.util
import random
import unittest
from dataclasses import replace

import numpy as np
import torch

from mysglang import ModelConfig, Qwen3ForCausalLM, Request, SamplingParams
from mysglang.config import VisionConfig
from mysglang.multimodal import ImageProcessor, build_multimodal_inputs, mrope_positions
from mysglang.scheduler import Scheduler, SchedulerConfig
from tests.helpers import drain_scheduler, export_hf_state_dict, reference_generate
from tests.test_hybrid import LAYER_TYPES, _TorchFallbackKernels

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
        Qwen2VLImageProcessorPil,
    )
except ImportError:
    Qwen3_5ForConditionalGeneration = None

IMAGE, START, END = 60, 58, 59
VISION = VisionConfig(
    depth=2,
    hidden_size=32,
    intermediate_size=64,
    num_heads=2,
    out_hidden_size=32,
    patch_size=4,
    num_position_embeddings=16,
    image_token_id=IMAGE,
    vision_start_token_id=START,
    vision_end_token_id=END,
)
PROCESSOR = ImageProcessor(VISION, min_pixels=64, max_pixels=4096)


def vl_config(*, mtp: bool = False) -> ModelConfig:
    return ModelConfig(
        model_type="qwen3_5",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_layers=len(LAYER_TYPES),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        partial_rotary_factor=0.5,
        attn_output_gate=True,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=LAYER_TYPES,
        mtp_num_hidden_layers=1 if mtp else 0,
        # 8 rotary frequencies interleaved as t,h,w,t,h,w,t,h: covers both h and w slots.
        mrope_section=(3, 3, 2),
        vision=VISION,
    )


def make_vl(seed: int = 3, *, mtp: bool = False) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(vl_config(mtp=mtp), enable_mtp=mtp, enable_vision=True).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight") and not parameter.abs().sum():
                parameter.add_(torch.randn_like(parameter) * 0.1)
    return model


def random_image(height: int, width: int, seed: int):
    pixels = np.random.RandomState(seed).rand(height, width, 3) * 255
    return Image.fromarray(pixels.astype("uint8"))


def image_prompt(images, text=(5, 6, 7), prefix=(1, 2, 3)):
    """A chat-like prompt with one ``<vision_start><image><vision_end>`` per image."""
    tokens = list(prefix)
    for _ in images:
        tokens += [START, IMAGE, END]
    return build_multimodal_inputs(tokens + list(text), images, PROCESSOR)


@torch.inference_mode()
def multimodal_reference(model, prompt, multimodal, max_new_tokens):
    """Full recomputation of every step with M-RoPE and vision features (no cache)."""
    features = model.encode_images(multimodal.pixel_values, multimodal.image_grid_thw)
    rows = torch.tensor(multimodal.image_token_positions)
    tokens = list(prompt)
    output = []
    for step in range(max_new_tokens):
        generated = torch.arange(len(prompt), len(prompt) + step) + multimodal.rope_delta
        positions = torch.cat((multimodal.mrope_positions, generated.expand(3, -1)), dim=1)
        logits = model(
            torch.tensor([tokens]),
            mrope_positions=positions[:, None, :],
            embedding_overrides=(rows, features),
        )
        token = int(logits[0, -1].argmax())
        output.append(token)
        tokens.append(token)
    return output


@unittest.skipUnless(Image is not None, "requires pillow")
class PreprocessingTest(unittest.TestCase):
    @unittest.skipUnless(Qwen3_5ForConditionalGeneration is not None, "requires transformers")
    def test_patches_match_hugging_face_pil_processor(self) -> None:
        reference = Qwen2VLImageProcessorPil(
            size={"shortest_edge": 64, "longest_edge": 4096},
            patch_size=4,
            temporal_patch_size=2,
            merge_size=2,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )
        for height, width in ((24, 40), (5, 9), (200, 130)):
            with self.subTest(size=(height, width)):
                image = random_image(height, width, height)
                pixels, grid = PROCESSOR.preprocess(image)
                expected = reference(images=[image], return_tensors="pt")
                self.assertEqual([list(grid)], expected["image_grid_thw"].tolist())
                torch.testing.assert_close(pixels, expected["pixel_values"], atol=0, rtol=0)

    def test_placeholders_expand_and_text_resumes_after_the_widest_side(self) -> None:
        tokens, multimodal = image_prompt([random_image(24, 40, 1)])
        # 24x40 -> 6x10 patches -> 3x5 merged tokens.
        self.assertEqual(multimodal.image_grid_thw, ((1, 6, 10),))
        self.assertEqual(multimodal.num_image_tokens, 15)
        # prefix + <vision_start> + 15 image tokens + <vision_end> + text.
        self.assertEqual(len(tokens), 3 + 1 + 15 + 1 + 3)
        positions = multimodal.mrope_positions
        first_image = multimodal.image_token_positions[0]
        torch.testing.assert_close(positions[:, first_image], torch.tensor([4, 4, 4]))
        torch.testing.assert_close(positions[:, first_image + 14], torch.tensor([4, 6, 8]))
        # <vision_end> follows at 4 + max(3, 5) = 9.
        self.assertEqual(positions[:, first_image + 15].tolist(), [9, 9, 9])
        self.assertEqual(multimodal.rope_delta, int(positions.max()) + 1 - len(tokens))

    def test_text_only_positions_are_plain_indices(self) -> None:
        positions, delta = mrope_positions([1, 2, 3, 4], (), IMAGE, 2)
        self.assertEqual(delta, 0)
        torch.testing.assert_close(positions, torch.arange(4).expand(3, -1))


@unittest.skipUnless(
    Image is not None and Qwen3_5ForConditionalGeneration is not None,
    "requires pillow and transformers with Qwen3.5",
)
class HuggingFaceMultimodalOracleTest(unittest.TestCase):
    @torch.inference_mode()
    def test_image_prompt_logits_match_hugging_face(self) -> None:
        model = make_vl()
        config = model.config
        hf_config = Qwen3_5Config(
            text_config=dict(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_hidden_layers=config.num_layers,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                rope_parameters={
                    "rope_type": "default",
                    "rope_theta": config.rope_theta,
                    "partial_rotary_factor": config.partial_rotary_factor,
                    "mrope_section": list(config.mrope_section),
                    "mrope_interleaved": True,
                },
                linear_key_head_dim=8,
                linear_value_head_dim=8,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
                layer_types=list(LAYER_TYPES),
                tie_word_embeddings=False,
            ),
            vision_config=dict(
                depth=VISION.depth,
                hidden_size=VISION.hidden_size,
                intermediate_size=VISION.intermediate_size,
                num_heads=VISION.num_heads,
                out_hidden_size=VISION.out_hidden_size,
                patch_size=VISION.patch_size,
                num_position_embeddings=VISION.num_position_embeddings,
                spatial_merge_size=2,
                temporal_patch_size=2,
            ),
            image_token_id=IMAGE,
            vision_start_token_id=START,
            vision_end_token_id=END,
            tie_word_embeddings=False,
        )
        hf_config.text_config._attn_implementation = "eager"
        hf_config.vision_config._attn_implementation = "eager"
        with _TorchFallbackKernels():
            reference = Qwen3_5ForConditionalGeneration(hf_config).eval()
            state = {
                name.replace("model.", "model.language_model.", 1)
                if name.startswith("model.") and not name.startswith("model.visual.")
                else name: tensor
                for name, tensor in export_hf_state_dict(model).items()
            }
            state = {
                ("model." + name if name.startswith("visual.") else name): tensor
                for name, tensor in state.items()
            }
            missing, unexpected = reference.load_state_dict(state, strict=False)
            self.assertEqual((missing, unexpected), ([], []))
            tokens, multimodal = image_prompt(
                [random_image(24, 40, 1), random_image(40, 16, 2)], text=(5, 6, 7, 8)
            )
            ids = torch.tensor([tokens])
            expected = reference(
                input_ids=ids,
                pixel_values=multimodal.pixel_values,
                image_grid_thw=torch.tensor(multimodal.image_grid_thw),
                mm_token_type_ids=(ids == IMAGE).int(),
            ).logits
        features = model.encode_images(multimodal.pixel_values, multimodal.image_grid_thw)
        actual = model(
            ids,
            mrope_positions=multimodal.mrope_positions[:, None, :],
            embedding_overrides=(torch.tensor(multimodal.image_token_positions), features),
        )
        # Conv3d vs. an equivalent F.linear patch embedding differ only in summation order.
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@unittest.skipUnless(Image is not None, "requires pillow")
class MultimodalSchedulerTest(unittest.TestCase):
    def _generate(self, model, requests, **config):
        scheduler = Scheduler(
            model,
            SchedulerConfig(max_running_requests=3, num_pages=64, page_size=4, **config),
        )
        for request in requests:
            scheduler.add(request)
        outputs = drain_scheduler(scheduler)
        scheduler.check_integrity()
        return scheduler, outputs

    @torch.inference_mode()
    def test_chunked_image_prefill_mixed_with_text_matches_oracle(self) -> None:
        model = make_vl(seed=7, mtp=True)
        first_tokens, first = image_prompt([random_image(24, 40, 3)])
        second_tokens, second = image_prompt(
            [random_image(16, 16, 4), random_image(32, 24, 5)], text=(9, 10)
        )
        text = [11, 12, 13, 14, 15]
        expected = {
            "first": multimodal_reference(model, first_tokens, first, 9),
            "second": multimodal_reference(model, second_tokens, second, 7),
            "text": reference_generate(model, text, 8),
        }
        for steps in (0, 2):
            with self.subTest(mtp_steps=steps):
                requests = [
                    Request.from_token_ids("first", first_tokens, SamplingParams(9), first),
                    Request.from_token_ids("second", second_tokens, SamplingParams(7), second),
                    Request.from_token_ids("text", text, SamplingParams(8)),
                ]
                # A 7-token budget splits every image across several prefill chunks.
                scheduler, outputs = self._generate(
                    model,
                    requests,
                    prefill_token_budget=7,
                    speculative_mtp_steps=steps,
                )
                self.assertEqual(outputs, expected)
                if steps:
                    self.assertGreater(scheduler.stats.speculative_verify_forwards, 0)

    @torch.inference_mode()
    def test_prefix_cache_keys_images_by_pixel_fingerprint(self) -> None:
        model = make_vl(seed=9)
        cat = random_image(48, 48, 21)
        car = random_image(48, 48, 22)  # 尺寸相同 → 占位 token 完全一样，只有像素不同
        prompts = {
            "cat": image_prompt([cat], text=(5, 6, 7)),
            "cat-again": image_prompt([cat], text=(8, 9)),
            "car": image_prompt([car], text=(5, 6, 7)),
        }
        self.assertEqual(prompts["cat"][0], prompts["car"][0])  # token id 一模一样
        self.assertNotEqual(
            prompts["cat"][1].cache_key(prompts["cat"][0]),
            prompts["car"][1].cache_key(prompts["car"][0]),
        )
        # 只截一段前缀（快照边界切在图片中间）时，钥匙也要能算，且是整段钥匙的前缀。
        tokens, multimodal = prompts["cat"]
        middle = multimodal.image_token_positions[len(multimodal.image_token_positions) // 2]
        self.assertEqual(
            multimodal.cache_key(tokens[:middle]), multimodal.cache_key(tokens)[:middle]
        )
        expected = {
            key: multimodal_reference(model, tokens, multimodal, 5)
            for key, (tokens, multimodal) in prompts.items()
        }
        scheduler = Scheduler(
            model,
            SchedulerConfig(
                max_running_requests=1,
                prefill_token_budget=64,
                num_pages=64,
                page_size=4,
                state_snapshots=4,
            ),
        )
        outputs, prefilled = {}, {}
        for key, (tokens, multimodal) in prompts.items():
            before = scheduler.prefill_input_tokens
            scheduler.add(Request.from_token_ids(key, tokens, SamplingParams(5), multimodal))
            outputs.update(drain_scheduler(scheduler))
            prefilled[key] = scheduler.prefill_input_tokens - before
        self.assertEqual(outputs, expected)
        cat_length = len(prompts["cat"][0])
        self.assertEqual(prefilled["cat"], cat_length)
        # 同一张图：命中到图片之后的整页边界，只重算后面一小段（含图片在内的前缀都复用了）。
        self.assertLess(prefilled["cat-again"], 8)
        # 不同的图：钥匙在第一个图片 token 处就分叉，只能复用图片之前的纯文字（不足一页 → 0）。
        self.assertEqual(prefilled["car"], cat_length)
        scheduler.check_integrity()

    @torch.inference_mode()
    def test_abort_during_image_prefill_releases_everything(self) -> None:
        model = make_vl(seed=8)
        tokens, multimodal = image_prompt([random_image(48, 48, 6)])
        scheduler = Scheduler(
            model, SchedulerConfig(max_running_requests=2, prefill_token_budget=5, num_pages=64)
        )
        scheduler.add(Request.from_token_ids("image", tokens, SamplingParams(4), multimodal))
        scheduler.step()
        self.assertIsNotNone(scheduler._entries["image"].vision_features)
        scheduler.abort("image")
        scheduler.check_integrity()
        self.assertEqual(scheduler.cache.recurrent.stats.free_slots, 2)

    def test_image_requests_need_a_vision_model(self) -> None:
        tokens, multimodal = image_prompt([random_image(16, 16, 7)])
        torch.manual_seed(0)
        text_only = Qwen3ForCausalLM(vl_config()).eval()
        scheduler = Scheduler(text_only, SchedulerConfig(num_pages=64))
        with self.assertRaisesRegex(ValueError, "enable_vision"):
            scheduler.add(Request.from_token_ids("x", tokens, SamplingParams(2), multimodal))

    def test_random_mixture_of_images_is_consistent(self) -> None:
        rng = random.Random(0)
        for seed in range(3):
            count = rng.randrange(1, 3)
            sizes = [(rng.randrange(8, 64), rng.randrange(8, 64)) for _ in range(count)]
            images = [random_image(h, w, seed * 10 + i) for i, (h, w) in enumerate(sizes)]
            tokens, multimodal = image_prompt(images)
            self.assertEqual(multimodal.mrope_positions.shape, (3, len(tokens)))
            self.assertEqual(
                multimodal.num_image_tokens,
                sum(t * h * w // 4 for t, h, w in multimodal.image_grid_thw),
            )


def _multimodal_tp_worker(rank, world_size, init_method, config, state, requests, expected):
    import os

    import torch.distributed as dist

    from mysglang import TensorParallelContext, TensorParallelScheduler, load_huggingface_state_dict

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
    try:
        model = Qwen3ForCausalLM(
            config,
            tensor_parallel=TensorParallelContext.from_distributed(),
            enable_mtp=True,
            enable_vision=True,
        ).eval()
        load_huggingface_state_dict(model, state)
        scheduler = TensorParallelScheduler(
            model,
            SchedulerConfig(
                max_running_requests=2,
                prefill_token_budget=6,
                num_pages=64,
                page_size=4,
                speculative_mtp_steps=2,
            ),
        )
        if rank != 0:
            scheduler.run_worker_loop()
            return
        for request in requests:
            scheduler.add(request)
        actual: dict[str, list[int]] = {}
        while scheduler.has_work:
            for event in scheduler.step().outputs:
                actual.setdefault(event.request_id, []).append(event.token_id)
            scheduler.check_integrity()
        assert actual == expected, actual
        scheduler.shutdown()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(Image is not None, "requires pillow")
class MultimodalTensorParallelTest(unittest.TestCase):
    def test_tp2_image_requests_match_tp1(self) -> None:
        import tempfile

        import torch.distributed as dist
        import torch.multiprocessing as mp

        if not (dist.is_available() and dist.is_gloo_available()):
            self.skipTest("requires torch.distributed with Gloo")
        model = make_vl(seed=21, mtp=True)
        tokens, multimodal = image_prompt([random_image(24, 32, 11)])
        with torch.inference_mode():
            expected = {
                "image": multimodal_reference(model, tokens, multimodal, 6),
                "text": reference_generate(model, [7, 8, 9], 5),
            }
        requests = [
            Request.from_token_ids("image", tokens, SamplingParams(6), multimodal),
            Request.from_token_ids("text", [7, 8, 9], SamplingParams(5)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _multimodal_tp_worker,
                args=(
                    2,
                    f"file://{directory}/pg",
                    model.config,
                    export_hf_state_dict(model),
                    requests,
                    expected,
                ),
                nprocs=2,
                join=True,
            )


class ContentPartTest(unittest.TestCase):
    def test_openai_parts_become_template_placeholders(self) -> None:
        from mysglang.serving.service import _split_images

        messages, images = _split_images(
            [
                {"role": "system", "content": "be brief"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
                        {"type": "text", "text": "what is "},
                        {"type": "image", "image": "/tmp/b.png"},
                        {"type": "text", "text": "this?"},
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "plain"}]},
            ]
        )
        self.assertEqual(images, ["data:image/png;base64,AA", "/tmp/b.png"])
        self.assertEqual(
            messages[1]["content"],
            [
                {"type": "image"},
                {"type": "text", "text": "what is "},
                {"type": "image"},
                {"type": "text", "text": "this?"},
            ],
        )
        # Text-only messages stay strings so text-only templates keep working.
        self.assertEqual(messages[2]["content"], "plain")
        with self.assertRaisesRegex(ValueError, "video"):
            _split_images([{"role": "user", "content": [{"type": "video", "video": "x"}]}])


@unittest.skipUnless(
    Image is not None
    and torch.cuda.is_available()
    and importlib.util.find_spec("flash_attn") is not None,
    "requires pillow, CUDA and flash-attn",
)
class MultimodalCudaGraphTest(unittest.TestCase):
    @torch.inference_mode()
    def test_graph_decode_uses_each_requests_rope_delta(self) -> None:
        from mysglang import FlashAttentionBackend

        torch.manual_seed(12)
        config = replace(vl_config(mtp=True), vocab_size=64, head_dim=32)
        model = Qwen3ForCausalLM(
            config,
            attention_backend=FlashAttentionBackend(),
            enable_mtp=True,
            enable_vision=True,
        )
        model = model.to(device="cuda", dtype=torch.bfloat16).eval()
        tokens, multimodal = image_prompt([random_image(40, 56, 9)])
        outputs = []
        for graph in (False, True):
            scheduler = Scheduler(
                model,
                SchedulerConfig(
                    max_running_requests=2,
                    prefill_token_budget=16,
                    num_pages=4,
                    page_size=256,
                    speculative_mtp_steps=2,
                    decode_cuda_graph_batch_sizes=(1, 2) if graph else (),
                ),
            )
            scheduler.add(Request.from_token_ids("image", tokens, SamplingParams(14), multimodal))
            scheduler.add(Request.from_token_ids("text", [3, 4, 5, 6], SamplingParams(10)))
            outputs.append(drain_scheduler(scheduler))
            if graph:
                self.assertGreater(scheduler.stats.cuda_graph_replays, 0)
            scheduler.close()
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
