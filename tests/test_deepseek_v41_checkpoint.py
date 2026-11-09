"""损坏/不完整权重必须在分配模型前被拒绝，内存核算必须含复制和量化元数据。"""

import io
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mysglang.modeling.deepseek_v41 import DeepSeekV41Config
from mysglang.modeling.deepseek_v41.checkpoint import (
    TensorHeader,
    audit_checkpoint,
    expected_backbone_shapes,
    rank_storage,
    read_checkpoint_headers,
    read_shard_header,
    validate_quantization,
)


def write_checkpoint(root: Path):
    config = DeepSeekV41Config.tiny()
    metadata, offset = {}, 0
    for name, shape in expected_backbone_shapes(config).items():
        size = math.prod(shape) * 4
        metadata[name] = {
            "shape": list(shape),
            "dtype": "F32",
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(metadata).encode()
    payload = struct.pack("<Q", len(encoded)) + encoded
    shard = root / "model.safetensors"
    shard.write_bytes(payload + b"\xff" * offset)
    index = {
        "metadata": {"total_size": offset},
        "weight_map": {name: shard.name for name in metadata},
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    return config, shard, payload, index


class DeepSeekCheckpointTest(unittest.TestCase):
    def test_header_audit_reads_no_weight_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            config, shard, header, _ = write_checkpoint(Path(directory))
            real_open = Path.open

            class HeaderOnly(io.BytesIO):
                def read(self, size=-1):
                    if size < 0 or self.tell() + size > len(header):
                        raise AssertionError("attempted to read tensor data")
                    return super().read(size)

            def open_header(path, *args, **kwargs):
                return HeaderOnly(header) if path == shard else real_open(path, *args, **kwargs)

            with patch.object(Path, "open", open_header):
                result = audit_checkpoint(directory, config, tp_size=2)
            self.assertEqual(result["header_bytes_read"], len(header))
            self.assertEqual(result["tensor_count"], len(expected_backbone_shapes(config)))
            self.assertGreater(
                sum(result["estimated_rank_weight_bytes"]), result["checkpoint_bytes"]
            )

    def test_missing_or_misplaced_index_tensor_is_rejected(self):
        for change in ("missing", "misplaced", "wrong_total"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                _, _, _, index = write_checkpoint(Path(directory))
                if change == "missing":
                    index["weight_map"]["missing.weight"] = "model.safetensors"
                elif change == "misplaced":
                    index["weight_map"].pop("embed.weight")
                else:
                    index["metadata"]["total_size"] += 1
                (Path(directory) / "model.safetensors.index.json").write_text(json.dumps(index))
                with self.assertRaises(ValueError):
                    read_checkpoint_headers(directory)

    def test_missing_backbone_tensor_is_rejected_even_when_index_agrees(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _, _ = write_checkpoint(Path(directory))
            data, report = read_checkpoint_headers(directory)
            data.pop("layers.1.attn.compressor.wgate.weight")
            with patch(
                "mysglang.modeling.deepseek_v41.checkpoint.read_checkpoint_headers",
                return_value=(data, report),
            ):
                with self.assertRaisesRegex(ValueError, "missing backbone"):
                    audit_checkpoint(directory, config, tp_size=2)

    def test_truncation_byte_mismatch_and_overlap_fail(self):
        for metadata, body in (
            ({"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}, b"0" * 4),
            (
                {
                    "x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                    "y": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                },
                b"0" * 4,
            ),
            ({"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, b"0" * 4),
        ):
            with tempfile.TemporaryDirectory() as directory:
                raw = json.dumps(metadata).encode()
                path = Path(directory) / "bad.safetensors"
                path.write_bytes(struct.pack("<Q", len(raw)) + raw + body)
                with self.assertRaises(ValueError):
                    read_shard_header(path)

    def test_duplicate_header_names_and_path_escape_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = b'{"x":{},"x":{}}'
            path = root / "bad.safetensors"
            path.write_bytes(struct.pack("<Q", len(raw)) + raw)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_shard_header(path)
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {"x": "../outside.safetensors"},
                        "metadata": {"total_size": 0},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "inside checkpoint"):
                read_checkpoint_headers(root)

    def test_fp4_scale_uses_unpacked_k_dimension(self):
        name = "layers.0.ffn.experts.0.w1.weight"
        scale_name = name.replace(".weight", ".scale")
        weight = TensorHeader(name, "a", "I8", (64, 32), 2048)
        scale = TensorHeader(scale_name, "a", "F8_E8M0", (64, 2), 128)
        validate_quantization({name: weight, scale_name: scale})
        bad = TensorHeader(scale_name, "a", "F8_E8M0", (2, 2), 4)
        with self.assertRaisesRegex(ValueError, "scale layout"):
            validate_quantization({name: weight, scale_name: bad})
        with self.assertRaisesRegex(ValueError, "missing quantization scale"):
            validate_quantization({name: weight})

    def test_engram_uses_per_row_scales_instead_of_dense_32_by_32_blocks(self):
        name = "layers.1.engram.embed.weight"
        scale_name = name.replace(".weight", ".scale")
        weight = TensorHeader(name, "a", "F8_E4M3", (67, 256), 67 * 256)
        scale = TensorHeader(scale_name, "a", "F8_E8M0", (67, 8), 67 * 8)
        validate_quantization({name: weight, scale_name: scale})
        dense_scale = TensorHeader(scale_name, "a", "F8_E8M0", (3, 8), 24)
        with self.assertRaisesRegex(ValueError, "scale layout"):
            validate_quantization({name: weight, scale_name: dense_scale})
        wrong_dtype = TensorHeader(scale_name, "a", "F32", (67, 8), 67 * 8 * 4)
        with self.assertRaisesRegex(ValueError, "UE8M0"):
            validate_quantization({name: weight, scale_name: wrong_dtype})

    def test_engram_sharding_pads_the_last_rank_including_scales(self):
        config = DeepSeekV41Config.tiny()
        table = TensorHeader("layers.1.engram.embed.weight", "a", "F8_E4M3", (67, 256), 67 * 256)
        scale = TensorHeader("layers.1.engram.embed.scale", "a", "F8_E8M0", (67, 8), 67 * 8)
        self.assertEqual(rank_storage(table, config, 4)[0], [17 * 256] * 4)
        self.assertEqual(rank_storage(scale, config, 4)[0], [17 * 8] * 4)

    def test_experts_assigned_once_and_unsharded_parameters_replicated(self):
        config = DeepSeekV41Config.tiny()
        totals = [0, 0]
        for expert in range(4):
            item = TensorHeader(
                f"layers.0.ffn.experts.{expert}.w1.weight", "a", "I8", (64, 32), 2048
            )
            sizes, _ = rank_storage(item, config, 2)
            self.assertEqual(sum(sizes), item.nbytes)
            totals = [a + b for a, b in zip(totals, sizes, strict=True)]
        self.assertEqual(totals, [4096, 4096])
        norm = TensorHeader("layers.0.attn_norm.weight", "a", "BF16", (32,), 64)
        self.assertEqual(rank_storage(norm, config, 2)[0], [64, 64])

    def test_wo_a_memory_accounts_for_bf16_conversion_and_removed_scale(self):
        config = DeepSeekV41Config.tiny()
        weight = TensorHeader("layers.0.attn.wo_a.weight", "a", "F8_E4M3", (64, 32), 2048)
        scale = TensorHeader("layers.0.attn.wo_a.scale", "a", "F8_E8M0", (2, 1), 2)
        self.assertEqual(rank_storage(weight, config, 2)[0], [2048, 2048])
        self.assertEqual(rank_storage(scale, config, 2)[0], [0, 0])
        bad = TensorHeader("layers.0.attn.wq_b.scale", "a", "F8_E8M0", (3, 1), 3)
        with self.assertRaisesRegex(ValueError, "cannot be divided"):
            rank_storage(bad, config, 2)


if __name__ == "__main__":
    unittest.main()
