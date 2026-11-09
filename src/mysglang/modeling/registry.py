"""模型配置分派；参考模型入口与目前仅支持 Qwen 的服务加载入口分别维护。"""

from __future__ import annotations

import json
from pathlib import Path

from mysglang.config import ModelConfig

from .deepseek_v41.config import DeepSeekV41Config


def load_model_config(path: str | Path) -> ModelConfig | DeepSeekV41Config:
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    raw = json.loads(path.read_text())
    kind = raw.get("model_type")
    if kind in {"deepseek_v41", "deepseek_v41_text"} or (
        kind is None and "kv_source_layers" in raw and "hc_mult" in raw
    ):
        return DeepSeekV41Config.from_dict(raw)
    return ModelConfig.from_pretrained(path)


def build_reference_model(config: ModelConfig | DeepSeekV41Config, **kwargs):
    if isinstance(config, DeepSeekV41Config):
        from .deepseek_v41 import DeepSeekV41ReferenceModel

        return DeepSeekV41ReferenceModel(config, **kwargs)
    if isinstance(config, ModelConfig):
        from .qwen3 import Qwen3ForCausalLM

        return Qwen3ForCausalLM(config, **kwargs)
    raise TypeError(f"unsupported model config: {type(config).__name__}")
