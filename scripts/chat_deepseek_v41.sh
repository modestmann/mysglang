#!/usr/bin/env bash
# 复用现有 uv Python；只在 mysglang 内写入运行缓存，不激活 Conda。
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${MYSGLANG_PYTHON:-$repo_dir/../.venv-vllm/bin/python}"
model_dir="${MYSGLANG_DS_MODEL:-/spiritx/nfs_data/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/main}"
config_file="${MYSGLANG_DS_CONFIG:-$model_dir/inference/config.json}"
tp_size="${MYSGLANG_DS_TP:-8}"
cache_dir="$repo_dir/benchmark-results/deepseek-v41-chat"

if [[ ! -x "$python_bin" ]]; then
    echo "找不到 Python：$python_bin；请通过 MYSGLANG_PYTHON 指定已安装 deepseek extra 的环境。" >&2
    exit 1
fi
if [[ ! -r "$config_file" ]]; then
    echo "无法读取配置：$config_file；请通过 MYSGLANG_DS_CONFIG 指定可读配置。" >&2
    exit 1
fi

mkdir -p "$cache_dir"/{tmp,xdg,triton,cuda,torch-extensions}
cd -- "$repo_dir"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TMPDIR="$cache_dir/tmp"
export XDG_CACHE_HOME="$cache_dir/xdg"
export TRITON_CACHE_DIR="$cache_dir/triton"
export CUDA_CACHE_PATH="$cache_dir/cuda"
export TORCH_EXTENSIONS_DIR="$cache_dir/torch-extensions"

exec "$python_bin" -m torch.distributed.run --standalone --nproc-per-node="$tp_size" \
    -m mysglang.deepseek_cli --model "$model_dir" --config "$config_file" --interactive "$@"
