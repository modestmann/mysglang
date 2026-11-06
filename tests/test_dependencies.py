"""基础文本路径不能被图片预处理的可选依赖阻断。"""

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class OptionalVisionDependenciesTest(unittest.TestCase):
    def run_without(self, modules: tuple[str, ...], code: str) -> None:
        # 新进程避免已导入模块掩盖依赖泄漏；同时拦截子模块导入。
        blocker = f"""
import importlib.abc
import sys

class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {modules!r}:
            raise ModuleNotFoundError(fullname, name=fullname)

sys.meta_path.insert(0, BlockImports())
"""
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(
            [sys.executable, "-c", blocker + textwrap.dedent(code)],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_text_generation_without_numpy_or_pillow(self) -> None:
        self.run_without(
            ("numpy", "PIL"),
            """
            import torch
            import mysglang
            from tests.helpers import make_model, reference_generate

            model = make_model()
            output = reference_generate(model, [1, 2, 3], max_new_tokens=3)
            assert len(output) == 3
            assert all(0 <= token < model.config.vocab_size for token in output)
        """,
        )

    def test_preprocessing_reports_missing_vision_extra(self) -> None:
        for module in ("numpy", "PIL"):
            with self.subTest(module=module):
                self.run_without(
                    (module,),
                    """
                    from mysglang.config import VisionConfig
                    from mysglang.multimodal import ImageProcessor

                    try:
                        ImageProcessor(VisionConfig()).preprocess(None)
                    except RuntimeError as exc:
                        assert 'vision' in str(exc), str(exc)
                    else:
                        raise AssertionError('missing image dependency was not reported')
                """,
                )


if __name__ == "__main__":
    unittest.main()
