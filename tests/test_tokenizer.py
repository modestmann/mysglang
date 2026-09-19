import unittest
from pathlib import Path

from mysglang import HuggingFaceTokenizer, ModelConfig

MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "KuiperLLama" / "artifacts" / "qwen3-0.6b" / "hf-source"
)


@unittest.skipUnless(MODEL_PATH.is_dir(), "requires the local Qwen3 tokenizer artifact")
class HuggingFaceTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = HuggingFaceTokenizer.from_pretrained(MODEL_PATH)

    def test_real_config_and_chat_template(self) -> None:
        config = ModelConfig.from_pretrained(MODEL_PATH)
        self.assertEqual(config.model_type, "qwen3")
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.hidden_size, 1024)
        self.assertEqual(config.num_layers, 28)

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is AI?"}],
            enable_thinking=False,
        )
        self.assertEqual(
            input_ids,
            [
                151644,
                872,
                198,
                3838,
                374,
                15235,
                30,
                151645,
                198,
                151644,
                77091,
                198,
                151667,
                271,
                151668,
                271,
            ],
        )

    def test_incremental_decode_matches_full_decode(self) -> None:
        token_ids = self.tokenizer.encode("Hello, 世界! This is Qwen3.")
        decoder = self.tokenizer.new_incremental_decoder()
        pieces = [
            decoder.decode(token_id, final=index == len(token_ids) - 1)
            for index, token_id in enumerate(token_ids)
        ]
        self.assertEqual("".join(pieces), self.tokenizer.decode(token_ids))

    def test_incremental_decode_keeps_cjk_prefix_before_latin_word(self) -> None:
        token_ids = [104198, 15469]  # ``我是`` followed by ``AI`` without a space.
        decoder = self.tokenizer.new_incremental_decoder()
        pieces = [
            decoder.decode(token_id, final=index == len(token_ids) - 1)
            for index, token_id in enumerate(token_ids)
        ]
        self.assertEqual("".join(pieces), "我是AI")


if __name__ == "__main__":
    unittest.main()
