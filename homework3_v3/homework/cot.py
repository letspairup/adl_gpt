from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Formats the question using the tokenizer's chat template for better LLM results.
        Falls back to a CoT instruction if not available.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            # Try using the chat template (recommended for modern HuggingFace LLMs)
            # 'messages' is a list of dicts as expected by HuggingFace chat template
            messages = [
                {"role": "user", "content": f"{question}\nPlease solve step by step, show your chain of thought, and put the final answer in <answer></answer> tags."}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return prompt
        else:
            # Fallback: add CoT hint and answer tags
            return (
                f"{question}\nPlease solve step by step, show your chain of thought, and put the final answer in <answer></answer> tags."
            )

def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
