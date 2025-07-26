from .base_llm import BaseLLM
from pathlib import Path
from peft import PeftModel

class CoTModel(BaseLLM):
    def __init__(self):
        super().__init__()
        model_path = Path(__file__).parent / "sft_model"
        self.model = PeftModel.from_pretrained(self.model, model_path).to(self.device)
        self.model.eval()

    def format_prompt(self, question: str) -> str:
        # Use raw question — SFT model already trained on this format
        return question
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
