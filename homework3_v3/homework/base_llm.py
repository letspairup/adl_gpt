from typing import overload
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        You don't need to change this function for now.
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        max_new_tokens = 100
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            outputs = self.model(input_ids=generated, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            generated = torch.cat([generated, next_token], dim=-1)

            # Update attention mask
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)], dim=-1
            )

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def batched_generate(
            self, prompts: list[str], num_return_sequences: int = 1, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """Batched version of generation."""
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=temperature > 0,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id
        )

        if num_return_sequences == 1:
            return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        else:
            batch_size = len(prompts)
            return [
                self.tokenizer.batch_decode(
                    output_ids[i * num_return_sequences: (i + 1) * num_return_sequences],
                    skip_special_tokens=True
                )
                for i in range(batch_size)
            ]

    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire
    Fire({"test": test_model})
