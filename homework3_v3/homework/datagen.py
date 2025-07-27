import random
import json
import os
from tqdm import tqdm

from .data import Dataset
from .cot import CoTModel
from fire import Fire

def generate_dataset(output_json: str = 'rft.json', oversample: int = 10, temperature: float = 0.7):
    import json
    from tqdm import tqdm
    from .cot import CoTModel
    from .data import Dataset, is_answer_valid
    from pathlib import Path

    model = CoTModel()
    dataset = Dataset("train")
    results = []
    success_count = 0

    for i, (question, correct_answer) in enumerate(tqdm(dataset.data, desc="Generating RFT dataset")):
        generations = model.batched_generate(
            [model.format_prompt(question)],
            num_return_sequences=oversample,
            temperature=temperature,
        )[0]  # List of strings

        for gen in generations:
            if is_answer_valid(model.parse_answer(gen), correct_answer):
                results.append([question, correct_answer, gen])
                success_count += 1
                break  # Only keep the first correct one

        if (i + 1) % 100 == 0:
            print(f"⏳ Processed {i + 1} samples, Collected {success_count} valid examples ---")

    print(f"Generated {len(results)} RFT samples out of {len(dataset)}")
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    from fire import Fire
    Fire(generate_dataset)

