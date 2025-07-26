import random
import json
import os
from tqdm import tqdm

from .data import Dataset
from .cot import CoTModel
from fire import Fire


def generate_dataset(output_file="data/rft.json", num_samples_per_q=10, max_examples=1000):
    data = Dataset("train")[:max_examples]
    print(f"Loaded {len(data)} examples from Dataset('train')")
    cot_model = CoTModel()
    final_data = []

    for question, expected in tqdm(data, desc="Generating RFT dataset"):
        generations = cot_model.batched_generate(
            [question] * num_samples_per_q,
            num_return_sequences=num_samples_per_q,
            temperature=0.8,
            )

        found = False
        for g in generations:
            if "<answer>" in g and "</answer>" in g:
                try:
                    answer_str = g.split("<answer>")[1].split("</answer>")[0]
                    pred = float(answer_str)

                    expected_rounded = round(float(expected), 2)
                    pred_rounded = round(pred, 2)

                    if abs(pred_rounded - expected_rounded) < 0.05:
                        # Extract only the reasoning part
                        lines = g.strip().splitlines()
                        reasoning_lines = [
                            line.strip() for line in lines
                            if ("<answer>" in line) or ("=" in line) or ("*" in line)
                        ]
                        reasoning = " ".join(reasoning_lines).strip()

                        # Fallback to entire generation if no reasoning found
                        if not reasoning:
                            reasoning = g.strip()

                        # Ensure answer is present
                        if "<answer>" not in reasoning:
                            reasoning += f" <answer>{pred:.6f}</answer>"

                        final_data.append([question, expected, reasoning])
                        print("✓ Reasoning kept:", reasoning)
                        found = True
                        break
                except Exception as e:
                    print(f"Error parsing generation: {e}")
                    continue
        if not found:
            print(f"✗ No valid generation for: {question}")
            continue

    if not final_data:
        print("No data to save. Exiting...")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        print(f"\nSaving {len(final_data)} examples to {output_file}...")
        json.dump(final_data, f, indent=2)
        print("✅ Saved!")


def cli(max_examples=1, output_file="data/rft.json", num_samples_per_q=1):
    generate_dataset(output_file=output_file, num_samples_per_q=num_samples_per_q, max_examples=max_examples)


if __name__ == "__main__":
    Fire(cli)
