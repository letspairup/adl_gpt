import random
import json
from tqdm import tqdm

from homework3_v3.homework import Dataset
from homework3_v3.homework.cot import CoTModel

# Supported units and their conversion logic
CONVERSIONS = {
    ("hour", "minute"): lambda x: x * 60,
    ("hour", "second"): lambda x: x * 3600,
    ("minute", "second"): lambda x: x * 60,
    ("day", "hour"): lambda x: x * 24,
    ("week", "day"): lambda x: x * 7,
    ("year", "week"): lambda x: x * 52.1775,
    ("century", "year"): lambda x: x * 100,
    ("kg", "gram"): lambda x: x * 1000,
    ("kg", "pound"): lambda x: x * 2.20462,
    ("GB", "MB"): lambda x: x * 1000,
    ("MB", "kB"): lambda x: x * 1000,
    ("kB", "bit"): lambda x: x * 8192,
    ("liter", "ml"): lambda x: x * 1000,
    ("gallon", "liter"): lambda x: x * 3.78541,
    ("quart", "pint"): lambda x: x * 2,
}

PHRASINGS = [
    "Convert {qty} {from_unit} to {to_unit}.",
    "How many {to_unit} are in {qty} {from_unit}?",
    "Please convert {qty} {from_unit} into {to_unit}.",
    "{qty} {from_unit} equals how many {to_unit}?",
    "What is {qty} {from_unit} in {to_unit}?"
]

def make_example():
    (from_unit, to_unit), func = random.choice(list(CONVERSIONS.items()))
    qty = round(random.uniform(1, 500), 2)
    prompt_template = random.choice(PHRASINGS)
    question = prompt_template.format(qty=qty, from_unit=from_unit, to_unit=to_unit)
    answer_val = func(qty)
    answer_str = f"<answer>{answer_val:.6f}</answer>"
    return question, answer_str

def generate_dataset_basic(n=1000):
    return [make_example() for _ in range(n)]

def generate_dataset(output_file="data/rft.json", num_samples_per_q=10):
    data = Dataset("train")
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
                    if abs(pred - float(expected)) < 1e-2:
                        final_data.append([question, expected, g])
                        found = True
                        break
                except:
                    continue
        if not found:
            continue

    with open(output_file, "w") as f:
        json.dump(final_data, f, indent=2)
    print(f"Saved {len(final_data)} examples to {output_file}")

if __name__ == "__main__":
    for i in range(10):
        q, a = make_example()
        print(f"Q{i+1}: {q}\nA{i+1}: {a}\n")
