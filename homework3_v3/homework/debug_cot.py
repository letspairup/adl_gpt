# debug_cot.py
from homework3_v3.homework.cot import CoTModel

# Initialize the model (loads weights and tokenizer)
model = CoTModel()
for q in [
    "How many feet are there in 4 yards?",
    "Convert 3 yards to feet.",
    "How many ounces in 3 pounds?"
]:
    print(model.generate(q))