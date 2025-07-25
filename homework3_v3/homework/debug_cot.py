# debug_cot.py
from homework3_v3.homework.cot import CoTModel

# Initialize the model (loads weights and tokenizer)
model = CoTModel()

# Your test question (should be a CoT-style prompt)
question = "If 5 liters of paint covers 40 square meters, how much area can 8 liters cover? Show steps."

# Generate output
output = model.generate(question)

print("QUESTION:", question)
print("MODEL OUTPUT:", output)
