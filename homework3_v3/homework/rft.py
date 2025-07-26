from .base_llm import BaseLLM
from .sft import test_model
from .data import Dataset
from pathlib import Path
from peft import PeftModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
import random
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def load() -> BaseLLM:
    model_path = Path(__file__).parent / "rft_model"
    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    return llm

def compute_reward(pred: str, target: float) -> float:
    try:
        pred_val = float(pred.split("<answer>")[1].split("</answer>")[0])
        return -abs(pred_val - target)
    except:
        return -100.0

def train_model(output_dir: str, epochs=3, batch_size=4, lr=5e-5):
    train_data = Dataset("train")

    llm = BaseLLM()
    tokenizer = llm.tokenizer
    device = llm.device

    # Load SFT model with LoRA applied already
    sft_path = Path(__file__).parent / "sft_model"
    llm.model = PeftModel.from_pretrained(llm.model, sft_path).to(device)
    llm.model.train()

    optimizer = AdamW(llm.model.parameters(), lr=lr)
    total_steps = len(train_data) // batch_size * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    for epoch in range(epochs):
        total_reward = 0
        random.shuffle(train_data.data)

        for i in range(0, len(train_data), batch_size):
            batch = train_data.data[i : i + batch_size]
            loss = 0

            for question, true_answer in batch:
                inputs = tokenizer(question, return_tensors="pt").to(device)

                outputs = llm.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=100
                )
                pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
                reward = compute_reward(pred, true_answer)
                logprob_outputs = llm.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["input_ids"],
                    return_dict=True
                )
                logprob = logprob_outputs.loss

                if not logprob.requires_grad:
                    logprob.requires_grad_()
                loss += -reward * logprob
                total_reward += reward

            loss = loss / batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"Epoch {epoch + 1}: avg reward = {total_reward / len(train_data):.4f}")

    llm.model.save_pretrained(output_dir)
    test_model(output_dir)

if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})
