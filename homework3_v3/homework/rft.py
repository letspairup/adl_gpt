from .base_llm import BaseLLM
from .data import benchmark
from torch.utils.data import DataLoader
import torch, json
from peft import get_peft_model, LoraConfig, TaskType
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from .data import Dataset as Dataset


def load():
    from pathlib import Path
    from peft import PeftModel
    model_path = Path(__file__).parent / "rft_model"
    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    return llm


def format_entry(entry):
    question, _, full_reasoning = entry
    return {"question": question, "answer": full_reasoning}


def tokenize(tokenizer, question: str, answer: str):
    full_text = f"{question} {answer}"
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)
    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])
    labels = [-100] * question_len + input_ids[question_len:]
    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100
    full["labels"] = labels
    return full


def collate_fn(batch):
    out = {}
    for k in batch[0]:
        out[k] = torch.tensor([d[k] for d in batch], dtype=torch.long)
    return out


def train_model(output_dir: str, epochs=3, batch_size=8, lr=2e-4, **kwargs):
    tokenizer = BaseLLM().tokenizer
    llm = BaseLLM()
    with open("data/rft.json") as f:
        data = json.load(f)

    dataset = [tokenize(tokenizer, **format_entry(entry)) for entry in data]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
    )
    llm.model = get_peft_model(llm.model, peft_config)
    llm.model.train()

    optimizer = AdamW(llm.model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(dataloader) * epochs)

    device = llm.device
    llm.model.to(device)

    for epoch in range(epochs):
        total_reward = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = llm.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_reward += -loss.item()  # Treat negative loss as reward
        print(f"Epoch {epoch + 1}: avg reward = {total_reward / len(dataloader):.4f}")

    llm.model.save_pretrained(output_dir)
    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()
    from peft import PeftModel
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    result = benchmark(llm, testset, 100)
    print(f"{result.accuracy=}  {result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})
