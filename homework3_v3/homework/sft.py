from .base_llm import BaseLLM
from .data import Dataset, benchmark
import torch, random, numpy as np
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    #llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model = PeftModel.from_pretrained(llm.model, model_path, is_trainable=False).to(llm.device)

    llm.model.eval()

    return llm

def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full

def format_example(prompt: str, answer) -> dict[str, str]:
    # Format the answer exactly to 6 decimals.
    if isinstance(answer, float):
        answer = f"{answer:.6f}"
    else:
        answer = str(answer)
    # DO NOT add extra instruction in prompt, just the actual question!
    question = prompt
    answer = f"<answer>{answer}</answer>"
    return {"question": question, "answer": answer}


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)

def collate_fn(batch):
    # batch is a list of dicts, collate to dict of tensors
    out = {}
    for k in batch[0]:
        out[k] = torch.tensor([d[k] for d in batch], dtype=torch.long)
    return out

def train_model(
        output_dir: str,
        epochs: int = 1,
        batch_size: int = 8,
        lr: float = 2e-4,
        **kwargs,
):
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup
    from torch.optim import AdamW
    from peft import get_peft_model, LoraConfig, TaskType

    # Prepare datasets
    trainset = Dataset("train")
    valset = Dataset("valid")
    llm = BaseLLM()
    tokenizer = llm.tokenizer

    # --- Add debug prints here! ---
    print("Example train sample:", trainset[0])
    print("Example val sample:", valset[0])
    print("Formatted example:", format_example(*trainset[0]))
    print("Tokenized example:", tokenize(tokenizer, **format_example(*trainset[0])))
    # -----------------------------


    train_data = TokenizedDataset(tokenizer, trainset, format_example)
    val_data = TokenizedDataset(tokenizer, valset, format_example)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn)

    # Setup LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    llm.model = get_peft_model(llm.model, peft_config)

    llm.model.train()

    optimizer = AdamW(llm.model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    device = llm.device
    llm.model.to(device)

    for epoch in range(epochs):
        llm.model.train()
        total_loss = 0
        for batch in train_loader:
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
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Training loss: {total_loss / len(train_loader)}")

    # Save model (LoRA adapter only)
    llm.model.save_pretrained(output_dir)
    for i in range(5):
        prompt = format_example(*trainset[i])["question"]
        print("Train Q:", trainset[i][0])
        print("Expected:", trainset[i][1])
        print("Model output:", llm.generate(prompt))

    # Evaluate immediately after training
    test_model(output_dir)

def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")
    for i in range(5):
        sample_prompt = format_example(*testset[i])["question"]
        print("\nSample valset question:", testset[i][0])
        print("Expected answer:", testset[i][1])
        print("Model output:", llm.generate(sample_prompt))


if __name__ == "__main__":
    import torch
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
