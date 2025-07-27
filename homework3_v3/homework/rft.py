from .sft import TokenizedDataset
from .base_llm import BaseLLM
from .data import benchmark, Dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import TrainingArguments, Trainer
from pathlib import Path

def load() -> BaseLLM:
    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    return llm

def format_example(prompt: str, answer: float, reasoning: str) -> dict[str, str]:
    """
    Construct a question + reasoning / answer pair. Answer must be wrapped in <answer>...</answer>.
    """
    return {
        "question": prompt,
        "answer": reasoning  # reasoning already includes the <answer>...</answer> tag
    }

def train_model(output_dir: str, **kwargs):
    # Load dataset
    dataset = Dataset("rft")

    # Format and tokenize
    llm = BaseLLM()
    format_fn = lambda q, a, r: format_example(q, a, r)
    tokenized_dataset = TokenizedDataset(llm.tokenizer, dataset, format_fn)

    # LoRA config
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(llm.model, config)
    model.enable_input_require_grads()

    # Training args
    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        learning_rate=5e-3,
        gradient_checkpointing=True,
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=1,
        logging_strategy="epoch"
        #bf16=True
        #fp16=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    model.save_pretrained(Path(__file__).parent / "rft_model")
    test_model(output_dir)

def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    from peft import PeftModel
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")

if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})
