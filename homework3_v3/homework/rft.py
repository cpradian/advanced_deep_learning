import json
from pathlib import Path

from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


class RFTDataset:
    def __init__(self, tokenizer, json_path: str):
        from .sft import tokenize

        self.tokenizer = tokenizer
        self.tokenize_fn = tokenize

        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, _correct_answer, reasoning = self.data[idx]
        return self.tokenize_fn(self.tokenizer, question, reasoning)


def train_model(
    output_dir: str | None = None,
    **kwargs,
):
    from peft import LoraConfig, get_peft_model
    from transformers import DefaultDataCollator, Trainer, TrainingArguments

    if output_dir is None:
        output_dir = str(Path(__file__).parent / "rft_model")

    rft_json = Path(__file__).parent.parent / "data" / "rft.json"

    llm = BaseLLM()

    # Slightly larger LoRA than SFT is reasonable for RFT
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    llm.model = get_peft_model(llm.model, lora_config)

    if llm.device == "cuda":
        llm.model.enable_input_require_grads()

    trainset = RFTDataset(llm.tokenizer, str(rft_json))

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        per_device_train_batch_size=32,
        num_train_epochs=5,
        learning_rate=2e-4,
        gradient_checkpointing=True,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=trainset,
        data_collator=DefaultDataCollator(),
    )

    trainer.train()
    trainer.save_model(output_dir)
    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})