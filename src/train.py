from transformers import Trainer, TrainingArguments
import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_dataset, eval_dataset, output_dir, compute_metrics):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    device = get_device()
    model.to(device)

    trainer.train()
    return trainer
