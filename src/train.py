import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer,  get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import os

# 导入自定义模块
from config import Config
from dataset import PoetryDataset
from model import build_model
from utils import set_seed, compute_accuracy


def train():
    """
    Main training loop.
    Initializes model, optimizer, scheduler, and runs the training and validation loops.
    Saves the best model based on validation accuracy.
    """
    # 1. 初始化配置和环境
    config = Config()
    set_seed(config.seed)

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    print(f"Loading Tokenizer and Model: {config.model_name}...")
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    model = build_model(config.model_name)
    model.to(config.device)

    # 2. 准备数据
    print("Loading Data...")
    train_dataset = PoetryDataset(config.train_path, tokenizer, config)
    valid_dataset = PoetryDataset(config.dev_path, tokenizer, config)  # 确保你有验证集

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    # 3. 优化器与调度器
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, eps=config.adam_epsilon)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # 4. 训练循环
    best_acc = 0.0

    for epoch in range(config.epochs):
        print(f"\n======== Epoch {epoch + 1} / {config.epochs} ========")

        # --- Training ---
        model.train()
        total_train_loss = 0
        train_loop = tqdm(train_loader, desc="Training")

        for batch in train_loop:
            # 将数据搬到 GPU
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            # 如果有 token_type_ids 也需要搬运
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(config.device)

            model.zero_grad()

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪防止爆炸
            optimizer.step()
            scheduler.step()

            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"  Average training loss: {avg_train_loss:.4f}")

        # --- Validation ---
        print("Running Validation...")
        model.eval()
        val_accuracy = 0
        total_examples = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['labels'].to(config.device)
                token_type_ids = batch.get('token_type_ids').to(config.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

                logits = outputs.logits
                val_accuracy += compute_accuracy(logits, labels)
                total_examples += labels.size(0)

        avg_val_acc = val_accuracy / total_examples
        print(f"  Validation Accuracy: {avg_val_acc:.4f}")

        # 保存最好的模型
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            print(f"  New Best Accuracy! Saving model to {config.save_path}")
            model.save_pretrained(config.save_path)
            tokenizer.save_pretrained(config.save_path)

    print("Training complete!")


if __name__ == "__main__":
    train()