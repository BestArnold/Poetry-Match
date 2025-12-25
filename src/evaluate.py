import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForMultipleChoice
from tqdm import tqdm
import json

from config import Config
from dataset import PoetryDataset
from utils import compute_accuracy


def evaluate():
    """
    Evaluates the best saved model on the test dataset.
    Computes and prints the accuracy.
    """
    config = Config()
    device = config.device

    # 加载训练好的最佳模型
    print(f"Loading best model from {config.save_path}...")
    try:
        tokenizer = BertTokenizer.from_pretrained(config.save_path)
        model = BertForMultipleChoice.from_pretrained(config.save_path)
    except OSError:
        print("未找到保存的模型，请先运行 train.py")
        return

    model.to(device)
    model.eval()

    # 加载测试集
    test_dataset = PoetryDataset(config.test_path, tokenizer, config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    total_correct = 0
    total_samples = 0

    # 如果你需要保存错误样本进行分析
    error_log = []

    print("Starting Evaluation on Test Set...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            token_type_ids = batch.get('token_type_ids').to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)



    acc = total_correct / total_samples
    print(f"\nTest Set Accuracy: {acc:.4f}")


if __name__ == "__main__":
    evaluate()