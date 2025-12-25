import json
import torch
from torch.utils.data import Dataset


class PoetryDataset(Dataset):
    """
    Custom Dataset for the Poetry Matching task.
    Reads data from a JSONL file and processes it for the Multiple Choice model.
    """
    def __init__(self, file_path, tokenizer, config):
        """
        Args:
            file_path (str): Path to the jsonl data file.
            tokenizer (PreTrainedTokenizer): Tokenizer to process text.
            config (Config): Configuration object containing max_len.
        """
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_len = config.max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a single sample processed for the model.
        
        Returns:
            dict: {
                'input_ids': tensor,
                'attention_mask': tensor,
                'token_type_ids': tensor,
                'labels': tensor
            }
        """
        item = self.data[index]
        translation = item['translation']
        choices = item['choices']  # list of 4 strings
        label = int(item['answer'])

        # 构造输入对： [CLS] translation [SEP] choice_i [SEP]
        # 我们需要构建4个这样的对
        first_sentences = [translation] * 4
        second_sentences = choices

        # 使用 tokenizer 批量处理这4对
        encoding = self.tokenizer(
            first_sentences,
            second_sentences,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        # tokenizer 返回的形状是 (4, seq_len)
        # 这正是 BertForMultipleChoice 需要的单个样本格式

        return {
            'input_ids': encoding['input_ids'],  # shape: (4, max_len)
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding['token_type_ids'],
            'labels': torch.tensor(label, dtype=torch.long)
        }