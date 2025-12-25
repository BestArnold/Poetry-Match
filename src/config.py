import torch


class Config:
    """
    Configuration class for the PoetryMatch project.
    
    Attributes:
        train_path (str): Path to the training dataset.
        dev_path (str): Path to the validation dataset.
        test_path (str): Path to the test dataset.
        save_path (str): Directory to save the best model checkpoints.
        model_name (str): Name or path of the pre-trained model to use (e.g., 'bert-base-chinese').
        num_choices (int): Number of candidate choices for each question.
        max_len (int): Maximum sequence length for tokenization.
        device (torch.device): Computing device (CPU or CUDA).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training and evaluation.
        learning_rate (float): Learning rate for the optimizer.
        seed (int): Random seed for reproducibility.
        adam_epsilon (float): Epsilon value for AdamW optimizer.
    """
    def __init__(self):
        # 路径设置
        self.train_path = './datasets/train.jsonl'
        self.dev_path = './datasets/val.jsonl'
        self.test_path = './datasets/test.jsonl'
        self.save_path = './models/saved_best/'

        # 模型参数
        # 建议：后期可以尝试将 'bert-base-chinese' 换成 'ethanyt/guwenbert-base'
        self.model_name = 'bert-base-chinese'
        self.num_choices = 4
        self.max_len = 128

        # 训练参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = 4
        self.batch_size = 8  # 如果显存大(>=16G)，可以设为16或32
        self.learning_rate = 2e-5
        self.seed = 42
        self.adam_epsilon = 1e-8