from transformers import BertForMultipleChoice

def build_model(model_name):
    """
    Builds and returns the BertForMultipleChoice model.
    
    Args:
        model_name (str): Name of the pre-trained model.
        
    Returns:
        BertForMultipleChoice: The instantiated model.
    """
    # 自动下载并加载预训练权重
    # BertForMultipleChoice 会在 BERT 输出层后加一个线性层把 hidden_size 映射到 1
    model = BertForMultipleChoice.from_pretrained(model_name)
    return model