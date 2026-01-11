import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "learning_rate": 1e-4,
    "batch_size": 64,  # 每个 batch 的样本数量
    "num_samples": 1e6,  # 总样本数量
    "signal_length": 1024,  # 每个信号的长度
    "num_workers": 8,  # DataLoader 的 num_workers
    "save_interval": 1e4,  # 每隔多少个 samples 保存一次模型, data are generated on the fly
    "fm_model_path": None,

    # model hyperparameters
    "patch_size": 16,
    "hidden_size": 256,
    "depth": 6,
    "num_heads": 8,
    "num_classes": 10,  # not used in unconditional model


}
