import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "times_log": 20,  # 每次训练报告 x 次损失日志
    "times_save": 5,  # 每次训练保存 x 次模型参数
    # Checkpoint paths
    "fm_model_path": None,
    "classifier_model_path": None,
    # Classifier training hyperparameters
    "classifier_model": "GRUformer",  # 可选: "GRU", "GRUformer"
    "classifier_num_samples": 4e6,  # 分类训练总样本数量
    "classifier_snr_range": (-10, 20),  # 分类器训练时的 SNR 范围 (dB)

    # FM training hyperparameters
    "learning_rate": 1e-4,
    "batch_size": 64*8,  # 每个 batch 的样本数量
    "num_samples": 3e6,  # 总样本数量
    "signal_length": 1024,  # 每个信号的长度
    "num_workers": 8,  # DataLoader 的 num_workers

    # model hyperparameters
    "patch_size": 16,
    "hidden_size": 256*2,
    "depth": 10,
    "num_heads": 8,
    "num_classes": 10,  # not used in unconditional model


}
