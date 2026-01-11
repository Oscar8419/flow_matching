import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "learning_rate": 1e-4,
    "batch_size": 64*8,  # 每个 batch 的样本数量
    "num_samples": 5e6,  # 总样本数量
    "signal_length": 1024,  # 每个信号的长度
    "num_workers": 8,  # DataLoader 的 num_workers
    # "save_interval": 2e5,  # 每隔多少个 samples 保存一次模型, data are generated on the fly
    "times_log":20, # 每次训练报告 x 次损失日志
    "times_save":5, # 每次训练保存 x 次模型参数
    "fm_model_path": "/root/code/flow_matching/checkpoints/01-11--16-21/model_final.pth",

    # model hyperparameters
    "patch_size": 16,
    "hidden_size": 256,
    "depth": 6,
    "num_heads": 8,
    "num_classes": 10,  # not used in unconditional model


}
