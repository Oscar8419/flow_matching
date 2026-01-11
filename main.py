import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
from datetime import datetime
import os
from config import CONFIG, DEVICE
from signal_gen import RFSignalDataset
from fm_models import RFSignalDiT
from utils import train


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    # 配置 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler("training.log", mode='a', encoding='utf-8'),
            # logging.StreamHandler()
        ]
    )
    logging.info("="*30)
    logging.info("Starting main process...")

    # 创建带有时间戳的检查点目录，防止覆盖
    timestamp = datetime.now().strftime("%m-%d--%H-%M")
    run_checkpoint_dir = os.path.join("checkpoints", timestamp)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    logging.info(
        f"Checkpoints for this run will be saved to: {run_checkpoint_dir}")

    dataset = RFSignalDataset(
        length=CONFIG["num_samples"], signal_len=CONFIG["signal_length"])
    dataloader = DataLoader(
        dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    logging.info("Dataset and DataLoader initialized.")

    model = RFSignalDiT().to(DEVICE)
    if CONFIG['fm_model_path']:
        model.load_state_dict(torch.load(CONFIG['fm_model_path'], map_location=DEVICE))
        logging.info(f"load checkpoint from {CONFIG['fm_model_path']}")
    logging.info("Model initialized.")
    train(model, dataloader, run_checkpoint_dir)


if __name__ == "__main__":
    main()
