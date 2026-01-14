import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
from datetime import datetime
import os
from config import CONFIG, DEVICE
from signal_gen import RFSignalDataset
from fm_models import RFSignalDiT
from utils import get_classifier_model, train_fm, train_classifi
from models import *
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", action="store_true", default=False,
                        help="Train Flow Matching model (Default: False)")
    parser.add_argument("-c", action="store_true", default=False,
                        help="Train Classifier model (Default: False)")
    args = parser.parse_args()
    assert args.f or args.c, "At least one of -f (Flow Matching) or -c (Classifier) must be specified."

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
    logging.info(
        f"Starting main process... Arguments: FM={args.f}, CLS={args.c}")

    # 创建带有时间戳的检查点目录，防止覆盖
    timestamp = datetime.now().strftime("%m-%d--%H-%M")
    run_checkpoint_dir = os.path.join("checkpoints", timestamp)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    logging.info(
        f"Checkpoints for this run will be saved to: {run_checkpoint_dir}")

    if args.f:
        logging.info(">>> Start Training Flow Matching Model")
        # Flow Matching 使用干净数据 (snr_range=None)
        dataset_fm = RFSignalDataset(
            num_samples=CONFIG["num_samples"],
            signal_len=CONFIG["signal_length"],
            snr_range=None
        )
        dataloader_fm = DataLoader(
            dataset_fm,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"]
        )
        logging.info("Clean Dataset and DataLoader initialized for FM.")

        model_fm = RFSignalDiT().to(DEVICE)
        if CONFIG['fm_model_path']:
            model_fm.load_state_dict(torch.load(
                CONFIG['fm_model_path'], map_location=DEVICE))
            logging.info(f"load checkpoint from {CONFIG['fm_model_path']}")
        logging.info("Model_fm initialized.")
        train_fm(model_fm, dataloader_fm, run_checkpoint_dir)
    else:
        logging.info(">>> Skip Flow Matching Training")

    if args.c:
        logging.info(">>> Start Training Classifier Model")
        # Classifier 使用噪声数据 (snr_range from CONFIG)
        dataset_cls = RFSignalDataset(
            num_samples=CONFIG["classifier_num_samples"],
            signal_len=CONFIG["signal_length"],
            snr_range=CONFIG["classifier_snr_range"]
        )
        dataloader_cls = DataLoader(
            dataset_cls,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"]
        )
        logging.info(
            "Noisy Dataset and DataLoader initialized for Classifier.")

        model_classifi = get_classifier_model()
        if CONFIG['classifier_model_path']:
            model_classifi.load_state_dict(torch.load(
                CONFIG['classifier_model_path'], map_location=DEVICE))
            logging.info(
                f"load classifier checkpoint from {CONFIG['classifier_model_path']}")
        logging.info("Classifier model initialized.")
        train_classifi(model_classifi, dataloader_cls, run_checkpoint_dir)
    else:
        logging.info(">>> Skip Classifier Training")


if __name__ == "__main__":
    main()
