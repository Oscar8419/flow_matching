import torch
import torch.nn as nn
import torch.optim as optim
import os
from config import CONFIG, DEVICE
import logging
from tqdm import tqdm
from models import GRU


def get_classifier_model():
    model_name = CONFIG["classifier_model"]
    if model_name == "GRU":
        return GRU().to(DEVICE)
    else:
        raise ValueError(f"Unknown classifier model: {model_name}")


def train_fm(model, dataloader, checkpoint_dir):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.MSELoss()

    total_samples_processed = 0
    running_loss = 0.0

    # 估算总 batch 数，用于 tqdm
    total_samples = CONFIG["num_samples"]
    estimated_batches = int(total_samples / CONFIG["batch_size"])
    # 确保 log_interval 至少为 1，防止 ZeroDivisionError
    log_interval = max(1, estimated_batches //
                       CONFIG["times_log"])  # 每多少个 batch 打印一次日志
    save_interval = int(CONFIG["num_samples"] / CONFIG["times_save"])

    # 使用 tqdm 包装 dataloader
    pbar = tqdm(enumerate(dataloader), total=estimated_batches,
                desc="Training", unit="batch")

    for i, (x1, _) in pbar:
        # x1: 目标信号 (Batch, 2, L)
        # _: 标签，Flow Matching 训练暂时不用
        x1 = x1.to(DEVICE)
        B = x1.shape[0]

        # 1. Flow Matching Training Steps

        # A. 采样 x0 (Standard Gaussian Noise)
        # 为了能量匹配，我们将 x0 除以 sqrt(2)，使其总功率也为 1。
        x0 = torch.randn_like(x1).to(DEVICE) / 1.41421

        # B. 采样 t (Uniform [0, 1])
        # t shape: (B,)
        t = torch.rand(B, device=DEVICE)

        # C. 构造 Conditional Flow Path (Optimal Transport / Linear Interpolation)
        # x_t = (1 - t) * x0 + t * x1
        t_reshaped = t.view(-1, 1, 1)
        xt = (1 - t_reshaped) * x0 + t_reshaped * x1

        # D. 计算目标向量场 v_t (Velocity)
        # flow matching objective: v_t = x1 - x0
        target_v = x1 - x0

        # E. 模型预测
        pred_v = model(xt, t)

        # F. Loss & Backprop
        loss = criterion(pred_v, target_v)

        optimizer.zero_grad()
        loss.backward()

        # 增加梯度裁剪，防止 Transformer 梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 2. Logging & Checkpointing
        loss_val = loss.item()
        running_loss += loss_val
        total_samples_processed += B

        # 更新 tqdm 的后缀显示 Loss
        pbar.set_postfix({"loss": f"{loss_val:.6f}"})

        if (i + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            logging.info(
                f"Batch {i+1} | Samples {total_samples_processed} | Loss: {avg_loss:.6f}")
            running_loss = 0.0

        # 根据 total_samples_processed 保存模型
        if total_samples_processed >= CONFIG["num_samples"]:
            logging.info(f"final epoch loss: {loss_val}")
            logging.info(
                "Reached target number of samples. Saving final model and exiting.")
            save_path = os.path.join(checkpoint_dir, f"model_final.pth")
            torch.save(model.state_dict(), save_path)
            break

        # 定期保存 (基于样本数)
        # (total - B) // interval != total // interval 表示跨越了界限
        prev_count = total_samples_processed - B
        curr_count = total_samples_processed

        if (prev_count // save_interval) != (curr_count // save_interval):
            save_path = os.path.join(
                checkpoint_dir, f"model_samples_{curr_count}.pth")
            torch.save(model.state_dict(), save_path)
            logging.info(f"Checkpoint saved to {save_path}")


def train_classifi(model, dataloader, checkpoint_dir):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    total_samples_processed = 0
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    # 估算总 batch 数，用于 tqdm
    total_samples = CONFIG["num_samples"]
    estimated_batches = int(total_samples / CONFIG["batch_size"])
    # 确保 log_interval 至少为 1
    log_interval = max(1, estimated_batches // CONFIG["times_log"])
    save_interval = int(CONFIG["num_samples"] / CONFIG["times_save"])

    # 使用 tqdm 包装 dataloader
    pbar = tqdm(enumerate(dataloader), total=estimated_batches,
                desc="Training Classifier", unit="batch")

    for i, (x1, label) in pbar:
        # x1: (Batch, 2, L)
        # label: (Batch,)
        x1 = x1.to(DEVICE)
        label = label.to(DEVICE)
        B = x1.shape[0]

        # Forward
        logits = model(x1)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Metrics
        loss_val = loss.item()
        preds = torch.argmax(logits, dim=1)
        correct = (preds == label).sum().item()
        acc = correct / B

        running_loss += loss_val
        running_correct += correct
        running_total += B
        total_samples_processed += B

        # 更新 tqdm
        pbar.set_postfix({"loss": f"{loss_val:.4f}", "acc": f"{acc:.4f}"})

        # Logging
        if (i + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            avg_acc = running_correct / running_total
            logging.info(
                f"Batch {i+1} | Samples {total_samples_processed} | Loss: {avg_loss:.6f} | Acc: {avg_acc:.4f}")
            running_loss = 0.0
            running_correct = 0
            running_total = 0

        # 结束条件
        if total_samples_processed >= CONFIG["num_samples"]:
            logging.info(f"Final batch loss: {loss_val:.4f}, acc: {acc:.4f}")
            logging.info(
                "Reached target number of samples. Saving final classifier model and exiting.")
            save_path = os.path.join(
                checkpoint_dir, f"classifier_final.pth")
            torch.save(model.state_dict(), save_path)
            break

        # 定期保存
        prev_count = total_samples_processed - B
        curr_count = total_samples_processed

        if (prev_count // save_interval) != (curr_count // save_interval):
            save_path = os.path.join(
                checkpoint_dir, f"classifier_samples_{curr_count}.pth")
            torch.save(model.state_dict(), save_path)
            logging.info(f"Classifier checkpoint saved to {save_path}")
