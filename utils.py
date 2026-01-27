import torch
import torch.nn as nn
import torch.optim as optim
import os
from config import CONFIG, DEVICE
import logging
from tqdm import tqdm
from models import GRU, GRUformer, LSTM, MCLDNN, Transformer, ICAMC


def get_classifier_model(model_name=CONFIG["classifier_model"]):
    if model_name == "GRU":
        return GRU().to(DEVICE)
    elif model_name == "GRUformer":
        return GRUformer().to(DEVICE)
    elif model_name == "LSTM":
        return LSTM().to(DEVICE)
    elif model_name == "MCLDNN":
        return MCLDNN().to(DEVICE)
    elif model_name == "Transformer":
        return Transformer().to(DEVICE)
    elif model_name == "ICAMC":
        return ICAMC().to(DEVICE)
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

    for i, (x1, _, _) in pbar:
        # x1: 目标信号 (Batch, 2, L)
        # _: 标签，Flow Matching 训练暂时不用
        # _: SNR，Flow Matching 训练暂时不用 (因为这里我们是用 x1 自己加噪构造 Conditional Flow)
        x1 = x1.to(DEVICE)
        B = x1.shape[0]

        # 1. Flow Matching Training Steps

        # A. 采样 x0 (Standard Gaussian Noise)
        # 为了能量匹配，我们将 x0 除以 sqrt(2)，使其总功率也为 1。
        x0 = torch.randn_like(x1).to(DEVICE) / 1.41421

        # B. 采样 t, shape: (B,)
        # 0.24 对应 -10dB
        # 简单方案：以 90% 的概率采样 [0.24, 1]，10% 的概率采样 [0, 0.24]
        mask = torch.rand(B, device=DEVICE) < 0.9
        t_high = torch.rand(B, device=DEVICE) * (1 - 0.24) + 0.24
        t_low = torch.rand(B, device=DEVICE) * 0.24
        t = torch.where(mask, t_high, t_low)

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
    total_samples = CONFIG["classifier_num_samples"]
    estimated_batches = int(total_samples / CONFIG["batch_size"])
    # 确保 log_interval 至少为 1
    log_interval = max(1, estimated_batches // CONFIG["times_log"])
    save_interval = int(
        CONFIG["classifier_num_samples"] / CONFIG["times_save"])
    # 使用 tqdm 包装 dataloader
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    pbar = tqdm(enumerate(cycle(dataloader)), total=estimated_batches,
                desc="Training Classifier", unit="batch")

    for i, (x1, label, _) in pbar:
        # x1: (Batch, 2, L)
        # label: (Batch,)
        # _: SNR (暂时不用)
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
        if total_samples_processed >= CONFIG["classifier_num_samples"]:
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


def train_finetune(model_cls, model_fm, dataloader, checkpoint_dir):
    model_cls.train()
    model_fm.eval()  # Flow Matching 模型保持评估模式
    optimizer = optim.Adam(model_cls.parameters(), lr=CONFIG["learning_rate"]/5)
    criterion = nn.CrossEntropyLoss()

    total_samples_processed = 0
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    # 估算总 batch 数
    total_samples = CONFIG["num_samples_finetune"]
    estimated_batches = int(total_samples / CONFIG["batch_size"])
    log_interval = max(1, estimated_batches // CONFIG["times_log"])
    save_interval = int(total_samples / (CONFIG["times_save"]*4))

    pbar = tqdm(enumerate(dataloader), total=estimated_batches,
                desc="Finetuning Classifier", unit="batch")

    # FM 推理步数
    FM_STEPS = 10

    for i, (noisy_sig, label, snr) in pbar:
        noisy_sig = noisy_sig.to(DEVICE)
        label = label.to(DEVICE)
        snr = snr.to(DEVICE)
        B = noisy_sig.shape[0]

        # --- 1. FM 去噪过程 (无梯度) ---
        with torch.no_grad():
            # 计算扩散时间步 t
            # snr_db -> R -> t
            # R = 10^(snr / 20)
            # t = R / (1 + R)
            R = 10 ** (snr / 20.0)
            t_start = R / (1 + R)

            # 计算输入缩放因子 (Matching eval_model logic)
            # scale = sqrt((1-t)^2 + t^2)
            scale_factor = torch.sqrt(
                (1 - t_start)**2 + t_start**2).view(-1, 1, 1)

            # FM 输入准备
            xt = noisy_sig * scale_factor

            # --- Batched Euler ODE Solver ---
            # 计算每个样本的时间步长 dt = (1.0 - t_start) / steps
            dt = (0.75 - t_start) / FM_STEPS
            dt = dt.view(-1, 1, 1)  # (B, 1, 1) 用于广播

            t_curr = t_start.clone()  # (B,)
            curr_x = xt.clone()

            for step in range(FM_STEPS):
                # 预测速度场 v
                v_pred = model_fm(curr_x, t_curr)

                # Euler 更新: x_new = x_old + v * dt
                curr_x = curr_x + v_pred * dt

                # 更新时间: t_new = t_old + dt
                t_curr = t_curr + dt.squeeze()  # squeeze 回 (B,) 以传入 model_fm

            sig_denoised = curr_x

            # 归一化功率
            energy = torch.sum(sig_denoised**2, dim=(1, 2), keepdim=True)
            length = sig_denoised.shape[2]
            power = energy / length
            sig_denoised = sig_denoised / torch.sqrt(power + 1e-8)

        # --- 2. 微调分类器 (有梯度) ---
        # 此时 sig_denoised 就像是"增强"过的数据
        logits = model_cls(sig_denoised)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model_cls.parameters(), max_norm=1.0)
        optimizer.step()

        # --- Metrics & Logging ---
        loss_val = loss.item()
        preds = torch.argmax(logits, dim=1)
        correct = (preds == label).sum().item()
        acc = correct / B

        running_loss += loss_val
        running_correct += correct
        running_total += B
        total_samples_processed += B

        pbar.set_postfix({"loss": f"{loss_val:.4f}", "acc": f"{acc:.4f}"})

        if (i + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            avg_acc = running_correct / running_total
            logging.info(
                f"Batch {i+1} | Samples {total_samples_processed} | Loss: {avg_loss:.6f} | Acc: {avg_acc:.4f}")
            running_loss = 0.0
            running_correct = 0
            running_total = 0

        # 保存 Final Model
        if total_samples_processed >= total_samples:
            logging.info(f"Final batch loss: {loss_val:.4f}, acc: {acc:.4f}")
            logging.info(
                "Reached target number of samples. Saving finetuned model.")
            save_path = os.path.join(
                checkpoint_dir, f"classifier_finetuned_final.pth")
            torch.save(model_cls.state_dict(), save_path)
            break

        # Checkpoint Saving
        prev_count = total_samples_processed - B
        curr_count = total_samples_processed
        if (prev_count // save_interval) != (curr_count // save_interval):
            save_path = os.path.join(
                checkpoint_dir, f"classifier_finetuned_{curr_count}.pth")
            torch.save(model_cls.state_dict(), save_path)
            logging.info(f"Finetuned checkpoint saved to {save_path}")
