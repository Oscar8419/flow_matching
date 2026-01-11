import torch
import numpy as np
from config import DEVICE


@torch.no_grad()
def sampling(model, num_samples, signal_length=1024, steps=50, method='euler', device=DEVICE):
    """
    生成器主入口函数
    Args:
        model: 训练好的 Flow Matching 模型
        num_samples: 生成多少条样本
        steps: ODE 求解步数 (NFE)
        method: 'euler' or 'rk4'
    Returns:
        generated_signals: (num_samples, 2, signal_length)
    """
    model.eval()

    # 1. 初始化噪声 x0
    # 注意: 训练时我们将 Standard Gaussian (/sqrt(2)) 作为 x0
    # 因此这里必须保持一致
    x0 = torch.randn(num_samples, 2, signal_length).to(device) / 1.41421

    print(
        f"Sampling {num_samples} samples using {method} solver with {steps} steps...")

    # 2. 求解 ODE
    x1 = ode_solve(model, x0, steps, method)

    return x1


@torch.no_grad()
def ode_solve(model, x0, steps=50, method='euler'):
    """
    通用 ODE 求解器
    """
    x = x0.clone()
    B = x.shape[0]
    device = x.device

    dt = 1.0 / steps

    # 简单的进度展示，如果 steps 很大可以换成 tqdm
    for i in range(steps):
        t_scalar = i * dt

        # 当前 batch 的时间 t
        t = torch.full((B,), t_scalar, device=device)

        if method == 'euler':
            # v = f(x, t)
            v = model(x, t)
            # x_{k+1} = x_k + v * dt
            x = x + v * dt

        elif method == 'rk4':
            # Classical Runge-Kutta 4

            # k1
            k1 = model(x, t)

            # k2
            t_half = torch.full((B,), t_scalar + dt/2, device=device)
            k2 = model(x + k1 * (dt / 2), t_half)

            # k3
            k3 = model(x + k2 * (dt / 2), t_half)

            # k4
            t_next = torch.full((B,), t_scalar + dt, device=device)
            k4 = model(x + k3 * dt, t_next)

            x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        elif method == 'midpoint':
            # Midpoint method (Heun's method without iteration / RK2)
            # k1 = f(x, t)
            k1 = model(x, t)

            # k2 = f(x + k1 * dt/2, t + dt/2)
            t_half = torch.full((B,), t_scalar + dt/2, device=device)
            k2 = model(x + k1 * (dt / 2), t_half)

            x = x + k2 * dt

        else:
            raise ValueError(f"Unknown solver method: {method}")

    return x
