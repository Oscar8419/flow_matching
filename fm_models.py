import torch
import torch.nn as nn
import math
from config import CONFIG, DEVICE


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim)
        embeddings = torch.exp(torch.arange(
            half_dim,  dtype=torch.float32) * -embeddings).to(device=time.device)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiTBlock1D(nn.Module):
    """
    1D Transformer Block with Adaptive Layer Norm (adaLN) for time conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(
            hidden_size, eps=1e-6, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )

        # adaLN-Zero: Regress shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        # 这里为了简化，我们使用简单的 scale & shift 注入机制
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, t_emb):
        # x: [B, L, D]
        # t_emb: [B, D]

        # 计算 adaLN 调制参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            t_emb).chunk(6, dim=1)

        # 1. Attention Block
        x_norm1 = self.norm1(x)
        # 调制: x = x * (1 + scale) + shift
        x_norm1 = x_norm1 * (1 + scale_msa.unsqueeze(1)) + \
            shift_msa.unsqueeze(1)

        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # 2. MLP Block
        x_norm2 = self.norm2(x)
        x_norm2 = x_norm2 * (1 + scale_mlp.unsqueeze(1)) + \
            shift_mlp.unsqueeze(1)

        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class RFSignalDiT(nn.Module):
    def __init__(self, in_channels=2, input_len=CONFIG["signal_length"], patch_size=CONFIG["patch_size"],
                 hidden_size=CONFIG["hidden_size"], depth=CONFIG["depth"], num_heads=CONFIG["num_heads"], num_classes=CONFIG["num_classes"]):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.input_len = input_len
        self.num_patches = input_len // patch_size

        # 1. Input Embedding (Patchify)
        # 将 (B, 2, 1024) -> (B, 64, 2*16=32) -> (B, 64, Hidden)
        self.patch_embed = nn.Conv1d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        # 2. Positional Embedding (Learnable)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size))

        # 3. Time Embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # 4. Class Conditioning (Optional)
        self.class_embed = nn.Embedding(num_classes, hidden_size)

        # 5. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock1D(hidden_size, num_heads) for _ in range(depth)
        ])

        # 6. Final Layer (Unpatchify)
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        # Project back to patch vector
        self.final_linear = nn.Linear(hidden_size, patch_size * in_channels)

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        # 初始化 DiT 的标准做法
        nn.init.normal_(self.pos_embed, std=0.02)
        # 初始化 adaLN 最后一层为 0，这非常重要！这使得 block 初始化为 identity map，训练更稳定
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x):
        """
        x: (B, N, patch_size * C) -> (B, C, L)
        """
        B, N, _ = x.shape
        x = x.reshape(B, N, self.in_channels, self.patch_size)
        x = x.permute(0, 2, 1, 3)  # (B, C, N, P)
        x = x.reshape(B, self.in_channels, -1)
        return x

    def forward(self, x, t, class_labels=None):
        """
        x: (B, 2, 1024) - Noisy Signal
        t: (B,) - Time steps (0-1) or discrete
        class_labels: (B,) - Modulation type indices
        """
        # 1. Time & Class Embeddings
        t_emb = self.time_mlp(t)  # (B, Hidden)
        if class_labels is not None:
            # 将 class embedding 加到 time embedding 上 (简单的 conditioning)
            c_emb = self.class_embed(class_labels)
            t_emb = t_emb + c_emb

        # 2. Patchify & Embedding
        # x: (B, C, L) -> (B, Hidden, N)
        x = self.patch_embed(x)
        x = x.transpose(1, 2)  # (B, N, Hidden)

        # Add Positional Embedding
        x = x + self.pos_embed

        # 3. Transformer Blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # 4. Final Projection
        x = self.final_norm(x)
        x = self.final_linear(x)  # (B, N, patch_size*C)

        # 5. Unpatchify
        output = self.unpatchify(x)  # (B, C, L)

        # Flow Matching 预测的是向量场 v (即 x1 - x0 )
        return output

    @torch.no_grad()
    def predict_x1(self, xt, t_start, steps=50, method='euler'):
        """
        使用欧拉法 (Euler Method) 或中点法 (Midpoint Method) 从当前时刻 t_start 逐步推演。
        """
        # 如果是标量，扩展为 batch
        if isinstance(t_start, float) or (isinstance(t_start, torch.Tensor) and t_start.ndim == 0):
            t_curr = torch.ones(xt.shape[0], device=xt.device) * t_start
        else:
            t_curr = t_start

        # 即使 t_start 是向量，我们也假设大家步调一致，取第一个的值来生成时间网格
        # 如果 batch 内 t_start 不同，则需要更复杂的 mask 处理，这里简化假设一致。
        t_scalar = t_curr[0].item()

        if t_scalar >= 1.0:
            return xt

        target_t = 0.8
        dt = (target_t - t_scalar) / steps
        current_x = xt.clone()

        # 生成时间步 (t_start -> ... -> target_t)
        time_grid = torch.linspace(t_scalar, target_t, steps + 1, device=xt.device)

        for i in range(steps):
            # 当前时间 t
            t_now = time_grid[i]
            # 构造 batch 的 t 输入
            t_batch = torch.full((xt.shape[0],), t_now, device=xt.device)

            if method == 'euler':
                # 1. 预测速度场 v
                v_pred = self.forward(current_x, t_batch)
                # 2. 欧拉更新: x(new) = x(old) + v * dt
                current_x = current_x + v_pred * dt

            elif method == 'midpoint':
                # 1. 预测当前点速度 v1
                v1 = self.forward(current_x, t_batch)
                
                # 2. 预测中点状态
                x_mid = current_x + v1 * dt / 2
                t_mid = t_now + dt / 2
                t_mid_batch = torch.full((xt.shape[0],), t_mid, device=xt.device)
                
                # 3. 预测中点速度 v2
                v2 = self.forward(x_mid, t_mid_batch)
                
                # 4. 中点法更新
                current_x = current_x + v2 * dt
            
            else:
                raise ValueError(f"Unknown sampling method: {method}")

        # 归一化功率
        energy = torch.sum(current_x**2, dim=(1, 2), keepdim=True)
        length = current_x.shape[2]
        power = energy / length
        current_x = current_x / torch.sqrt(power + 1e-8)

        return current_x
