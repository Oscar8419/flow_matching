import torch.nn.functional as F
import torch
import torch.nn as nn
from config import CONFIG


class GRU(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 num_classes=CONFIG["num_classes"]):
        super().__init__()
        # Single layer unidirectional GRU
        self.gru1 = nn.GRU(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.gru2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # Classification head
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x: (Batch, 2, Signal_Length)
        """
        x = x.permute(0, 2, 1)  # (B, L, 2)


        # 2. GRU Forward
        # output shape: (Batch, Seq_Len, Hidden_Size)
        output1, _ = self.gru1(x)
        output2, _ = self.gru2(output1)

        # 3. Take logic of last time step
        last_output = output2[:, -1, :]
        last_output = F.dropout(last_output, p=0.1)

        # 4. Classification
        logits = self.classifier(last_output)

        return logits

class LSTM(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 num_classes=CONFIG["num_classes"]):
        super().__init__()
        # Single layer unidirectional LSTM
        self.lstm1 = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # Classification head
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x: (Batch, 2, Signal_Length)
        """
        x = x.permute(0, 2, 1)  # (B, L, 2)


        # 2. LSTM Forward
        # output shape: (Batch, Seq_Len, Hidden_Size)
        output1, _ = self.lstm1(x)
        output2, _ = self.lstm2(output1)

        # 3. Take logic of last time step
        last_output = output2[:, -1, :]
        last_output = F.dropout(last_output, p=0.1)

        # 4. Classification
        logits = self.classifier(last_output)

        return logits

class MCLDNN(nn.Module):
    def __init__(self,  num_classes=CONFIG["num_classes"]):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(2, 8), padding='same'),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(1, 50, kernel_size=8, padding=0),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(1, 50, kernel_size=8, padding=0),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(1, 8), padding='same'),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=(2, 5), padding='valid'),
            nn.ReLU(),
        )
        self.lstm1 = nn.LSTM(input_size=100, hidden_size=128,
                             batch_first=True, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128,
                             batch_first=True, num_layers=1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor):
         # [N, 2, 1024]
        x = x.unsqueeze(1)  # [N, 1, 2,1024]
        x1 = self.conv1(x)  # x1.shape=[N, 50, 2,1024]
        x2 = self.conv2(x[:, :, 0, :])  # x2.shape=[N, 50, 1024]
        x3 = self.conv3(x[:, :, 1, :])  # x3.shape=[N, 50, 1024]
        # x4.shape=[N, 50, 2,1024]
        x4 = self.conv4(torch.stack([x2, x3], dim=2))
        # x5.shape=[N, 100, 1, 1020]
        x5 = self.conv5(torch.cat([x1, x4], dim=1))
        x5 = x5.permute(0, 2, 3, 1)  # [N, 1, 1020, 100]
        x6 = x5.reshape(x5.size(0), -1, 100)  # x6.shape=[N, 1020, 100]
        x, _ = self.lstm1(x6)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # shape = [N, 128]
        x = self.classifier(x)
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim=16, hidden_size=128, num_classes=CONFIG["num_classes"], signal_len=CONFIG["signal_length"], num_layer=4, drop_rate=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.signal_len = signal_len
        self.encoder = nn.Sequential(*[CustomTransformerEncoderLayer(d_model=hidden_size, nhead=4, dropout=drop_rate,
                                                                     custom_ffn=GluFfn(dim=hidden_size, hidden_dim=hidden_size*4, activation='gelu'), norm_first=True,)
                                       for _ in range(num_layer)])
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 2*signal_len//input_dim, hidden_size))
        self.cls_embed = nn.Parameter(torch.randn(1, 1, hidden_size)*0.02)
        self.project = nn.Linear(input_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)  # [N, signal_len, 2]
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, self.input_dim//2, 2)\
            .transpose(-2, -1).contiguous().reshape(batch_size, -1, self.input_dim)
        # x.shape = [N, a,self.input_dim] ,a = 2*signal_len // self.input_dim

        # [N, a-1,self.input_dim], drop the last to speed up transformer encoder
        x = x[:, :-1, :]
        x = self.project(x)
        cls_token = self.cls_embed.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [N, a,hidden_size]
        x = x + self.pos_embed
        x = self.encoder(x)
        y = x[:, 0, :]
        y = F.dropout(y, p=0.1)
        out = self.fc(y)
        return out

class ICAMC(nn.Module):
    def __init__(self, input_shape=[2, CONFIG["signal_length"]], num_classes=CONFIG["num_classes"]):
        super(ICAMC, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=8, padding='same')
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=4, padding='same')
        self.conv3 = nn.Conv1d(64, 128, kernel_size=8,  padding='same')
        self.maxpool2 = nn.MaxPool1d(kernel_size=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=8,  padding='same')
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.dropout = nn.Dropout(0.1)
        self.flatten = nn.Flatten()
        flattened_shape = self._get_flattened_shape(input_shape)

        self.dense1 = nn.Linear(flattened_shape, 128)
        self.dense2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
          # [N, 2, 1024]
        x = self.conv1(x)

        x = F.relu(x)

        x = self.maxpool1(x)
        x = self.conv2(x)

        x = F.relu(x)

        x = self.conv3(x)

        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = self.conv4(x)

        x = F.relu(x)
        x = self.dropout(x)
        x = self.flatten(x)

        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        # x = self.softmax(x)
        return x

    def _get_flattened_shape(self, input_shape):
        x = torch.zeros(1, *input_shape)
        # x = x.unsqueeze(0)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.conv4(x)
        flattened_shape = self.flatten(x).shape[1]
        return flattened_shape


class GluFfn(nn.Module):
    """
    确保它接收 dim 输入并返回 dim 输出
    """

    def __init__(
        self,
        dim: int = 128,
        hidden_dim: int | None = 128*4,
        activation: str = "gelu",
        dropout: float = 0.2,
        bias: bool = True
    ):
        super().__init__()
        if hidden_dim is None:
            self.hidden_dim = dim * 4
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        if activation.lower() == "silu" or activation.lower() == "swish":
            self.activation_fn = F.silu
            self.activation_name = "SwiGLU"
        elif activation.lower() == "gelu":
            self.activation_fn = lambda x: F.gelu(x, approximate="tanh")
            self.activation_name = "GeGLU"
        else:
            raise ValueError(
                f"Unsupported activation: {activation}. Choose 'silu' or 'gelu'.")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        value = self.w3(x)
        gated_value = self.activation_fn(gate) * value
        gated_value = self.dropout(gated_value)
        output = self.w2(gated_value)
        return output


class CustomTransformerEncoderLayer(nn.Module):
    """
    一个自定义的 Transformer 编码器层，允许使用自定义的 FFN 模块。
    """

    def __init__(self,
                 d_model: int,                 # 输入/输出特征维度
                 nhead: int,                   # 多头注意力机制的头数
                 custom_ffn: nn.Module,        # 一个实例化的自定义 FFN 模块
                 dropout: float = 0.2,         # 注意力和 FFN 中使用的 dropout 率
                 layer_norm_eps: float = 1e-5,  # LayerNorm 中的 epsilon
                 norm_first: bool = True,
                 **mha_kwargs):                # 传递给 MultiheadAttention 的其他关键字参数
        super().__init__()
        if not isinstance(custom_ffn, nn.Module):
            raise TypeError("custom_ffn 必须是一个 nn.Module 实例")

        self.norm_first = norm_first

        # batch_first=True 让输入/输出格式为 (Batch, Sequence, Feature)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True, **mha_kwargs)

        self.ffn = custom_ffn  # 直接使用传入的自定义 FFN 实例

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)  # MHA 输出后的 Dropout
        self.dropout2 = nn.Dropout(dropout)  # FFN 输出后的 Dropout

    def _sa_block(self, x: torch.Tensor, attn_mask=None, key_padding_mask=None, is_causal=False) -> torch.Tensor:
        """ 执行自注意力块 (包括 dropout) """
        # MultiheadAttention 返回 (attn_output, attn_output_weights)
        attn_output, _ = self.self_attn(x, x, x,
                                        attn_mask=attn_mask,
                                        key_padding_mask=key_padding_mask,
                                        need_weights=False,
                                        is_causal=is_causal)
        return self.dropout1(attn_output)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """ 执行前馈网络块 (包括 dropout) """
        return self.dropout2(self.ffn(x))

    def forward(self,
                src: torch.Tensor,
                src_mask=None,             # 自注意力掩码 (例如用于填充或因果关系)
                src_key_padding_mask=None,  # 指示哪些 key 是填充的掩码
                is_causal: bool = False      # 是否应用因果掩码
                ) -> torch.Tensor:
        """
        Transformer 编码器层的前向传播。

        Args:
            src: 输入序列 (Batch, Sequence Length, d_model)
            src_mask: 自注意力机制的掩码 (加性或布尔)
            src_key_padding_mask: key 的填充掩码 (Batch, Sequence Length),True 表示填充
            is_causal: 如果为 True,自动应用因果掩码 

        Returns:
            输出序列 (Batch, Sequence Length, d_model)
        """
        x = src
        if self.norm_first:
            # Pre-Normalization
            attn_output = self._sa_block(self.norm1(
                x), src_mask, src_key_padding_mask, is_causal)
            x = x + attn_output
            # FFN
            ffn_output = self._ff_block(self.norm2(x))
            x = x + ffn_output
        else:
            # Post-Normalization
            attn_output = self._sa_block(
                x, src_mask, src_key_padding_mask, is_causal)
            x = self.norm1(x + attn_output)
            # FFN
            ffn_output = self._ff_block(x)
            x = self.norm2(x + ffn_output)

        return x


class GRUformer(nn.Module):
    def __init__(self, input_dim=16, hidden_size=128, num_classes=CONFIG["num_classes"], signal_len=CONFIG["signal_length"], num_layer=4, drop_rate=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.signal_len = signal_len

        self.gru1 = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            # shape of input is (batch, seq_len, input_size)
            batch_first=True,
            bidirectional=False
        )

        self.gru2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.encoder = nn.Sequential(*[CustomTransformerEncoderLayer(d_model=hidden_size, nhead=4, dropout=drop_rate,
                                                                     custom_ffn=GluFfn(dim=hidden_size, hidden_dim=hidden_size*4, activation='gelu'), norm_first=True,)
                                       for _ in range(num_layer)])
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 2*signal_len//input_dim, hidden_size))
        self.cls_embed = nn.Parameter(torch.randn(1, 1, hidden_size)*0.02)

        self.fc = nn.Linear(hidden_size, num_classes)

        # init weights
        self._init_weights()

    def _init_weights(self):
        """Xavier init"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)  # [N, signal_len, 2]
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, self.input_dim//2, 2)\
            .transpose(-2, -1).contiguous().reshape(batch_size, -1, self.input_dim)
        # x.shape = [N, a,self.input_dim] ,a = 2*signal_len // self.input_dim

        # [N, a-1,self.input_dim], drop the last to speed up transformer encoder
        x = x[:, :-1, :]

        # first GRU
        gru1_out, _ = self.gru1(x)  # out shape:[N, a-1,hidden_size]
        # second GRU
        gru2_out, _ = self.gru2(gru1_out)  # out shape: [N, a-1,hidden_size]

        cls_token = self.cls_embed.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, gru2_out], dim=1)  # [N, a,hidden_size]
        x = x + self.pos_embed
        x = self.encoder(x)
        y = x[:, 0, :]
        y = F.dropout(y, p=0.1)
        out = self.fc(y)
        return out
