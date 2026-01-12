import torch
import torch.nn as nn
from config import CONFIG


class GRU(nn.Module):
    def __init__(self,
                 signal_length=CONFIG["signal_length"],
                 patch_size=16,
                 hidden_size=256,
                 num_classes=CONFIG["num_classes"]):
        super().__init__()
        self.patch_size = patch_size

        # Ensure signal length is divisible by patch size
        assert signal_length % patch_size == 0, "Signal length must be divisible by patch size"

        self.seq_len = signal_length // patch_size
        # Input dimension per step: 2 channels (I/Q) * patch_size
        self.input_size = 2 * patch_size

        # Single layer unidirectional GRU
        self.gru = nn.GRU(
            input_size=self.input_size,
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
        B, C, L = x.shape

        # 1. Slice and flatten Logic
        # View as (Batch, 2, Num_Patches, Patch_Size)
        x = x.view(B, C, self.seq_len, self.patch_size)

        # Permute to (Batch, Num_Patches, 2, Patch_Size)
        x = x.permute(0, 2, 1, 3)

        # Flatten last two dimensions to create sequence element features
        # (Batch, Num_Patches, 2 * Patch_Size)
        x = x.reshape(B, self.seq_len, -1)

        # 2. GRU Forward
        # output shape: (Batch, Seq_Len, Hidden_Size)
        output, _ = self.gru(x)

        # 3. Take logic of last time step
        last_output = output[:, -1, :]

        # 4. Classification
        logits = self.classifier(last_output)

        return logits
