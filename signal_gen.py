import torch
from torch.utils.data import IterableDataset
import numpy as np
from config import CONFIG


class SignalGenerator:
    def __init__(self, sample_rate=1e6, sps=8, num_symbols=128):
        """
        sps: Samples Per Symbol (每个符号采样的点数，通常 4 或 8)
        num_symbols: 信号包含的符号数量
        """
        self.sps = sps
        self.num_symbols = num_symbols
        self.fs = sample_rate

        # 定义调制方式
        self.mod_schemes = {}

        # PSK
        self.mod_schemes['2PSK'] = [1+0j, -1+0j]
        self.mod_schemes['QPSK'] = [1+1j, 1-1j, -1+1j, -1-1j]
        self.mod_schemes['16PSK'] = [
            np.exp(1j * 2 * np.pi * k / 16) for k in range(16)]

        # ASK
        self.mod_schemes['2ASK'] = [-1+0j, 1+0j]
        self.mod_schemes['4ASK'] = [-3+0j, -1+0j, 1+0j, 3+0j]
        self.mod_schemes['8ASK'] = [complex(x) for x in range(-7, 8, 2)]

        # QAM
        self.mod_schemes['16QAM'] = [x + 1j*y for x in [-3, -1, 1, 3]
                                     for y in [-3, -1, 1, 3]]
        self.mod_schemes['64QAM'] = [
            x + 1j*y for x in range(-7, 8, 2) for y in range(-7, 8, 2)]

        # APSK (DVB-S2 standard-like radii ratios)
        # 16APSK: 4+12
        r1, r2 = 1.0, 2.6
        self.mod_schemes['16APSK'] = [r1 * np.exp(1j * (np.pi/4 + k*np.pi/2)) for k in range(4)] + \
                                     [r2 * np.exp(1j * (np.pi/6 + k*np.pi/6))
                                      for k in range(12)]

        # 32APSK: 4+12+16
        r1, r2, r3 = 1.0, 2.5, 4.3
        self.mod_schemes['32APSK'] = [r1 * np.exp(1j * (np.pi/4 + k*np.pi/2)) for k in range(4)] + \
                                     [r2 * np.exp(1j * (np.pi/6 + k*np.pi/6)) for k in range(12)] + \
                                     [r3 * np.exp(1j * (k*np.pi/8))
                                      for k in range(16)]

        # 预先归一化星座图能量
        for k, v in self.mod_schemes.items():
            constellation = np.array(v)
            self.mod_schemes[k] = constellation / \
                np.sqrt(np.mean(np.abs(constellation)**2))

        # 生成脉冲成型滤波器 (RRC Filter) - 限制信号的带宽（PSD），防止干扰邻道。
        num_taps = 11 * sps
        beta = 0.35  # 滚降系数 #TODO: may sample this randomly later
        t = np.arange(num_taps) - (num_taps-1)//2
        self.h_rrc = np.sinc(t/sps) * np.cos(np.pi*beta *
                                             t/sps) / (1 - (2*beta*t/sps)**2)

    def add_awgn(self, waveform, snr_db):
        """
        添加高斯白噪声 (AWGN)
        """
        # 计算信号功率
        sig_power = np.mean(np.abs(waveform)**2)

        # 根据 SNR 计算噪声功率
        # SNR(dB) = 10 * log10(Ps / Pn)
        # Pn = Ps / 10^(SNR/10)
        noise_power = sig_power / (10**(snr_db / 10.0))

        # 生成复高斯噪声 (实部虚部各分一半功率)
        noise_std = np.sqrt(noise_power / 2.0)
        noise = noise_std * (np.random.randn(len(waveform)) +
                             1j * np.random.randn(len(waveform)))

        return waveform + noise

    def generate_signal(self, mod_type='QPSK', snr_db=None):
        """
        生成一条信号
        snr_db: 如果不为 None,则添加指定 SNR 的 AWGN 噪声
        """
        # 1. 生成随机符号
        constellation = self.mod_schemes[mod_type]
        indices = np.random.randint(0, len(constellation), self.num_symbols)
        symbols = np.array([constellation[i] for i in indices])

        # 2. 上采样 (Upsampling)
        upsampled = np.zeros(self.num_symbols * self.sps, dtype=complex)
        upsampled[::self.sps] = symbols

        # 3. 脉冲成型 (卷积 RRC 滤波器)
        waveform = np.convolve(upsampled, self.h_rrc, mode='same')

        # --- 以下是添加信道损伤 (关键步骤) ---

        # A. 添加相位偏移 (Phase Offset)
        phase_offset = np.random.uniform(0, 2 * np.pi)
        waveform = waveform * np.exp(1j * phase_offset)

        # B. 添加频率偏移 (CFO)
        # 模拟收发机时钟不同步
        # 降低频偏范围，避免星座图旋转过快 (0.01 -> 0.0005)
        freq_offset = np.random.uniform(-0.0005, 0.0005)  # 归一化频偏
        t_vec = np.arange(len(waveform))
        waveform = waveform * np.exp(1j * 2 * np.pi * freq_offset * t_vec)

        # C. 简单的多径衰落 (Rician-like Fading)
        # 模拟强直射路径 + 弱散射
        # k_factor_mag = 0.1 (散射分量强度) TODO: may sample this randomly later
        h_scatter = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
        h = 1.0 + 0.1 * h_scatter
        waveform = waveform * h

        # 4. 功率归一化 (非常重要，Flow Matching 对幅度敏感)
        # 归一化到单位功率
        power = np.mean(np.abs(waveform)**2)
        waveform = waveform / np.sqrt(power)

        # 5. 添加 AWGN (如果指定了 snr_db)
        if snr_db is not None:
            waveform = self.add_awgn(waveform, snr_db)
            # 再次归一化? TODO: may change later
            # 如果再次归一化，SNR 保持不变，但幅度范围受控，这对神经网络通常更好。
            power = np.mean(np.abs(waveform)**2)
            waveform = waveform / np.sqrt(power)

        # 6. 格式转换: Complex -> (2, N) 的实数数组 (I路和Q路)
        # 这种格式可以直接喂给 PyTorch
        output = np.stack([waveform.real, waveform.imag],
                          axis=0).astype(np.float32)

        return output


class RFSignalDataset(IterableDataset):
    def __init__(self, num_samples=CONFIG["num_samples"], signal_len=CONFIG["signal_length"], snr_range=None):
        """
        num_samples: 数据集长度 (样本数量)
        signal_len: 信号长度
        snr_range: tuple (min_db, max_db) 或 None。
                   如果为 None,生成无 AWGN 的"干净"信号 (用于 Flow Matching)
                   如果为 tuple,生成随机 SNR 的噪声信号 (用于分类器训练)
        """
        self.num_samples = num_samples
        self.signal_len = signal_len
        self.snr_range = snr_range
        self.gen = SignalGenerator(num_symbols=signal_len//8)  # 假设 sps=8
        self.mod_types = ['2PSK', 'QPSK', '16PSK',
                          '2ASK', '4ASK', '8ASK',
                          '16QAM', '64QAM',
                          '16APSK', '32APSK']
        # 建立类别映射表
        self.mod_to_idx = {mod: i for i, mod in enumerate(self.mod_types)}

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        # 简单实现：每个 worker (如果有) 都生成一部分数据，或者在这里处理 worker_info
        # 由于是随机生成，只要长度一致即可。
        # 为了兼容 DataLoader 的 len(dataloader) 计算，我们需要确保生成的总数大约为 self.num_samples

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single-process
            num_samples = self.num_samples
        else:  # Multi-process
            # 将总长度平均分配给每个 worker
            per_worker = int(
                np.ceil(self.num_samples / float(worker_info.num_workers)))
            worker_id = worker_info.id
            # 最后一个 worker 可能会少一点，或者多一点，这里简单处理
            num_samples = per_worker

            # 重要：为了防止所有 worker 生成相同的随机数序列，需要重新 seed
            # 可以在这里基于 seed + worker_id 设置 numpy 的随机种子
            # np.random.seed(seed + worker_id)
            np.random.seed((torch.initial_seed() + worker_id) % (2**32))

        for _ in range(num_samples):
            # 1. 随机选一种调制类型
            mod_type = np.random.choice(self.mod_types)

            # 2. 确定 SNR
            current_snr = None
            if self.snr_range is not None:
                current_snr = np.random.uniform(
                    self.snr_range[0], self.snr_range[1])

            # 3. 生成 x1
            x1_np = self.gen.generate_signal(mod_type, snr_db=current_snr)
            x1 = torch.from_numpy(x1_np)

            # 总是返回 (signal, label)
            label = self.mod_to_idx[mod_type]
            yield x1, torch.tensor(label, dtype=torch.long)
