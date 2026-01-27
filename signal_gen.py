import torch
from torch.utils.data import IterableDataset
import numpy as np
from scipy.signal import hilbert
from config import CONFIG


class SignalGenerator:
    def __init__(self, sps=8, num_symbols=128):
        """
        sps: Samples Per Symbol (每个符号采样的点数，通常 4 或 8)
        num_symbols: 信号包含的符号数量
        """
        self.sps = sps
        self.num_symbols = num_symbols

        # 定义调制方式
        self.mod_schemes = {}

        # PSK
        self.mod_schemes['2PSK'] = [1+0j, -1+0j]
        self.mod_schemes['QPSK'] = [1+1j, 1-1j, -1+1j, -1-1j]
        self.mod_schemes['8PSK'] = [
            np.exp(1j * 2 * np.pi * k / 8) for k in range(8)]
        self.mod_schemes['16PSK'] = [
            np.exp(1j * 2 * np.pi * k / 16) for k in range(16)]

        # ASK
        self.mod_schemes['2ASK'] = [-1+0j, 1+0j]
        self.mod_schemes['4ASK'] = [-3+0j, -1+0j, 1+0j, 3+0j]
        self.mod_schemes['8ASK'] = [complex(x) for x in range(-7, 8, 2)]

        # QAM
        self.mod_schemes['16QAM'] = [x + 1j*y for x in [-3, -1, 1, 3]
                                     for y in [-3, -1, 1, 3]]
        # 32QAM: 6x6 grid minus 4 corners (size 1)
        self.mod_schemes['32QAM'] = [x + 1j*y for x in range(-5, 6, 2) for y in range(-5, 6, 2)
                                     if not (abs(x) > 3 and abs(y) > 3)]
        self.mod_schemes['64QAM'] = [
            x + 1j*y for x in range(-7, 8, 2) for y in range(-7, 8, 2)]
        # 128QAM: 12x12 grid minus 4 corners (size 2x2, total 16 pts removed)
        self.mod_schemes['128QAM'] = [x + 1j*y for x in range(-11, 12, 2) for y in range(-11, 12, 2)
                                      if not (abs(x) > 7 and abs(y) > 7)]

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

        # 64APSK: 4+12+20+28
        r1, r2, r3, r4 = 1.0, 2.5, 4.3, 6.2
        self.mod_schemes['64APSK'] = [r1 * np.exp(1j * (np.pi/4 + k*np.pi/2)) for k in range(4)] + \
                                     [r2 * np.exp(1j * (np.pi/6 + k*np.pi/6)) for k in range(12)] + \
                                     [r3 * np.exp(1j * (k*np.pi/10)) for k in range(20)] + \
                                     [r4 * np.exp(1j * (k*np.pi/14))
                                      for k in range(28)]

        # 128APSK: 8+16+24+32+48
        r_list = [1.0, 2.6, 4.2, 5.8, 7.4]
        n_list = [8, 16, 24, 32, 48]
        self.mod_schemes['128APSK'] = []
        for ri, ni in zip(r_list, n_list):
            self.mod_schemes['128APSK'].extend(
                [ri * np.exp(1j * (k * 2 * np.pi / ni)) for k in range(ni)]
            )

        # 预先归一化星座图能量
        for k, v in self.mod_schemes.items():
            constellation = np.array(v)
            self.mod_schemes[k] = constellation / \
                np.sqrt(np.mean(np.abs(constellation)**2))

        # 生成脉冲成型滤波器 (RRC Filter) - 限制信号的带宽（PSD），防止干扰邻道。
        num_taps = 11 * sps
        beta = 0.35
        t = np.arange(num_taps) - (num_taps - 1) // 2
        t_norm = t / sps

        self.h_rrc = np.zeros_like(t_norm)
        for i, ti in enumerate(t_norm):
            if abs(ti) < 1e-8:
                self.h_rrc[i] = 1 - beta + 4 * beta / np.pi
            elif abs(abs(ti) - 1 / (4 * beta)) < 1e-8:
                self.h_rrc[i] = (beta / np.sqrt(2)) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                    (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                )
            else:
                self.h_rrc[i] = (np.sin(np.pi * ti * (1 - beta)) +
                                 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / \
                                (np.pi * ti * (1 - (4 * beta * ti) ** 2))

        # 能量归一化
        self.h_rrc = self.h_rrc / np.sqrt(np.sum(self.h_rrc ** 2))

        # 生成高斯滤波器 (GMSK Pulse Shaping)
        bt = 0.3  # GMSK BT product (e.g. GSM standard)
        # sigma derived from BT: sigma = sqrt(ln2) / (2*pi*BT)
        sigma_gauss = np.sqrt(np.log(2)) / (2 * np.pi * bt)
        # Gaussian function
        # Note: t is in samples. t/sps normalized to symbol periods.
        t_gauss = np.arange(num_taps) - (num_taps-1)//2
        self.h_gauss = (1 / (np.sqrt(2*np.pi) * sigma_gauss * sps)) * \
            np.exp(-((t_gauss/sps)**2) / (2 * sigma_gauss**2))
        # Normalize so that the sum (integral) over time is 1
        self.h_gauss = self.h_gauss / np.sum(self.h_gauss)

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
        if mod_type == 'GMSK':
            # GMSK 调制
            # 1. 生成二进制序列 (-1, +1)
            bits = 2 * np.random.randint(0, 2, self.num_symbols) - 1

            # 2. 上采样
            upsampled = np.zeros(self.num_symbols * self.sps)
            upsampled[::self.sps] = bits

            # 3. 高斯滤波 (得到频率脉冲)
            freq_pulse = np.convolve(upsampled, self.h_gauss, mode='same')

            # 4. 积分得到相位 (相位累积)
            # MSK modulation index h = 0.5, total phase shift per symbol is pi/2
            # Since sum(h_gauss) = 1, cumsum(freq_pulse) for one symbol increases by 1.
            # We scale by pi/2.
            phase = np.cumsum(freq_pulse) * (np.pi / 2)
            waveform = np.exp(1j * phase)

        elif mod_type == 'FM':
            # 模拟 FM 信号 (Analog FM)
            # 1. 源信号: 使用随机的高斯噪声模拟连续的音频/模拟信号
            source = np.random.randn(self.num_symbols)

            # 2. 上采样: 扩展到目标采样率
            upsampled = np.zeros(self.num_symbols * self.sps)
            upsampled[::self.sps] = source

            # 3. 滤波平滑 (使用 RRC 作为低通重建滤波器) -> 得到带限消息信号 m(t)
            msg_signal = np.convolve(upsampled, self.h_rrc, mode='same')

            # 4. FM 调制: x(t) = exp(j * (2*pi*f_c*t + cumsum(m(t))))
            # 基带信号仅需关心相位项: exp(j * phase)
            # sensitivity 控制调制指数 (Modulation Index) 和带宽
            sensitivity = 1.0
            phase = np.cumsum(msg_signal) * sensitivity
            waveform = np.exp(1j * phase)

        elif mod_type == 'AM-SSB':
            # 模拟 AM-SSB 信号 (Analog Single Sideband - USB)
            # 1. 源信号: 使用随机的高斯噪声模拟连续的音频/模拟信号
            source = np.random.randn(self.num_symbols)

            # 2. 上采样
            upsampled = np.zeros(self.num_symbols * self.sps)
            upsampled[::self.sps] = source

            # 3. 滤波平滑
            msg_signal = np.convolve(upsampled, self.h_rrc, mode='same')

            # 4. 希尔伯特变换得到解析信号 (Complex Baseband for USB)
            # Analytic signal = m(t) + j * hilbert(m(t))
            waveform = hilbert(msg_signal)

        else:
            # 数字调制 (PSK/QAM/APSK等)
            # 1. 生成随机符号
            constellation = self.mod_schemes[mod_type]
            indices = np.random.randint(
                0, len(constellation), self.num_symbols)
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
    def __init__(self, num_samples=CONFIG["num_samples"], signal_len=CONFIG["signal_length"], snr_range=None, seed=42):
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
        self.seed = seed
        self.gen = SignalGenerator(
            sps=CONFIG["sps"], num_symbols=signal_len//CONFIG["sps"])  #
        self.mod_types = ['QPSK', '8PSK',
                          '2ASK', '4ASK', '8ASK',
                          '16QAM', '32QAM', '64QAM', '128QAM',
                          '16APSK', '32APSK', '64APSK', '128APSK', 'GMSK']
        # 建立类别映射表
        self.mod_to_idx = {mod: i for i, mod in enumerate(self.mod_types)}

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        # 简单实现：每个 worker (如果有) 都生成一部分数据，或者在这里处理 worker_info
        # 由于是随机生成，只要长度一致即可。
        # 为了兼容 DataLoader 的 len(dataloader) 计算，我们需要确保生成的总数大约为 self.num_samples

        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed # Base seed for reproducibility
        if worker_info is None:  # Single-process
            num_samples = int(self.num_samples)
            np.random.seed(seed)
        else:  # Multi-process
            # 将总长度平均分配给每个 worker
            per_worker = int(
                np.ceil(self.num_samples / float(worker_info.num_workers)))
            worker_id = worker_info.id
            # 最后一个 worker 可能会少一点，或者多一点，这里简单处理
            num_samples = per_worker

            # 固定种子，确保每个 Epoch/循环 产生的数据序列是相同的
            np.random.seed(seed + worker_id)

        for _ in range(num_samples):
            # 1. 随机选一种调制类型
            mod_type = np.random.choice(self.mod_types)

            # 2. 确定 SNR
            current_snr = None
            snr_val = 100.0  # 默认 100dB 表示这里是 Clean Signal
            if self.snr_range is not None:
                current_snr = np.random.uniform(
                    self.snr_range[0], self.snr_range[1])
                snr_val = current_snr

            # 3. 生成 x1
            x1_np = self.gen.generate_signal(mod_type, snr_db=current_snr)
            x1 = torch.from_numpy(x1_np)

            # 总是返回 (signal, label, snr)
            label = self.mod_to_idx[mod_type]
            yield x1, torch.tensor(label, dtype=torch.long), torch.tensor(snr_val, dtype=torch.float32)
