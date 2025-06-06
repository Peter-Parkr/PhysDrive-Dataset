import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
# -*- coding: UTF-8 -*-
import sys
from torchvision import models
from torch.nn import InstanceNorm2d, BatchNorm2d
# from models.basic_modules import *
from scipy.signal import firwin # Used for designing FIR filter coefficients

class RadarSignalProcessor(nn.Module):
    def __init__(self, num_frames, fs,
                 breathing_band_hz=[0.1, 0.5],
                 heartbeat_band_hz=[0.8, 2.0],
                 filter_order=64):
        """
        Initializes the radar signal processing module.

        Args:
            num_frames (int): Number of time frames in the input signal (F).
            fs (float): Sampling frequency of the frames in Hz.
            breathing_band_hz (list): [low_cutoff_hz, high_cutoff_hz] for breathing.
            heartbeat_band_hz (list): [low_cutoff_hz, high_cutoff_hz] for heartbeat.
            filter_order (int): Order of the FIR bandpass filters.
                                For scipy.signal.firwin using 'bandpass', this should be even.
                                The number of filter taps will be filter_order + 1.
        """
        super().__init__()
        self.num_frames = num_frames
        self.fs = fs
        self.filter_order = filter_order
        kernel_size = filter_order + 1

        # --- Bandpass Filters (as 1D Convolutional Layers) ---
        # These Conv1d layers will apply FIR filters.
        # in_channels=1, out_channels=1 because we process each "pixel's" time series independently.
        # Padding is set to achieve 'same' output length.
        padding = filter_order // 2

        self.breathing_filter = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False # FIR filters typically don't have a bias term
        )
        self.heartbeat_filter = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        # --- Initialize Filter Weights ---
        nyquist = fs / 2.0
        breathing_coeffs = firwin(kernel_size,
                                  [breathing_band_hz[0] / nyquist, breathing_band_hz[1] / nyquist],
                                  pass_zero=False, window='hamming')
        heartbeat_coeffs = firwin(kernel_size,
                                  [heartbeat_band_hz[0] / nyquist, heartbeat_band_hz[1] / nyquist],
                                  pass_zero=False, window='hamming')
        
        # # Convert to PyTorch tensors and assign to filter weights
        self.breathing_filter.weight.data = torch.tensor(breathing_coeffs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.heartbeat_filter.weight.data = torch.tensor(heartbeat_coeffs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.breathing_filter.weight.requires_grad = False
        self.heartbeat_filter.weight.requires_grad = False


    def _apply_filter(self, phase_signals, conv_filter):
        """
        Helper function to apply a 1D convolutional filter.
        Input phase_signals shape: [B, F, D, A, R]
        Output filtered_signals shape: [B, F, D, A, R]
        """
        B, F, D, A, R = phase_signals.shape

        # Permute to bring F (num_frames) to the last dim: [B, D, A, R, F]
        x_permuted = phase_signals.permute(0, 2, 3, 4, 1)

        # Reshape to flatten spatial/Doppler/angle/range dims for batching: [B*D*A*R, F]
        x_reshaped = x_permuted.reshape(-1, F)

        # Unsqueeze to add channel dimension for Conv1d: [B*D*A*R, 1, F]
        x_unsqueezed = x_reshaped.unsqueeze(1)

        # Apply the 1D convolution (filter)
        filtered_unsqueezed = conv_filter(x_unsqueezed) # Output: [B*D*A*R, 1, F]

        # Squeeze channel dimension: [B*D*A*R, F]
        filtered_reshaped = filtered_unsqueezed.squeeze(1)

        # Reshape back to [B, D, A, R, F]
        filtered_permuted = filtered_reshaped.view(B, D, A, R, F)

        # Permute back to original F-first order: [B, F, D, A, R]
        filtered_final = filtered_permuted.permute(0, 4, 1, 2, 3)

        return filtered_final

    def forward(self, x):
        """
        Processes the radar signal.

        Args:
            x (torch.Tensor): Input tensor with shape
                              [B, num_frames, Real/Imag=2, num_doppler, num_angles, num_rangebins]
                              (B, F, C_complex, D, A, R)

        Returns:
            torch.Tensor: Output tensor with shape
                          [B, num_frames, Filtered_bands=2, num_doppler, num_angles, num_rangebins]
                          (B, F, 2, D, A, R)
        """
        # --- 1. Convert complex data to phase signals ---
        # x shape: [B, F, C_complex=2, D, A, R]
        # Real part is at index 0, Imaginary part is at index 1 along dim 2 (C_complex)
        real_part = x[:, :, 0, ...]  # Shape: [B, F, D, A, R]
        imag_part = x[:, :, 1, ...]  # Shape: [B, F, D, A, R]

        # Calculate phase: atan2(imag, real)
        phase_signals = torch.atan2(imag_part, real_part)
        # phase_signals shape: [B, F, D, A, R]

        # --- 2. Bandpass filtering ---
        # Apply breathing filter
        breathing_filtered = self._apply_filter(phase_signals, self.breathing_filter)
        # breathing_filtered shape: [B, F, D, A, R]

        # Apply heartbeat filter
        heartbeat_filtered = self._apply_filter(phase_signals, self.heartbeat_filter)
        # heartbeat_filtered shape: [B, F, D, A, R]

        # Stack the two filtered signals along a new dimension (dim 2)
        # Output shape: [B, F, 2, D, A, R]
        output = torch.stack([breathing_filtered, heartbeat_filtered], dim=2)

        return output

class AttentionEncoderBlock(nn.Module):
    """
    A single block for the Attention Encoder, consisting of Conv2D -> BatchNorm -> GELU.
    The convolution is applied primarily along the 'height' dimension (inp_sz/time).
    """
    def __init__(self, in_channels, out_channels, kernel_height, stride_height):
        super().__init__()
        # kernel_size=(kernel_height, 1) means convolution along inp_sz dimension
        # stride=(stride_height, 1)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(kernel_height, 1),
                              stride=(stride_height, 1),
                              padding=0) # 'valid' padding
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.bn(self.conv(x)))

class BinAttentionFlatAnt(nn.Module):
    def __init__(self, inp_sz, num_antennas=12, num_bins=64, attention_heads=8, scale=1.0):
        super(BinAttentionFlatAnt, self).__init__()
        self.inp_sz = inp_sz                # W: sequence length / time dimension
        self.num_antennas = num_antennas    # N: number of antennas
        self.num_bins = num_bins            # F: number of bins
        self.attention_heads = attention_heads # Fa: number of attention heads
        self.scale = scale

        # --- Attention Encoder ---
        # Takes input (B, N, W, F) after initial permutations
        # Outputs (B, 32, W_reduced, F)
        self.attention_encoder_blocks = nn.Sequential(
            AttentionEncoderBlock(self.num_antennas, 32, kernel_height=16, stride_height=2),
            AttentionEncoderBlock(32, 32, kernel_height=16, stride_height=2),
            AttentionEncoderBlock(32, 32, kernel_height=16, stride_height=2)
        )

        # --- Attention Projector ---
        # Dense layer, takes features per bin (32 features) and projects to attention_heads
        # Input to Linear will be (B, F, 32)
        self.dense_projector = nn.Linear(32, self.attention_heads)
        # Softmax over the F (num_bins) dimension for attention_weights
        # Input to Softmax will be (B, F, attention_heads)
        self.softmax = nn.Softmax(dim=1) # Apply softmax along the F (num_bins) dimension


    def forward(self, model_input):
        # model_input shape: (batch_size, inp_sz, num_antennas, num_bins) -> (B, W, N, F)
        batch_size = model_input.size(0)

        # TF Step 1 & 2: Permute input for Conv2D processing
        # model_input (B, W, N, F) -> TF Permute([1,3,2]) -> (B, W, F, N)
        x_tf_permuted = model_input.permute(0, 1, 3, 2) # Shape: (B, W, F, N)

        # Prepare for PyTorch Conv2D: (B, Channels, Height, Width)
        # We need (B, N, W, F) where N=num_antennas (channels), W=inp_sz (height), F=num_bins (width)
        x_conv_input = x_tf_permuted.permute(0, 3, 1, 2) # Shape: (B, N, W, F)

        # TF Step 3: Attention Encoder
        # Input: (B, N, W, F)
        # Output: (B, 32, W_reduced, F)
        encoded_attention = self.attention_encoder_blocks(x_conv_input)

        # TF Step 4: Reduce mean along the (now reduced) W dimension
        # TF input was (B, W_reduced, F, 32), mean over axis 1 (W_reduced)
        # PyTorch input is (B, 32, W_reduced, F), mean over axis 2 (W_reduced)
        # Output shape: (B, 32, F)
        projector_input = torch.mean(encoded_attention, dim=2)

        # TF Step 5: Dense projection
        # TF input was (B, F, 32). PyTorch input is (B, 32, F). Permute for Linear.
        projector_input = projector_input.permute(0, 2, 1) # Shape: (B, F, 32)
        # Output shape: (B, F, attention_heads)
        attention_scores = self.dense_projector(projector_input)

        # TF Step 6: Softmax and scale to get attention weights
        # Input: (B, F, attention_heads). Softmax along F (dim=1).
        # Output: (B, F, attention_heads)
        multi_head_attention_weights = self.softmax(attention_scores) * self.scale

        # TF Step 7: Reshape original model_input for matmul
        # model_input: (B, W, N, F) -> Reshape to (B, W*N, F)
        reshaped_model_input = model_input.reshape(
            batch_size,
            self.inp_sz * self.num_antennas,
            self.num_bins
        ) # Shape: (B, W*N, F)

        # TF Step 8: Matmul with attention weights
        # (B, W*N, F) @ (B, F, attention_heads) -> (B, W*N, attention_heads)
        attended_features = torch.matmul(reshaped_model_input, multi_head_attention_weights)

        # TF Step 9: Reshape after matmul
        # (B, W*N, attention_heads) -> (B, W, N, attention_heads)
        attended_features = attended_features.reshape(
            batch_size,
            self.inp_sz,
            self.num_antennas,
            self.attention_heads
        ) # Shape: (B, W, N, attention_heads)

        # TF Step 10: Permute
        # (B, W, N, attention_heads) -> TF Permute([1,3,2]) -> (B, W, attention_heads, N)
        attended_features = attended_features.permute(0, 1, 3, 2) # Shape: (B, W, attention_heads, N)

        return attended_features
        
class EncoderDecoderBlock(nn.Module):
    """
    A block for the encoder part of EncoderDecoder, using Conv2D.
    TF order: Conv(act='gelu') -> Dropout -> [Optional Add Residual] -> BN
    """
    def __init__(self, in_channels, out_channels, apply_residual):
        super().__init__()
        self.apply_residual = apply_residual

        # Convolution: kernel (3,1), stride (2,1), padding 'same'
        # For kernel=3, stride=2, padding=1 achieves 'same' height output: ceil(H_in/2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.2)

        if self.apply_residual:
            # AveragePooling for residual: kernel (3,1), stride (2,1), padding 'same'
            # For kernel=3, stride=2, padding=1 achieves 'same' height output: ceil(H_in/2)
            self.avg_pool_residual = nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
            # Residual connection is added only if not the first block (kk > 0 in TF)
            # And channels must match. This is implicitly handled by TF code structure where
            # for kk>0, conv `filters` matches input channels of that iteration.

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        # Main path
        out = self.conv(x)
        out = self.gelu(out)
        out = self.dropout(out)

        # Residual path
        if self.apply_residual: # Corresponds to if kk > 0 in TF
            # In TF, rx = AvgPool(xx_at_start_of_iter)
            # xx_after_conv = Conv(xx_at_start_of_iter)
            # if kk > 0: xx_after_conv = xx_after_conv + rx
            # This implies channels of rx (from xx_at_start_of_iter) must match channels of xx_after_conv.
            # This holds if the conv for kk>0 iterations doesn't change channels relative to its input.
            # The TF code structure ensures this:
            # e.g., 1st loop: kk=0 (1ch -> 32*fx ch), kk=1 (32*fx ch -> 32*fx ch)
            rx = self.avg_pool_residual(identity)
            if rx.shape[1] != out.shape[1]:
                # This case should ideally not happen if TF logic is strictly followed for kk>0 iterations.
                # If it were to happen, a projection on rx would be needed.
                # However, given the TF structure, `filters` for kk>0 Conv layers match input channels.
                # So, this explicit projection is skipped, assuming direct addability.
                # For safety, one might add a check or ensure `in_channels == out_channels` when `apply_residual` is True.
                # print(f"Warning: Channel mismatch in residual add: rx {rx.shape[1]} vs out {out.shape[1]}")
                pass # Assuming channels match due to TF structure for kk>0
            out = out + rx

        out = self.bn(out)
        return out

class DecoderBlock(nn.Module):
    """
    A block for the decoder part of EncoderDecoder, using ConvTranspose2D.
    TF order: ConvTranspose(act='gelu') -> Dropout -> [Optional Add Residual] -> BN
    """
    def __init__(self, in_channels, out_channels, apply_residual):
        super().__init__()
        self.apply_residual = apply_residual

        # ConvTranspose2D: kernel (3,1), stride (2,1), padding 'same'
        # To double height: kernel=3, stride=2, padding=1, output_padding=1
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels,
                                                 kernel_size=(3, 1), stride=(2, 1),
                                                 padding=(1, 0), output_padding=(1, 0))
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.2)

        if self.apply_residual:
            # UpSampling for residual: size (2,1)
            self.upsample_residual = nn.Upsample(scale_factor=(2, 1), mode='nearest')
            # Similar to EncoderBlock, direct addition implies channel compatibility for kk>0.

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        # Main path
        out = self.conv_transpose(x)
        out = self.gelu(out)
        out = self.dropout(out)

        # Residual path
        if self.apply_residual: # Corresponds to if kk > 0 in TF
            rx = self.upsample_residual(identity)
            if rx.shape[1] != out.shape[1]:
                # Similar logic to EncoderBlock, assuming channels match for kk>0.
                # print(f"Warning: Channel mismatch in decoder residual add: rx {rx.shape[1]} vs out {out.shape[1]}")
                pass
            out = out + rx

        out = self.bn(out)
        return out

class EncoderDecoder(nn.Module):
    def __init__(self, initial_input_channels=1, fx=2, attention_width=None):
        super(EncoderDecoder, self).__init__()
        self.fx = fx
        self.attention_width = attention_width # Expected width after attention (attention_heads * num_antennas)

        # --- Encoder Part 1 (enc(t)) ---
        # Input channels: initial_input_channels (should be 1)
        # Output channels of this stage: 32*fx
        enc_t_layers = []
        current_channels = initial_input_channels
        for kk in range(4):
            out_ch = 32 * self.fx
            # Residual is applied if kk > 0.
            # is_first_block_in_stage is True for kk=0, meaning no residual add for this Conv's output.
            # For kk > 0, the conv in_channels should be out_ch from previous, and conv out_channels is also out_ch.
            block_in_channels = current_channels if kk == 0 else out_ch
            enc_t_layers.append(EncoderDecoderBlock(in_channels=block_in_channels, 
                                                    out_channels=out_ch, 
                                                    apply_residual=(kk > 0)))
            current_channels = out_ch # Update for the next potential iteration if logic changes
        self.encoder_part1 = nn.Sequential(*enc_t_layers)
        
        # --- Encoder Part 2 (enc(t,a)) ---
        # Input channels: 32*fx (output of encoder_part1)
        # Output channels of this stage: 64*fx
        enc_t_layers = []
        # current_channels is already 32*fx
        for kk in range(2):
            out_ch = 64 * self.fx
            block_in_channels = current_channels if kk == 0 else out_ch
            enc_t_layers.append(EncoderDecoderBlock(in_channels=block_in_channels, 
                                                    out_channels=out_ch, 
                                                    apply_residual=(kk > 0)))
            current_channels = out_ch
        self.encoder_part2 = nn.Sequential(*enc_t_layers)

        # --- Pooling Part ---
        # Input channels: 64*fx (output of encoder_part2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # Reduces H, W to 1, 1
        # Dense layer, equivalent to Conv2D with kernel (1,1) or Linear
        # Input to Linear will be 64*fx channels
        self.dense_pool = nn.Linear(64 * self.fx, 256)
        self.gelu_pool = nn.GELU()
        self.bn_pool = nn.BatchNorm1d(256) # Applied on (B, 256)

        # --- Decoder Part ---
        # Input channels: 256 (output of pooling_part, reshaped to B, C, 1, 1)
        # Output channels of this stage: 64*fx, then eventually 1
        decoder_layers = []
        current_channels = 256
        for kk in range(7):
            out_ch = 64 * self.fx
            block_in_channels = current_channels if kk == 0 else out_ch
            decoder_layers.append(DecoderBlock(in_channels=block_in_channels, 
                                               out_channels=out_ch, 
                                               apply_residual=(kk > 0)))
            current_channels = out_ch
        self.decoder_part = nn.Sequential(*decoder_layers)

        # --- Final Dense Layer ---
        # Input channels: 64*fx (output of decoder_part)
        # Output channels: 1
        # This Dense in TF is applied on a 4D tensor.
        # If input is (B, C, H, W=1), Dense(1) acts on C.
        # Equivalent to a Conv2D with 1 filter, kernel (1,1) if applied per spatial location.
        # Or, if H also becomes 1 before this, then it's simpler.
        # The TF code implies it's applied while H is still > 1 and W=1.
        self.final_dense = nn.Conv2d(64 * self.fx, 1, kernel_size=1) # Linear activation is default

        print(f"EncoderDecoder initialized with fx={fx}. Expects input (B, {initial_input_channels}, H, W_attn={attention_width})")


    def forward(self, x):
        # Expected input x shape: (batch_size, 1, inp_sz, attention_width)
        # print(f"EncoderDecoder input shape: {x.shape}")

        x = self.encoder_part1(x)
        # print(f"After encoder_part1: {x.shape}")
        x = self.encoder_part2(x)
        # print(f"After encoder_part2: {x.shape}")

        # Pooling Part
        x = self.global_avg_pool(x) # Shape: (B, 64*fx, 1, 1)
        # print(f"After global_avg_pool: {x.shape}")
        
        # Prepare for Dense (Linear): (B, C, 1, 1) -> (B, C)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1) # Shape: (B, 64*fx)
        x = self.dense_pool(x_flat)     # Shape: (B, 256)
        x = self.gelu_pool(x)
        x = self.bn_pool(x)             # Shape: (B, 256)
        # print(f"After dense_pool block: {x.shape}")

        # Reshape for Decoder: (B, 256) -> (B, 256, 1, 1) to act as input to ConvTranspose2D
        x = x.view(batch_size, 256, 1, 1)
        # print(f"Reshaped for decoder_part: {x.shape}")

        x = self.decoder_part(x)
        # print(f"After decoder_part: {x.shape}") # Should be (B, 64*fx, H_decoded, W=1)

        return x

class VitaNet(nn.Module):
    def __init__(self, num_frames=200,  num_range_bins=8, num_doppler=8, num_angles=16):
        super().__init__()
        self.data_processor = RadarSignalProcessor(num_frames=num_frames, fs=20.0, filter_order=4)
        self.doppler_layer = nn.Sequential(
                            nn.Linear(num_doppler, num_doppler*4),
                            nn.LeakyReLU(0.3),
                            nn.Linear(num_doppler*4, 1),
                            nn.LeakyReLU(0.3),
                        )
        self.attn = BinAttentionFlatAnt(inp_sz=num_frames, num_bins=num_range_bins, num_antennas=num_angles)
        self.encoder_decoder = EncoderDecoder(initial_input_channels=2, fx=2, attention_width=num_range_bins*num_angles)
        
        self.ecg = nn.Sequential(
            nn.Linear(128, num_frames),
            nn.LeakyReLU(0.3),
            nn.Conv1d(128, 32, 11, 1, 5, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3),
            nn.Conv1d(32, 16, 11, 1, 5, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3),
            nn.Conv1d(16, 1, 11, 1, 5, bias=False)
        )
        self.resp = nn.Sequential(
            nn.Linear(128, num_frames),
            nn.LeakyReLU(0.3),
            nn.Conv1d(128, 32, 11, 1, 5, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3),
            nn.Conv1d(32, 16, 11, 1, 5, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3),
            nn.Conv1d(16, 1, 11, 1, 5, bias=False)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.hr = nn.Linear(128, 1)
        self.rr = nn.Linear(128, 1)

    def forward(self, x):
        ## x: [B, 200, 2, 8, 16, 8]
        ### x: [B, num_frames, Real/Imag, num_doppler, num_angles, num_rangebins]

        ### 1. convert x from complex data to phase signals
        # x shape: [B, num_frames, Real/Imag, num_doppler, num_angles, num_rangebins]

        ### 2. convert the phase signals with breathing and heartbeat bandpass filtered signals
        # x shape: [B, num_frames, 2, num_doppler, num_angles, num_rangebins]

        x = self.data_processor(x)

        ### 3. reduce the num_doppler with self.doppler_layer
        # x shape: [B, num_frames, 2, num_angles, num_rangebins]
        x = x.permute(0, 1, 2, 4, 5, 3)
        x = self.doppler_layer(x).squeeze(-1)  #### [B, num_frames, 2, num_angles, num_rangebins]

        ### 4. feed x to the attention layer separately
        x0 = self.attn(x[:, :, 0, :, :])
        x1 = self.attn(x[:, :, 1, :, :])
        x = torch.cat([x0, x1], dim=2)
        ### 5. reshape x to [B, 2, num_frames, num_angles * num_rangebins]
        x = x.reshape(x.shape[0], x.shape[1], 2, -1).permute(0, 2, 1, 3) 

        ### 6. feed x to the encoder-decoder
        # x shape from the decoder: [B, 128, 128]
        x = self.encoder_decoder(x).squeeze(-1)
        ####
        ecg = self.ecg(x)
        resp = self.resp(x)
        hr = self.hr(self.pool(x).squeeze(-1))
        rr = self.rr(self.pool(x).squeeze(-1))
        return ecg, resp, hr, rr
    

if __name__ == "__main__":
    x = torch.randn((2, 200, 2, 8, 16, 8))

    model = VitaNet(num_frames=200, num_angles=16, num_doppler=8, num_range_bins=8)
    print(model)
    y = model(x)
    for i in y:
        print(i.shape)