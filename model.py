from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from src.clip import clip
from src.utils.layers import GraphConvolution, DistanceAdj, AsymmetricDistanceAdj


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


# =============================================================================
# Co-Attention Fusion Module (Idea 1)
# =============================================================================
class CoAttentionFusion(nn.Module):
    """
    Multi-Head Cross-Attention with learnable gating for audio-visual fusion.

    Unlike simple addition (visual + audio), this module allows each modality
    to dynamically attend to the other, learning which modality to trust at
    each timestep.
    """

    def __init__(self, d_model: int, n_head: int = 4):
        super().__init__()
        assert d_model % n_head == 0, f"d_model {d_model} must be divisible by n_head {n_head}"
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        # Visual attends to Audio
        self.visual_to_audio_attn = nn.MultiheadAttention(
            d_model, n_head, batch_first=True
        )
        self.visual_norm1 = LayerNorm(d_model)

        # Audio attends to Visual
        self.audio_to_visual_attn = nn.MultiheadAttention(
            d_model, n_head, batch_first=True
        )
        self.audio_norm1 = LayerNorm(d_model)

        # FFN after cross-attention
        self.visual_ffn = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.audio_ffn = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.visual_norm2 = LayerNorm(d_model)
        self.audio_norm2 = LayerNorm(d_model)

        # Learnable gating: controls how much each modality contributes to fusion
        self.gate_v = nn.Parameter(torch.zeros(1))  # visual gate
        self.gate_a = nn.Parameter(torch.zeros(1))  # audio gate

        # Output projection
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, visual: torch.Tensor, audio: torch.Tensor,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            visual: [B, T, D] visual features
            audio:  [B, T, D] audio features
            padding_mask: [B, T] True for padded positions
        Returns:
            fused: [B, T, D] fused features
        """
        B, T, D = visual.shape

        # ---- Cross-Attention: Visual attends to Audio ----
        v2a, _ = self.visual_to_audio_attn(
            query=visual,
            key=audio,
            value=audio,
            key_padding_mask=padding_mask
        )
        visual = visual + v2a
        visual = self.visual_norm1(visual)
        visual = visual + self.visual_ffn(visual)
        visual = self.visual_norm2(visual)

        # ---- Cross-Attention: Audio attends to Visual ----
        a2v, _ = self.audio_to_visual_attn(
            query=audio,
            key=visual,
            value=visual,
            key_padding_mask=padding_mask
        )
        audio = audio + a2v
        audio = self.audio_norm1(audio)
        audio = audio + self.audio_ffn(audio)
        audio = self.audio_norm2(audio)

        # ---- Gated Fusion ----
        # sigmoid gates ensure 0-1 range, initialized to 0.5 (equal weight)
        g_v = torch.sigmoid(self.gate_v + torch.log(torch.tensor(2.0)))
        g_a = torch.sigmoid(self.gate_a + torch.log(torch.tensor(2.0)))
        # Normalize so they sum to 1
        g_sum = g_v + g_a + 1e-8
        g_v = g_v / g_sum
        g_a = g_a / g_sum

        fused = g_v * visual + g_a * audio

        return fused


# =============================================================================
# Audio Temporal Encoder (replaces LSTM)
# =============================================================================
class AudioTemporalEncoder(nn.Module):
    """
    Temporal Transformer encoder for audio features.
    Replaces the original LSTM with a position-aware Transformer,
    which better captures long-range temporal dependencies in audio.
    """

    def __init__(self, d_model: int, n_layers: int, n_head: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.encoder = Transformer(
            width=d_model,
            layers=n_layers,
            heads=n_head,
            attn_mask=attn_mask
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D_audio] audio features (D_audio may differ from D_visual)
        Returns:
            [B, T, D_model] encoded audio
        """
        # x: [B, T, D] -> [T, B, D] for Transformer
        x = x.permute(1, 0, 2)
        x, _ = self.encoder((x, padding_mask))
        x = x.permute(1, 0, 2)  # [B, T, D]
        return x


# =============================================================================
# Main Model: CLIPVAD with Co-Attention Fusion
# =============================================================================
class CLIPVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device,
                 # New args for Co-Attention fusion
                 audio_hidden_dim: int = 512,
                 coattn_n_head: int = 4,
                 coattn_layers: int = 1,
                 use_coattn: bool = True):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device
        self.use_coattn = use_coattn

        # ---- Visual Temporal Encoder (unchanged) ----
        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        # ---- GCN for Visual (unchanged) ----
        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        # ---- Audio Temporal Encoder (replaces LSTM) ----
        # Project audio features to visual_width dimension
        self.audio_proj = nn.Linear(audio_hidden_dim, visual_width)
        self.audio_temporal = AudioTemporalEncoder(
            d_model=visual_width,
            n_layers=coattn_layers,
            n_head=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )
        self.audio_norm = LayerNorm(visual_width)

        # ---- Co-Attention Fusion Module (Idea 1) ----
        if self.use_coattn:
            self.coattn_fusion = CoAttentionFusion(
                d_model=visual_width,
                n_head=coattn_n_head
            )

        # ---- Classification Heads (unchanged structure) ----
        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        # ---- CLIP Model (frozen) ----
        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()

        # ---- Original LSTM (kept for ablation comparison) ----
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_norm = nn.LayerNorm(visual_width)

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window,
                     i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length,
                     i * attn_window: self.visual_length] = 0
        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1))
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2 / (x_norm_x + 1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2
        return output

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)

        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])

        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x

    def encode_audio(self, audio: torch.Tensor, padding_mask=None) -> torch.Tensor:
        """
        Encode audio features using AudioTemporalEncoder (Transformer).
        Replaces the original LSTM encoder.
        """
        # Project audio to visual_width dimension
        audio = self.audio_proj(audio)  # [B, T, 512] -> [B, T, visual_width]
        # Temporal encoding with Transformer
        audio = self.audio_temporal(audio, padding_mask)  # [B, T, visual_width]
        audio = self.audio_norm(audio)
        return audio

    def encode_audio_lstm(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Original LSTM-based audio encoder (for ablation comparison).
        """
        audio_feat, _ = self.lstm(audio)  # [B, T, 512]
        audio_feat = self.lstm_norm(audio_feat)
        return audio_feat

    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(
            torch.arange(77).to(self.device)
        ).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = \
                word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = \
                word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = \
                word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)
        return text_features

    def forward(self, visual, padding_mask, text, lengths):
        """
        Forward pass with Co-Attention Fusion.

        Compared to original CLIPVAD:
        - Audio encoder: LSTM -> AudioTemporalEncoder (Transformer)
        - Fusion:       addition -> Co-Attention with learnable gating
        """
        visual_only = visual[:, :, :512]      # [B, T, 512]
        audio_only = visual[:, :, 512:]       # [B, T, 512]

        # ---- Visual branch (unchanged) ----
        visual_features = self.encode_video(visual_only, padding_mask, lengths)

        # ---- Audio branch: choose encoder ----
        if self.use_coattn:
            # New: Transformer-based audio encoder
            audio_features = self.encode_audio(audio_only, padding_mask)
        else:
            # Original: LSTM audio encoder
            audio_features = self.encode_audio_lstm(audio_only)

        # ---- Co-Attention Fusion (replaces simple addition) ----
        if self.use_coattn:
            fused_features = self.coattn_fusion(
                visual_features, audio_features, padding_mask
            )
        else:
            # Fallback: simple addition (same as original)
            fused_features = visual_features + audio_features

        # ---- Text & Classification (unchanged) ----
        text_features_ori = self.encode_textprompt(text)
        logits1 = self.classifier(
            fused_features + self.mlp2(fused_features)
        )

        text_features = text_features_ori
        logits_attn = logits1.permute(0, 2, 1)

        visual_attn = logits_attn @ fused_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        visual_attn = visual_attn.expand(
            visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2]
        )
        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(
            visual_attn.shape[0], text_features.shape[1], text_features.shape[2]
        )
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        fused_features_norm = fused_features / fused_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)

        logits2 = fused_features_norm @ text_features_norm.type(fused_features_norm.dtype) / 0.07

        return text_features_ori, logits1, logits2
