# VadCLIP with Co-Attention Audio-Visual Fusion

**基于 VadCLIP (AAAI 2024) 的改进：引入共注意 Transformer 实现音频-视觉融合**

---

## 主要改进：Idea 1 — Co-Attention Transformer Fusion

### 原始 VadCLIP（简单加法融合）
```
视觉特征 = CLIP ViT → Temporal Transformer → GCN
音频特征 = wav2clip → LSTM
融合方式 = 简单加法: visual + audio
```

### 改进后（Co-Attention 融合）
```
视觉特征 = CLIP ViT → Temporal Transformer → GCN
音频特征 = wav2clip → AudioTemporalEncoder (Transformer)

融合模块 = Multi-Head Cross-Attention + Learnable Gating:
  - Visual attends to Audio (cross-attention)
  - Audio attends to Visual (cross-attention)
  - 可学习门控动态加权两个模态的贡献
```

### 核心设计

**CoAttentionFusion 模块：**
1. **双向交叉注意力**：视觉查询音频、音频查询视觉，每个模态都能看到对方的信息
2. **可学习门控**：`gate_v` 和 `gate_a` 两个参数自动学习哪个模态在哪个时候更可靠
3. **残差连接**：每层交叉注意力后接 FFN，保证梯度稳定

```
# 伪代码
v2a = cross_attn(visual, audio, audio)   # 视觉关注音频
a2v = cross_attn(audio, visual, visual)   # 音频关注视觉
fused = sigmoid(gate_v) * v2a + sigmoid(gate_a) * a2v
```

---

## 消融实验设计

| 实验 | `use_coattn` | 音频编码器 | 融合方式 | 目的 |
|------|-------------|-----------|---------|------|
| **Baseline** | `False` | LSTM | Addition | 原始 VadCLIP + 音频 |
| **Ours** | `True` | Transformer | Co-Attention | 本文方法 |
| **Audio-Only** | `False` | - | - | 仅视觉分支 |
| **Visual-Only** | - | - | - | 仅视觉分支 |

---

## 使用方法

### 训练

```bash
# Co-Attention fusion (本文方法)
python train.py --use-coattn True --coattn-n-head 4 --coattn-layers 1

# 简单加法融合 (消融基线)
python train.py --use-coattn False
```

### 测试

```bash
python test.py --use-coattn True --model-path ./model/model_xd.pth
```

### 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use-coattn` | `True` | 是否使用 Co-Attention 融合 |
| `--coattn-n-head` | `4` | Co-Attention 注意力头数 |
| `--coattn-layers` | `1` | Co-Attention Transformer 层数 |
| `--audio-hidden-dim` | `512` | 音频特征维度 (wav2clip 默认 512) |

---

## 数据要求

- **CLIP 视觉特征**：形状 `[T, 512]`，拼接在数据的前 512 维
- **wav2clip 音频特征**：形状 `[T, 512]`，拼接在数据的后 512 维
- 推荐使用 `xd_CLIP_rgb.csv` / `xd_CLIP_audio.csv` 格式

---

## 论文引用

如果你使用了这个代码，请同时引用 VadCLIP (AAAI 2024)：

```bibtex
@article{wu2024vadclip,
  title={VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection},
  author={Wu, Peng and Zhou, Xuerong and Pang, Guansong and Zhou, Lingru and Yan, Qingsen and Wang, Peng and Zhang, Yanning},
  booktitle={AAAI},
  year={2024}
}
```
