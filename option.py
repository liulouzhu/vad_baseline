import argparse

parser = argparse.ArgumentParser(description='VadCLIP with Co-Attention Fusion')

# Random seed
parser.add_argument('--seed', default=234, type=int)

# CLIP VAD args
parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=1, type=int)
parser.add_argument('--attn-window', default=64, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=7, type=int)
parser.add_argument('--smooth-lamda', default=0.6, type=float)

# Training
parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--batch-size', default=96, type=int)
parser.add_argument('--lr', default=1e-5)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[3, 6, 10])

# Paths
parser.add_argument('--model-path', default='./model/model_xd.pth')
parser.add_argument('--checkpoint-path', default='./model/checkpoint.pth')
parser.add_argument('--train-list', default='./list/xd_CLIP_rgb.csv')
parser.add_argument('--audio-list', default='./list/xd_CLIP_audio.csv')
parser.add_argument('--test-list', default='./list/xd_CLIP_rgbtest.csv')
parser.add_argument('--audio-test', default='./list/xd_CLIP_audiotest.csv')
parser.add_argument('--gt-path', default='./list/gt.npy')
parser.add_argument('--gt-segment-path', default='./list/gt_segment.npy')
parser.add_argument('--gt-label-path', default='./list/gt_label.npy')
parser.add_argument('--logger-path', default='./log_info.log')

# Checkpoint
parser.add_argument('--use-checkpoint', default=False, type=bool)

# ============================================================
# Co-Attention Fusion args (Idea 1)
# ============================================================
# Whether to use Co-Attention fusion (True) or simple addition (False, for ablation)
parser.add_argument('--use-coattn', default=True, type=bool,
                    help='Use Co-Attention fusion. False = simple addition (baseline)')

# Number of attention heads in Co-Attention module
parser.add_argument('--coattn-n-head', default=4, type=int,
                    help='Number of attention heads in Co-Attention fusion')

# Number of layers in Co-Attention Transformer
parser.add_argument('--coattn-layers', default=1, type=int,
                    help='Number of layers in Co-Attention Transformer')

# Audio feature dimension (wav2clip outputs 512-dim)
parser.add_argument('--audio-hidden-dim', default=512, type=int,
                    help='Dimension of audio features from wav2clip')
