import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import src.utils.tools as tools

class XDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, audio_path: str, test_mode: bool, label_map: dict):
        self.df = pd.read_csv(file_path)
        self.audio_df = pd.read_csv(audio_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])

        if self.test_mode == False:
            audio_feature = np.load(self.audio_df.loc[index // 10]['path'])
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
            audio_feature, _ = tools.process_feat(audio_feature, self.clip_dim)
        else:
            audio_feature = np.load(self.audio_df.loc[index]['path'])
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
            audio_feature, _ = tools.process_split(audio_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        audio_feature = torch.tensor(audio_feature)

        clip_feature = torch.cat((clip_feature, audio_feature), dim=-1)

        clip_label = self.df.loc[index]['label']

        return clip_feature, clip_label, clip_length