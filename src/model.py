import torch
import torch.nn as nn


class MiniVLA(nn.Module):
    def __init__(
        self,
        vocab_size,
        text_embed_dim=32,
        text_feat_dim=64,
        image_feat_dim=64,
        hidden_dim=128,
        action_dim=3,
        pad_idx=0,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        # 图像编码器
        # 关键改动：
        # 1. 保留更多空间信息，不再直接池化到 1x1
        # 2. 最后输出 [B, image_feat_dim, 4, 4]
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64 -> 32

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16

            nn.Conv2d(32, image_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 保留空间布局
        )

        # 文本编码器
        self.text_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=text_embed_dim,
            padding_idx=pad_idx,
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, text_feat_dim),
            nn.ReLU(),
        )

        # 计算融合维度
        image_out_dim = image_feat_dim * 4 * 4
        fusion_dim = image_out_dim + text_feat_dim

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 动作输出头
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def encode_image(self, image):
        feat = self.image_encoder(image)    # [B, C, 4, 4]
        feat = feat.flatten(start_dim=1)    # [B, C*4*4]
        return feat

    def encode_text(self, instruction):
        # instruction: [B, T]
        emb = self.text_embedding(instruction)  # [B, T, D]

        mask = (instruction != self.pad_idx).unsqueeze(-1).float()  # [B, T, 1]
        emb_sum = (emb * mask).sum(dim=1)                           # [B, D]
        token_count = mask.sum(dim=1).clamp(min=1.0)                # [B, 1]
        pooled = emb_sum / token_count                              # [B, D]

        feat = self.text_proj(pooled)                               # [B, text_feat_dim]
        return feat

    def forward(self, image, instruction):
        image_feat = self.encode_image(image)
        text_feat = self.encode_text(instruction)

        fused = torch.cat([image_feat, text_feat], dim=1)
        fused = self.fusion(fused)

        action = self.action_head(fused)
        return action