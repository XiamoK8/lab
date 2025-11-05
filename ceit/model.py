import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
        output = x.div(keep_prob) * random_tensor
        return output

class Image2Tokens(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 out_channels=64, 
                 kernel_size=7, 
                 stride=2):
        super(Image2Tokens, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=kernel_size // 2)
        self.BN = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class HybridEmbed(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_channels=3, 
                 embed_dim=768,
                 i2t_out_channels=64, 
                 i2t_kernel=7, 
                 i2t_stride=2):
        super().__init__()
        self.i2t = Image2Tokens(in_channels=in_channels, 
                                out_channels=i2t_out_channels, 
                                kernel_size=i2t_kernel, 
                                stride=i2t_stride)
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        conv_out_size = (
            (img_size[0] - i2t_kernel + 2 * (i2t_kernel // 2)) // i2t_stride + 1,
            (img_size[1] - i2t_kernel + 2 * (i2t_kernel // 2)) // i2t_stride + 1
            )
        self.feature_size = (
            (conv_out_size[0] - 3 + 2 * 1) // i2t_stride + 1,
            (conv_out_size[1] - 3 + 2 * 1) // i2t_stride + 1
            )
        self.feature_dim = i2t_out_channels

        self.num_patches = (self.feature_size[0] // patch_size[0]) * (self.feature_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_channels=self.feature_dim, 
                              out_channels=embed_dim, 
                              kernel_size=patch_size, 
                              stride=patch_size)

    def forward(self, x):
        x = self.i2t(x)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class SequencePreprocessor(nn.Module):
    def __init__(self, embed_dim, num_patches, drop_rate=0.):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x):
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads=8, 
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop_rate=0.,
                 proj_drop_rate=0.):
        super().__init__()
        self.num_heads = num_heads
        per_head_dim = dim // num_heads
        self.scale = qk_scale or per_head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttentionLCA(Attention):
    def __init__(self, 
                 dim, 
                 num_heads=8, 
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop_rate=0., 
                 proj_drop_rate=0.):
        super().__init__(dim, 
                         num_heads, 
                         qkv_bias, 
                         qk_scale, 
                         attn_drop_rate, 
                         proj_drop_rate)
        self.dim = dim
        self.qkv_bias = qkv_bias

    def forward(self, x):
        B, N, C = x.shape
        _, last_token = torch.split(x, [N - 1, 1], dim=1)

        q_weight = self.qkv.weight[:self.dim, :]
        q_bias = None if not self.qkv_bias else self.qkv.bias[:self.dim]
        kv_weight = self.qkv.weight[self.dim:, :]
        kv_bias = None if not self.qkv_bias else self.qkv.bias[self.dim:]

        q = F.linear(last_token, q_weight, q_bias).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = F.linear(x, kv_weight, kv_bias).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_ = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_ = self.proj(x_)
        x_ = self.proj_drop(x_)
        return x_

class MLP(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop_rate=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LocallyEnhancedFeedForward(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop_rate=0.,
                 kernel_size=3, 
                 with_bn=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Conv2d(in_channels=in_features, 
                               out_channels=hidden_features, 
                               kernel_size=1, 
                               stride=1, 
                               padding=0)
        self.conv2 = nn.Conv2d(in_channels=hidden_features, 
                               out_channels=hidden_features, 
                               kernel_size=kernel_size, 
                               stride=1, 
                               padding=(kernel_size - 1) // 2, 
                               groups=hidden_features)
        self.conv3 = nn.Conv2d(in_channels=hidden_features, 
                               out_channels=out_features, 
                               kernel_size=1, 
                               stride=1, 
                               padding=0)
        self.act = act_layer()
        self.with_bn = with_bn
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(hidden_features)
            self.bn2 = nn.BatchNorm2d(hidden_features)
            self.bn3 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        B, N, C = x.size()
        cls_token, tokens = torch.split(x, [1, N - 1], dim=1)
        x = tokens.reshape(B, int(math.sqrt(N - 1)), int(math.sqrt(N - 1)), C).permute(0, 3, 1, 2)
        if self.with_bn:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act(x)
            x = self.conv3(x)
            x = self.bn3(x)
        else:
            x = self.conv1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.act(x)
            x = self.conv3(x)

        tokens = x.flatten(2).permute(0, 2, 1)
        out = torch.cat((cls_token, tokens), dim=1)
        return out

class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 mlp_ratio, 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.,
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,
                 kernel_size=3, 
                 with_bn=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_prob=drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.attn = Attention(dim, 
                                num_heads=num_heads, 
                                qkv_bias=qkv_bias, 
                                qk_scale=qk_scale,
                                attn_drop_rate=attn_drop_rate, 
                                proj_drop_rate=drop_rate)
        self.leff = LocallyEnhancedFeedForward(in_features=dim, 
                                                hidden_features=int(dim * mlp_ratio), 
                                                act_layer=act_layer, 
                                                drop_rate=drop_rate, 
                                                kernel_size=kernel_size, 
                                                with_bn=with_bn)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.leff(self.norm2(x)))
        return x, x[:, 0]


class LCA(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 mlp_ratio, 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.,
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_prob=drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.attn_lca = AttentionLCA(dim=dim, 
                                     num_heads=num_heads, 
                                     qkv_bias=qkv_bias, 
                                     qk_scale=qk_scale, 
                                     attn_drop_rate=attn_drop_rate, 
                                     proj_drop_rate=drop_rate)
        self.ff = MLP(in_features=dim, 
                      hidden_features=int(dim * mlp_ratio), 
                      act_layer=act_layer, 
                      drop_rate=drop_rate)

    def forward(self, x):
        _, last_token = torch.split(x, [x.size(1) - 1, 1], dim=1)
        x = last_token + self.drop_path(self.attn_lca(self.norm1(x)))
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


class CeiT(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.):
        super().__init__()
        self.i2t = HybridEmbed(img_size=img_size, 
                               patch_size=patch_size, 
                               in_channels=in_channels, 
                               embed_dim=embed_dim)
        self.SeqPrep = SequencePreprocessor(embed_dim=embed_dim, 
                                            num_patches=self.i2t.num_patches, 
                                            drop_rate=drop_rate)
        self.pos_layer_embed = nn.Parameter(torch.zeros(1, depth, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, 
                  num_heads=num_heads, 
                  mlp_ratio=mlp_ratio, 
                  qkv_bias=qkv_bias, 
                  qk_scale=qk_scale,
                  drop_rate=drop_rate, 
                  attn_drop_rate=attn_drop_rate, 
                  drop_path_rate=dpr[i])
            for i in range(depth)])
        self.lca = LCA(dim=embed_dim, 
                       num_heads=num_heads, 
                       mlp_ratio=mlp_ratio, 
                       qkv_bias=qkv_bias, 
                       qk_scale=qk_scale, 
                       drop_rate=drop_rate, 
                       attn_drop_rate=attn_drop_rate, 
                       drop_path_rate=0.)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        nn.init.trunc_normal_(self.SeqPrep.cls_token, std=0.02)
        nn.init.trunc_normal_(self.SeqPrep.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_layer_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.i2t(x)

        x = self.SeqPrep(x)

        cls_token_list = []
        for block in self.blocks:
            x, curr_cls_token = block(x)
            cls_token_list.append(curr_cls_token)
            
        all_cls_token = torch.stack(cls_token_list, dim=1)
        all_cls_token = all_cls_token + self.pos_layer_embed
        last_cls_token = self.lca(all_cls_token)

        x = self.norm(last_cls_token)

        return x.flatten(1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x