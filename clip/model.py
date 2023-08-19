from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    # inplane : 输入通道数， plane ： 输出通道数, stride : 步长
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1

        # conv1 : 输入通道为inplane, 输出为plane, kernel_size = 1, bias = false
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # inplace为true对tensor的更新在原地进行
        self.relu1 = nn.ReLU(inplace=True)

        # kernel size = 3
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        # AvgPool : 对给定窗口内执行一次平均操作，在该例子中stride传递给AvgPool2d中的kernel_size
        # 如果stride > 1, 则进行一次特征平均
        # or进行Identity，即stide = 1直接将原始输入输出
        """
        
        由于没有指定stride，所以stride = kernel size，这样设置默认参数可以使得窗口在滑动过程中不重合
        
        """
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        """
        
        这段代码的意思是，如果stride > 1或者inplanes != planes * Bottleneck.expansion，则会创建一个下采样层(self.downsample)。
        下采样层由一个平均池化层(nn.AvgPool2d)和一个步长为1的卷积层(nn.Conv2d)组成。

        具体来说，下采样层的定义如下：
        - nn.AvgPool2d(stride)：使用步长为stride的平均池化层对输入进行下采样。
        - nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)：使用1x1的卷积核对输入进行卷积，将输入通道数从inplanes变换为planes * self.expansion。
        - nn.BatchNorm2d(planes * self.expansion)：对卷积输出进行批归一化。
        
        """
        
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        """
        x
        conv1(x) : 把x的通道从inplane变成plane，h和w不变
        bn(conv1(x)) : 在conv1(x)的每个通道上都进行归一化
        relu1(bn1(conv1(x))) : 将得到的结果原地执行relu操作，得到一个大小不变，但是 <0 的部分变为0， >0部分不变的矩阵

        同理conv2(out)，kernel_size = 3, padding = 1, stride = 1, h = (input_h + 2 * padding - kernel_size)/stride + 1
        即h和w不变，plane不变，进一步提取特征

        经过一个avgpool函数，如果stride = 1，直接把feature输出，
        如果stride > 1, 则把大小为stride * stride窗口内的值取平均并使得h和w变为1 / stride
        
        之后经过一个conv3，大小不变，通道变为planes * expansion

        """
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):

    """
    
    该代码定义了一个注意力池化操作
    
    """

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):

        """
        
        spacial_dim : input feature的宽度和高度
        embed_dim : feature embedding的维度
        num_head : attn heads的数量
        output dim : 输出维度


        forward
        x NCHW
        x.flatten(start_dim=2) : 把NCHW展平为NC(HW)
        permute(2, 0, 1) : 将维度换成2, 0, 1. 如原来shape为(a, b, c), permute(1, 0, 2)后为(b, a, c)
        则x.flatten.permute最后的结果为(HW)NC

        x.mean(dim=0, keepdim=true) : 作用是将x沿着dim=0计算均值，如x.shape = (2, 3, 4),则结果为x_mean.shape = (1, 3, 4)，将dim0压缩为1
        然后将[x.mean, x]cat起来，按行进行拼接（即讲x.mean作为一行添加到x的顶部，因为x_mean.shape=[]），最终结果应该为
        [x.mean,
         x      ]

        以x.shape = (2, 3, 10, 10)为例：
        x.flatten(s_d=2), x.shape = (2, 3, 100)
        x.permute(2, 0, 1), x.shape = (100, 2, 3)

        x.mean(dim=0, keepdim=true), x.shape = (1, 2, 3)
        由于x_mean的dim[0]=1，因此可以把x_mean当作一行输入到x的顶部
        x.cat(x_mean, x), x.shape = (1 + 100, 2, 3)

        """
        super().__init__()

        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):

        """
        x NCHW
        x.flatten(start_dim=2) : 把NCHW展平为NC(HW)
        permute(2, 0, 1) : 将维度换成2, 0, 1. 如原来shape为(a, b, c), permute(1, 0, 2)后为(b, a, c)
        则x.flatten.permute最后的结果为(HW)NC

        x.mean(dim=0, keepdim=true) : 作用是将x沿着dim=0计算均值，如x.shape = (2, 3, 4),则结果为x_mean.shape = (1, 3, 4)，将dim0压缩为1
        然后将[x.mean, x]cat起来，按行进行拼接（即讲x.mean作为一行添加到x的顶部，因为x_mean.shape=[]），最终结果应该为
        [x.mean,
         x      ]

        以x.shape = (2, 3, 10, 10)为例：
        x.flatten(s_d=2), x.shape = (2, 3, 100)
        x.permute(2, 0, 1), x.shape = (100, 2, 3)

        x.mean(dim=0, keepdim=true), x.shape = (1, 2, 3)
        由于x_mean的dim[0]=1，因此可以把x_mean当作一行输入到x的顶部
        x.cat(x_mean, x), x.shape = (1 + 100, 2, 3)

        x + self.positional_embedding[:, None, :]
        
        """

        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        """将x.mean和x拼接在一起，dim=0表示沿着第一个维度（行）进行拼接，dim=1表示沿着第二个维度（列）拼接"""
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        """
        为x添加位置编码
        """
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC

        """
        query = x[:1],表示对x的第一维度进行切片，切片范围为（0）：1(不包含1)，即只取x的第一维度的第一个元素
        若对第二维度切片，如x[:, 2:3, :], 则表示去第二维度的下标2分片
        此query为x的N个图像的c个通道沿着各个像素点求平均值后的均值

        embed_dim_to_check : x的通道数
        q_proj是Linear(embed_dim, embed_dim), embeddim -> embeddim
        q_proj.weight是生成一个embed_dim * embed_dim大小的随机矩阵，若linear（a， b),则矩阵维度为(b, a),因为映射操作为 Q_RPOJ*q
        """
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        """
        inchannel = 3 
        outchannel = width // 2， //是整除操作，结果是width / 2的整数部分
        stride = 2, kernel_size = 3, padding = 1, h = (h_i + 2 * padding - kernel_size) / stride + 1 = (h_i - 1)/2 + 1
        conv1的操作是把 3 * h * w的图像大小变为 w//2 * (h - 1)/2+1 * (w - 1)/2+1

        conv2操作是通道数不变，h和w均不变，但是进一步提取特征

        conv3把通道数翻倍，hw不变

        最后avgpool在2 * 2的窗口上取平均，并且每个窗口不重合，是的hw均减半
        """
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        """

        layers为一个列表，包含四个值，每个值表示对应layer的bottleneck block数目
        layer1 : 输入通道数 = width， 经过layer[0]个bottlenecks
        layer2, 3, 4同理

        特征维度为width * 32
        """
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        """
        
        layers是一个bottleneck层，他把inplanes通道映射到planes通道，这个过程会提取其中的特征
        定义完layers后，讲inplanes变为planes * expansion
        根据blocks，为layers添加相应数量的bottleneck块
        最后返回layers组成的神经网络模型

        """
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        """
        可以使用nn.Sequential定义神经网络模型
        
        如：
        model = nn.Sequential(
            nn.Linear(784, 256),  # 全连接层：输入784维，输出256维
            nn.ReLU(),            # ReLU激活函数
            nn.Linear(256, 10),   # 全连接层：输入256维，输出10维
            nn.Softmax(dim=1)     # Softmax函数，用于多分类问题
        )

        """
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

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

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)      
        # vocab_size表示词汇表中不同单词的数量，每个单词通过一个整数索引表示
        # transformer_width : 表示嵌入向量维度或嵌入空间大小
        # 使用nn.Embedding, v_b指定词汇表大小，transformer_width指定嵌入向量维度
        # nn.Embedding创建一个v_b * t_w大小的矩阵
        # 如v_b = 10000, t_w = 256, nn.Embedding(10000, 256)表示创建一个10000 * 256个参数矩阵，把10000个不同单词映射到256维空间
        
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
