from torch import nn
import torch.nn.functional as F

from utils.utils import _make_divisible

class h_sigmoid(nn.Module):
    """
    Hard Sigmoid activation: ReLU6(x + 3) / 6
    """

    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    """
    Hard Swish activation: x * HardSigmoid(x)
    """
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
    
class Mlp(nn.Module):
    """
    Simple feedforward MLP block with optional dropout and custom activation.

    Args:
        in_features (int): Input feature dimension.
        hidden_features (int, optional): Hidden layer dimension. Defaults to in_features.
        out_features (int, optional): Output feature dimension. Defaults to in_features.
        act_layer (nn.Module): Activation layer. Defaults to GELU.
        drop (float): Dropout rate.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CTEBlock(nn.Module):
    """
    Convolutional Token Embedding block.

    Converts input images into patch-like tokens using convolution, normalization,
    and pooling operations. Outputs flattened token embeddings along with their
    spatial dimensions for use in attention blocks.

    Args:
        in_channels (int): Number of input image channels.
        out_channels (int): Channels after initial convolution.
        kernel_size (int): Kernel size for the initial convolution.
        mp_kernel_size (int): Kernel size for max pooling.
        conv_stride (int): Stride for the initial convolution.
        mp_stride (int): Stride for max pooling.
        emb_dim (int): Output embedding dimension for each token.

    Inputs:
        x (Tensor): Input image tensor of shape (B, in_channels, H, W).

    Returns:
        x (Tensor): Flattened token embeddings of shape (B, N, emb_dim), where N = H' x W'.
        H (int): Spatial height of token map after conv and pooling.
        W (int): Spatial width of token map after conv and pooling.
    """

    def __init__(self, out_channels=64, emb_dim=128, in_channels=3, kernel_size=3, 
                 mp_kernel_size=2,conv_stride=1, mp_stride=2):
        super(CTEBlock, self).__init__()

        # Conv op
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size=kernel_size, 
                              stride=conv_stride,
                              padding=kernel_size // 2, 
                              bias=False)
        
        # BatchNorm
        self.bn = nn.BatchNorm2d(out_channels)

        # ReLU
        self.relu = nn.ReLU(inplace=True)
        
        # Maxpool
        self.maxpool = nn.MaxPool2d(
            kernel_size=mp_kernel_size, 
            stride=mp_stride, 
            padding=0)

        # Point-wise conv
        self.pw_conv = nn.Conv2d(out_channels, emb_dim, kernel_size=1, 
                                 stride=1, padding=0, groups=1, bias=False)
        
        self.proj = nn.Conv2d(64, 128, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Tokens set to have emb_dim
        # x = self.pw_conv(x)

        x = self.proj(x)

        # For maintaining consistency in the CPSA block
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).permute(0, 2, 1)
        return x, H, W
    
class SELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) block.

    Calculates importance of individual channels using global average pooling followed by 
    a lightweight MLP.

    Args:
        channel (int): Number of input channels.
        reduction (int): Reduction ratio for the intermediate hidden layer. Default is 4.

    Inputs:
        x (Tensor): Input tensor of shape (B, C, H, W).

    Returns:
        Tensor: Output tensor of shape (B, C, H, W), with channel-wise recalibration applied.
    """

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        assert len(x.size()) == 4, "For S&E, input must be of the shape (B, C, H, W)"
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class LFFNBlock(nn.Module):
    """
    Local Feed Forward Network (LFFN) Block.

    Enhances local spatial representations using depthwise and pointwise convolutions, 
    squeeze-and-excitation mechanisms, and non-linearities. Optionally includes a residual 
    connection.

    Args:
        in_channels (int): Number of input channels.
        kernel_size (int): Size of the depthwise convolution kernel. Defaults to 3.
        expansion_ratio (int): Factor by which the hidden channels are expanded. Defaults to 2.
        use_res_connect (bool): If True, adds a residual connection. Defaults to True.

    Inputs:
        x (Tensor): Input tensor of shape (B, N, in_channels), where N = H x W.
        H (int): Height of the input feature grid.
        W (int): Width of the input feature grid.

    Returns:
        Tensor: Output tensor of shape (B, N, in_channels), same as input shape.
    """

    def __init__(self, in_channels, kernel_size =3, expansion_ratio=2, use_res_connect=True):
        super(LFFNBlock, self).__init__()
        self.kernel_size = kernel_size
        self.expansion_ratio = expansion_ratio
        self.use_res_connection = use_res_connect
        self.in_channels = in_channels
        self.hidden_dim = in_channels * expansion_ratio

        # Depthwise conv layers
        self.dw_conv_regular = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=self.kernel_size, 
                            stride=1, padding=1, groups=self.in_channels)
        self.dw_conv_expanded = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=self.kernel_size, 
                            stride=1, padding=1, groups=self.hidden_dim)
        
        # Pointwise conv layers
        self.expand_pw_conv = nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size=1, 
                                 stride=1, padding=0, groups=1, bias=False)
        self.reduce_pw_conv =  nn.Conv2d(self.hidden_dim, self.in_channels, kernel_size=1, 
                                 stride=1, padding=0, groups=1, bias=False)
        
        # Squeeze and excite layers
        self.se_layer_regular = SELayer(self.in_channels)
        self.se_layer_expanded = SELayer(self.hidden_dim)

        # Swish non-linearity
        self.swish = h_swish()

        
        # DW_Conv -> Swish -> S&E(e x C)
        self.dw_conv_block_expanded = nn.Sequential(
            self.dw_conv_expanded,
            self.swish,
            self.se_layer_expanded
        )

        # DW_Conv -> Swish -> S&E(C)
        self.dw_conv_block_regular = nn.Sequential(
            self.dw_conv_regular,
            self.swish,
            self.se_layer_regular
        )

    # X has the shape B x N x C
    def forward(self, x, H, W):
        B, N, C = x.shape
        
        x_inp = None
        if self.use_res_connection:
            x_inp = x

        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        x = self.dw_conv_block_regular(x)
        x = self.expand_pw_conv(x)
        x = self.dw_conv_block_expanded(x)
        x = self.reduce_pw_conv(x)
        x = self.dw_conv_block_regular(x)

        x = x.permute(0, 2, 3, 1).view(B, N, C)
        
        if self.use_res_connection:
            return x + x_inp    
        else:
            return x

class CPSABlock(nn.Module):
    """
    Convolutional Parameter Sharing Multi-Head Attention (CPSA) Block.

    Performs efficient attention by computing full-resolution queries and 
    cross-scale keys/values derived via depthwise convolution.

    Args:
        dim (int): Input embedding dimension.
        num_heads (int): Number of attention heads.
        qs_bias (bool): Whether to include bias in Q and S projections.
        qk_scale (float, optional): Scaling factor for attention logits.
        attn_drop (float): Dropout rate applied to attention weights.
        proj_drop (float): Dropout rate applied after the output projection.

    Inputs:
        x (Tensor): Input tensor of shape (B, N, dim), where N = H x W.
        H (int): Height of the input feature grid.
        W (int): Width of the input feature grid.

    Returns:
        Tensor: Output tensor of shape (B, N, dim), same as input shape.
    """

    def __init__(self, dim=128, num_heads=2, qs_bias=False, 
                 qk_scale=None, attn_drop=0., proj_drop=0.,):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # Projects tokens to Q and S 
        self.q = nn.Linear(dim, dim, bias=qs_bias)

        # Shared Key and Value projections S
        self.s = nn.Linear(dim, dim, bias=qs_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm = nn.LayerNorm(dim)
        
        self.dw_conv_regular = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, 
                stride=2, padding=1, groups=self.dim))
        

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        s = self.s(x).view(B, H, W, C).permute(0, 3, 1, 2)    
        
        s = self.dw_conv_regular(s)
        
        # H and W get downsampled following the depth-wise convolution operation
        _, _, H_D, W_D = s.shape

        s = s.view(B, C, H_D * W_D).permute(0, 2, 1)
        s = s.reshape(B, H_D * W_D, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # attn = (q @ s.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        attn_output = F.scaled_dot_product_attention(q, s, s, dropout_p=self.attn_drop, is_causal=False)

        x = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x + self.norm(x)