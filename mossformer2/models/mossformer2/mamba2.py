from torch import nn

from models.mossformer2.conv_module import ConvModule
from models.mossformer2.layer_norm import CLayerNorm


def _load_mamba2():
    """Lazy import so baseline FSMN still works without Mamba dependencies."""
    try:
        from mamba_ssm import Mamba2 as mamba2_cls
    except Exception as exc:
        raise ImportError(
            "Failed to import Mamba2 from pip package `mamba-ssm`. "
            "Install `mamba-ssm`, `causal-conv1d`, `ninja`, and `packaging` before "
            "using recurrent_type='mamba2'."
        ) from exc
    return mamba2_cls


def _validate_mamba2_config(hidden_size, d_state, expand, headdim):
    """Fail early with actionable messages for invalid Mamba2 settings."""
    expanded_width = hidden_size * expand
    if expanded_width % headdim != 0:
        raise ValueError(
            "Invalid Mamba2 config: recurrent_inner_channels * mamba_expand must be divisible "
            f"by mamba_headdim, got {hidden_size} * {expand} and headdim={headdim}."
        )

    conv_channels = expanded_width + 2 * d_state
    if conv_channels % 8 != 0:
        raise ValueError(
            "Invalid Mamba2 config: expanded recurrent width + 2 * mamba_d_state must be a "
            f"multiple of 8 for the fused conv path, got {expanded_width} + 2 * {d_state} = {conv_channels}."
        )


class FFConvM(nn.Module):
    def __init__(self, dim_in, dim_out, norm_klass=nn.LayerNorm, dropout=0.1):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            ConvModule(dim_out),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mdl(x)


class Gated_Mamba2(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_size,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
    ):
        super().__init__()
        _validate_mamba2_config(hidden_size, d_state, expand, headdim)
        self.to_u = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        self.to_v = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        mamba2_cls = _load_mamba2()
        self.mamba = mamba2_cls(
            d_model=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )

    def forward(self, x):
        residual = x
        x_u = self.to_u(x)
        x_v = self.to_v(x)
        x_u = self.mamba(x_u)
        return x_v * x_u + residual


class Gated_Mamba2_Block(nn.Module):
    """Gated-Mamba2 block preserving the original FSMN block topology."""

    def __init__(
        self,
        dim,
        inner_channels=256,
        group_size=256,
        norm_type='scalenorm',
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
    ):
        super(Gated_Mamba2_Block, self).__init__()
        self.group_size = group_size

        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = CLayerNorm(inner_channels)
        self.gated_mamba = Gated_Mamba2(
            in_channels=inner_channels,
            hidden_size=inner_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):
        conv1 = self.conv1(input.transpose(2, 1))
        norm1 = self.norm1(conv1)
        seq_out = self.gated_mamba(norm1.transpose(2, 1))
        norm2 = self.norm2(seq_out.transpose(2, 1))
        conv2 = self.conv2(norm2)
        return conv2.transpose(2, 1) + input
