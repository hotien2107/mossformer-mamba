# Changelog cho tích hợp MossFormer2-Mamba2

Tài liệu này mô tả các thay đổi đã thực hiện trong code để thêm lựa chọn recurrent branch mới dùng Mamba-2. Mỗi mục gồm ba phần:

- Đoạn code cũ
- Đoạn code mới
- Ý nghĩa của thay đổi

## 0. Cập nhật bổ sung sau tích hợp ban đầu

Sau khi rà lại đường chạy training/inference thực tế, code đã được cập nhật thêm một số điểm ổn định vận hành:

- thêm preflight check trong `train.py` và `inference.py` cho `recurrent_type="mamba2"`
- fail sớm nếu môi trường không có CUDA khả dụng hoặc `mamba_ssm` import không thành công
- fail sớm nếu cấu hình `mamba_headdim` hoặc fused conv width không hợp lệ
- sửa logic `best_val_loss` trong `solver.py` để early stopping và halving bám theo best thật
- làm an toàn hơn luồng resume/fallback checkpoint khi đổi kiến trúc giữa `fsmn` và `mamba2`
- sửa điều kiện `init_checkpoint_path` để training mới không vô tình xử lý `None` như một đường dẫn checkpoint

Ý nghĩa thực tế:

- lỗi được báo gần điểm cấu hình hơn, dễ hiểu hơn
- giảm nguy cơ crash khi resume checkpoint không tương thích
- hành vi training hiện khớp hơn với kỳ vọng khi fine-tune từ baseline FSMN sang Mamba-2

## 1. `mossformer2/models/mossformer2/mossformer2_block.py`

### 1.1. Bổ sung lazy import cho Mamba-2

#### Code cũ

```python
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
```

#### Code mới

```python
import math
import os
import sys
import torch
import torch.nn.functional as F
from torch import nn, einsum


def _load_mamba2():
    """Lazy import so baseline FSMN still works without Mamba dependencies."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    local_mamba_root = os.path.join(project_root, "mamba")
    if os.path.isdir(local_mamba_root) and local_mamba_root not in sys.path:
        sys.path.insert(0, local_mamba_root)
    try:
        from mamba_ssm import Mamba2 as mamba2_cls
    except Exception as exc:
        raise ImportError(
            "Failed to import Mamba2. Install the `mamba-ssm` dependencies or make the local "
            "`mamba/` repository importable before using recurrent_type='mamba2'."
        ) from exc
    return mamba2_cls
```

#### Ý nghĩa

Trước đây file này không hề biết đến Mamba-2. Phần mới thêm một cơ chế import động:

- baseline `fsmn` vẫn chạy được ngay cả khi môi trường chưa cài `mamba-ssm`
- khi chọn `recurrent_type="mamba2"`, model mới cố import Mamba-2
- nếu dependency chưa sẵn sàng, lỗi sẽ rõ ràng và đúng nguyên nhân

Việc này giúp tránh làm hỏng workflow cũ của dự án chỉ vì thêm variant mới.

### 1.2. Thêm lõi recurrent mới `Gated_Mamba2`

#### Code cũ

```python
class Gated_FSMN_dilated(nn.Module):
    def __init__(self, in_channels, out_channels, lorder, hidden_size):
        ...
        self.to_u = FFConvM(...)
        self.to_v = FFConvM(...)
        self.fsmn = UniDeepFsmn_dilated(in_channels, out_channels, lorder, hidden_size)

    def forward(self, x):
        input = x
        x_u = self.to_u(x)
        x_v = self.to_v(x)
        x_u = self.fsmn(x_u)
        x = x_v * x_u + input
        return x
```

#### Code mới

```python
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
        self.to_u = FFConvM(...)
        self.to_v = FFConvM(...)
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
```

#### Ý nghĩa

Đây là lõi thay thế trực tiếp cho recurrent memory branch:

- vẫn giữ `to_u` và `to_v`
- vẫn giữ phép nhân gate `x_v * x_u`
- vẫn giữ residual bên trong nhánh recurrent
- chỉ thay bộ nhớ tuần tự từ `UniDeepFsmn_dilated` sang `Mamba2`

Nhờ vậy, so với bản gốc, thay đổi tập trung đúng vào memory model thay vì đổi cả topology của block.

### 1.3. Thêm block đầy đủ `Gated_Mamba2_Block`

#### Code cũ

```python
class Gated_FSMN_Block_Dilated(nn.Module):
    def __init__(self, dim, inner_channels=256, ...):
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = CLayerNorm(inner_channels)
        self.gated_fsmn = Gated_FSMN_dilated(...)
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):
        conv1 = self.conv1(input.transpose(2,1))
        norm1 = self.norm1(conv1)
        seq_out = self.gated_fsmn(norm1.transpose(2,1))
        norm2 = self.norm2(seq_out.transpose(2,1))
        conv2 = self.conv2(norm2)
        return conv2.transpose(2,1) + input
```

#### Code mới

```python
class Gated_Mamba2_Block(nn.Module):
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
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = CLayerNorm(inner_channels)
        self.gated_mamba = Gated_Mamba2(...)
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):
        conv1 = self.conv1(input.transpose(2,1))
        norm1 = self.norm1(conv1)
        seq_out = self.gated_mamba(norm1.transpose(2,1))
        norm2 = self.norm2(seq_out.transpose(2,1))
        conv2 = self.conv2(norm2)
        return conv2.transpose(2,1) + input
```

#### Ý nghĩa

Block mới này giữ nguyên toàn bộ “khung” của block cũ:

- bottleneck 1x1 conv
- `CLayerNorm`
- recurrent core
- `CLayerNorm`
- output 1x1 conv
- residual ngoài block

Điều này giúp model mới vẫn là “MossFormer2 + recurrent branch mới”, thay vì trở thành một separator khác.

### 1.4. Tổng quát hóa `MossformerBlock_GFSMN`

#### Code cũ

```python
self.fsmn = nn.ModuleList([Gated_FSMN_Block_Dilated(dim) for _ in range(depth)])

...

ii = 0
for flash in self.layers:
    x = flash(x, mask = mask)
    x = self.fsmn[ii](x)
    ii = ii + 1
return x
```

#### Code mới

```python
if recurrent_type == "fsmn":
    recurrent_blocks = [
        Gated_FSMN_Block_Dilated(...)
        for _ in range(depth)
    ]
elif recurrent_type == "mamba2":
    recurrent_blocks = [
        Gated_Mamba2_Block(...)
        for _ in range(depth)
    ]
else:
    raise ValueError(f"Unsupported recurrent_type: {recurrent_type}")

self.recurrent = nn.ModuleList(recurrent_blocks)

...

for i, flash in enumerate(self.layers):
    x = flash(x, mask = mask)
    x = self.recurrent[i](x)
return x
```

#### Ý nghĩa

Đây là thay đổi quan trọng nhất trong backbone:

- trước đây recurrent branch bị hard-code là FSMN
- bây giờ recurrent branch được chọn bằng config
- attention branch vẫn giữ nguyên

Thiết kế này cho phép:

- chạy baseline `fsmn`
- chạy variant `mamba2`
- dễ ablation và rollback

## 2. `mossformer2/models/mossformer2/mossformer2.py`

### 2.1. Propagate cấu hình recurrent từ trên xuống dưới

#### Code cũ

```python
self.mossformerM = MossformerBlock_GFSMN(
    dim=d_model,
    depth=num_blocks,
    group_size=group_size,
    query_key_dim=query_key_dim,
    expansion_factor=expansion_factor,
    causal=causal,
    attn_dropout=attn_dropout
)
```

#### Code mới

```python
self.mossformerM = MossformerBlock_GFSMN(
    dim=d_model,
    depth=num_blocks,
    group_size=group_size,
    query_key_dim=query_key_dim,
    expansion_factor=expansion_factor,
    causal=causal,
    attn_dropout=attn_dropout,
    recurrent_type=recurrent_type,
    recurrent_inner_channels=recurrent_inner_channels,
    mamba_d_state=mamba_d_state,
    mamba_d_conv=mamba_d_conv,
    mamba_expand=mamba_expand,
    mamba_headdim=mamba_headdim
)
```

#### Ý nghĩa

Thay đổi này được áp dụng xuyên suốt tại:

- `MossFormerM`
- `Computation_Block`
- `MossFormer_MaskNet`
- `MossFormer`
- `MossFormer2_SS`

Ý nghĩa là:

- cấu hình recurrent giờ được truyền từ `args` xuống tận block thật sự dùng recurrent
- người dùng không cần sửa code để đổi giữa `fsmn` và `mamba2`
- toàn bộ lựa chọn kiến trúc được đưa về cấu hình

### 2.2. Thêm cấu hình từ `args` khi khởi tạo model ngoài cùng

#### Code cũ

```python
self.model = MossFormer(
    in_channels=args.encoder_embedding_dim,
    out_channels=args.mossformer_sequence_dim,
    num_blocks=args.num_mossformer_layer,
    kernel_size=args.encoder_kernel_size,
    norm="ln",
    num_spks=args.num_spks,
    skip_around_intra=True,
    use_global_pos_enc=True,
    max_length=20000)
```

#### Code mới

```python
self.model = MossFormer(
    in_channels=args.encoder_embedding_dim,
    out_channels=args.mossformer_sequence_dim,
    num_blocks=args.num_mossformer_layer,
    kernel_size=args.encoder_kernel_size,
    norm="ln",
    num_spks=args.num_spks,
    skip_around_intra=True,
    use_global_pos_enc=True,
    max_length=20000,
    recurrent_type=args.recurrent_type,
    recurrent_inner_channels=args.recurrent_inner_channels,
    mamba_d_state=args.mamba_d_state,
    mamba_d_conv=args.mamba_d_conv,
    mamba_expand=args.mamba_expand,
    mamba_headdim=args.mamba_headdim)
```

#### Ý nghĩa

Trước thay đổi này, dù block dưới có hỗ trợ Mamba, model ngoài cùng cũng không thể bật được vì `args` chưa được nối xuống. Phần mới chính là “đầu vào cấu hình” của toàn bộ refactor.

## 3. `mossformer2/train.py`

### 3.1. Thêm CLI args cho recurrent branch

#### Code cũ

```python
parser.add_argument('--mossformer-squence-dim', ...)
parser.add_argument('--num-mossformer_layer', ...)
```

#### Code mới

```python
parser.add_argument('--recurrent-type', dest='recurrent_type', type=str, default='fsmn',
                    help='recurrent block type: fsmn or mamba2')
parser.add_argument('--recurrent-inner-channels', dest='recurrent_inner_channels', type=int, default=256,
                    help='bottleneck channels used by the recurrent branch')
parser.add_argument('--mamba-d-state', dest='mamba_d_state', type=int, default=64,
                    help='Mamba2 state size when recurrent_type=mamba2')
parser.add_argument('--mamba-d-conv', dest='mamba_d_conv', type=int, default=4,
                    help='Mamba2 local convolution width when recurrent_type=mamba2')
parser.add_argument('--mamba-expand', dest='mamba_expand', type=int, default=2,
                    help='Mamba2 expansion factor when recurrent_type=mamba2')
parser.add_argument('--mamba-headdim', dest='mamba_headdim', type=int, default=64,
                    help='Mamba2 head dimension; should divide the expanded recurrent width')
```

#### Ý nghĩa

Training giờ có thể được điều khiển hoàn toàn từ config/CLI:

- vẫn chạy baseline như cũ vì default là `fsmn`
- có thể bật variant mới bằng `--recurrent-type mamba2`

## 4. `mossformer2/inference.py`

### 4.1. Đồng bộ parser cho suy luận

#### Code cũ

`inference.py` chỉ có các tham số MossFormer cũ như `num_spks`, `encoder_embedding_dim`, `num_mossformer_layer`.

#### Code mới

```python
parser.add_argument('--recurrent-type', dest='recurrent_type', type=str, default='fsmn', ...)
parser.add_argument('--recurrent-inner-channels', dest='recurrent_inner_channels', type=int, default=256, ...)
parser.add_argument('--mamba-d-state', dest='mamba_d_state', type=int, default=64, ...)
parser.add_argument('--mamba-d-conv', dest='mamba_d_conv', type=int, default=4, ...)
parser.add_argument('--mamba-expand', dest='mamba_expand', type=int, default=2, ...)
parser.add_argument('--mamba-headdim', dest='mamba_headdim', type=int, default=64, ...)
```

#### Ý nghĩa

Train và inference giờ dùng cùng một hệ config. Điều này rất quan trọng vì:

- checkpoint train bằng `mamba2` phải được load lại bằng đúng recurrent config đó
- tránh lỗi do inference vẫn khởi tạo baseline trong khi checkpoint là variant mới

## 5. Các file YAML train/inference

### 5.1. Thêm khóa config mới

#### Code cũ

```yaml
num_spks: 2
encoder_kernel_size: 16
encoder_embedding_dim: 512
mossformer_sequence_dim: 512
num_mossformer_layer: 24
```

#### Code mới

```yaml
num_spks: 2
encoder_kernel_size: 16
encoder_embedding_dim: 512
mossformer_sequence_dim: 512
num_mossformer_layer: 24
recurrent_type: "fsmn"
recurrent_inner_channels: 256
mamba_d_state: 64
mamba_d_conv: 4
mamba_expand: 2
mamba_headdim: 64
```

#### Ý nghĩa

Việc thêm các khóa này vào YAML giúp:

- baseline vẫn chạy mặc định mà không đổi hành vi
- có một nơi chuẩn để chuyển sang `mamba2`
- tránh hard-code tham số trong model constructor

## 6. Kiểm tra đã thực hiện

Đã chạy kiểm tra cú pháp cho các file Python đã sửa:

- `mossformer2/models/mossformer2/mossformer2_block.py`
- `mossformer2/models/mossformer2/mossformer2.py`
- `mossformer2/train.py`
- `mossformer2/inference.py`

Kết quả:

- `python3 -m py_compile` chạy thành công cho cả 4 file

Lưu ý:

- chưa chạy forward test thật vì môi trường hiện tại không có sẵn `torch` để khởi tạo model
- phần import Mamba được thiết kế để báo lỗi rõ khi bạn thực sự bật `recurrent_type=mamba2` mà dependency chưa sẵn sàng
