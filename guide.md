# Hướng dẫn sử dụng MossFormer2-Mamba2

Tài liệu này hướng dẫn cách cấu hình và chạy biến thể mới của MossFormer2 có hỗ trợ hai loại recurrent branch:

- `fsmn`: baseline gốc
- `mamba2`: biến thể mới dùng Mamba-2

## 1. Ý tưởng mô hình

Model mới giữ nguyên:

- encoder/decoder
- attention branch `FLASH_ShareA_FFConvM`
- mask head
- output interface

Model chỉ thay recurrent branch bên trong `MossformerBlock_GFSMN`.

Điều đó có nghĩa:

- nếu `recurrent_type=fsmn`, hành vi gần như giữ nguyên baseline
- nếu `recurrent_type=mamba2`, recurrent branch sẽ dùng `Gated_Mamba2_Block`

## 2. Các tham số cấu hình mới

### 2.1. `recurrent_type`

Giá trị hợp lệ:

- `fsmn`
- `mamba2`

Ý nghĩa:

- chọn loại recurrent branch cho từng block

Khuyến nghị:

- dùng `fsmn` nếu muốn giữ baseline
- dùng `mamba2` khi muốn thử biến thể mới

### 2.2. `recurrent_inner_channels`

Ý nghĩa:

- số kênh bottleneck ở recurrent block
- đây là số chiều ẩn sau `conv1` trước khi đi vào recurrent core

Ảnh hưởng:

- tăng giá trị này sẽ tăng năng lực mô hình
- đồng thời tăng tham số và chi phí tính toán

Khuyến nghị mặc định:

- `256`

### 2.3. `mamba_d_state`

Ý nghĩa:

- state size của Mamba-2

Ảnh hưởng:

- tăng `d_state` có thể tăng khả năng modeling temporal memory
- nhưng cũng tăng chi phí tính toán và memory

Khuyến nghị mặc định:

- `64`

Khuyến nghị khi mở rộng:

- thử `128` sau khi bản `64` đã chạy ổn định

### 2.4. `mamba_d_conv`

Ý nghĩa:

- local convolution width bên trong Mamba-2

Khuyến nghị mặc định:

- `4`

Ghi chú:

- đây là setting phổ biến trong Mamba
- thường không cần thay đổi ở vòng thử nghiệm đầu tiên

### 2.5. `mamba_expand`

Ý nghĩa:

- hệ số mở rộng chiều ẩn trong Mamba-2

Ảnh hưởng:

- tăng `expand` sẽ tăng khả năng biểu diễn
- đồng thời tăng số tham số

Khuyến nghị mặc định:

- `2`

### 2.6. `mamba_headdim`

Ý nghĩa:

- head dimension bên trong Mamba-2

Ràng buộc:

- phải chia hết kích thước expanded recurrent width

Với bộ mặc định:

- `recurrent_inner_channels = 256`
- `mamba_expand = 2`
- expanded width xấp xỉ `512`
- `mamba_headdim = 64` là hợp lệ

Khuyến nghị mặc định:

- `64`

## 3. Bộ tham số chuẩn nên dùng

### 3.1. Baseline gốc

```yaml
recurrent_type: "fsmn"
recurrent_inner_channels: 256
mamba_d_state: 64
mamba_d_conv: 4
mamba_expand: 2
mamba_headdim: 64
```

Giải thích:

- các tham số Mamba vẫn có thể tồn tại trong config nhưng sẽ không được dùng khi `recurrent_type=fsmn`

### 3.2. Bộ tham số chuẩn cho MossFormer2-Mamba2

```yaml
recurrent_type: "mamba2"
recurrent_inner_channels: 256
mamba_d_state: 64
mamba_d_conv: 4
mamba_expand: 2
mamba_headdim: 64
```

Đây là preset khuyến nghị cho vòng chạy đầu tiên vì:

- gần nhất với thiết kế trong `plan.md`
- đủ an toàn để so sánh với baseline
- ít rủi ro hơn các cấu hình quá lớn

### 3.3. Bộ tham số mở rộng để ablation

Sau khi preset chuẩn chạy ổn định, có thể thử:

```yaml
recurrent_type: "mamba2"
recurrent_inner_channels: 256
mamba_d_state: 128
mamba_d_conv: 4
mamba_expand: 2
mamba_headdim: 64
```

hoặc:

```yaml
recurrent_type: "mamba2"
recurrent_inner_channels: 384
mamba_d_state: 64
mamba_d_conv: 4
mamba_expand: 2
mamba_headdim: 64
```

Khuyến nghị:

- chỉ thay một tham số mỗi lần
- luôn giữ baseline `fsmn` để đối chiếu

## 4. Cách sửa config

Các file config đã được mở rộng:

- `mossformer2/config/train/MossFormer2_SS_16K.yaml`
- `mossformer2/config/train/MossFormer2_SS_8K.yaml`
- `mossformer2/config/inference/MossFormer2_SS_16K.yaml`
- `mossformer2/config/inference/MossFormer2_SS_8K.yaml`

Để chạy variant mới, chỉ cần đổi:

```yaml
recurrent_type: "mamba2"
```

và giữ preset chuẩn:

```yaml
recurrent_inner_channels: 256
mamba_d_state: 64
mamba_d_conv: 4
mamba_expand: 2
mamba_headdim: 64
```

## 5. Yêu cầu dependency

Nếu muốn dùng `recurrent_type=mamba2`, bạn cần môi trường có Mamba-2 hoạt động được.

Ưu tiên:

1. cài `mamba-ssm`
2. có CUDA/Triton tương thích
3. có `torch` phù hợp với bản Mamba dùng

Lưu ý:

- nếu dependency chưa sẵn sàng, code sẽ báo lỗi rõ khi khởi tạo model với `recurrent_type=mamba2`
- nếu bạn vẫn dùng `recurrent_type=fsmn`, baseline vẫn không phụ thuộc vào Mamba-2

## 6. Cách chạy training

### 6.1. Chạy baseline FSMN

Giữ config:

```yaml
recurrent_type: "fsmn"
```

Sau đó chạy:

```bash
cd mossformer2
python3 train.py --config config/train/MossFormer2_SS_16K.yaml
```

hoặc với 8 kHz:

```bash
cd mossformer2
python3 train.py --config config/train/MossFormer2_SS_8K.yaml
```

### 6.2. Chạy training với Mamba-2

Sửa config train:

```yaml
recurrent_type: "mamba2"
recurrent_inner_channels: 256
mamba_d_state: 64
mamba_d_conv: 4
mamba_expand: 2
mamba_headdim: 64
```

Sau đó chạy:

```bash
cd mossformer2
python3 train.py --config config/train/MossFormer2_SS_16K.yaml
```

Bạn cũng có thể override trực tiếp từ CLI:

```bash
cd mossformer2
python3 train.py \
  --config config/train/MossFormer2_SS_16K.yaml \
  --recurrent-type mamba2 \
  --recurrent-inner-channels 256 \
  --mamba-d-state 64 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --mamba-headdim 64
```

## 7. Cách fine-tuning

Fine-tuning có hai tình huống chính.

### 7.1. Fine-tuning trên cùng một kiến trúc

Ví dụ:

- checkpoint cũ cũng là `mamba2`
- config hiện tại cũng là `mamba2` với cùng tham số recurrent

Khi đó:

1. giữ nguyên recurrent config
2. đặt `init_checkpoint_path` đến checkpoint muốn nạp
3. dùng `finetune_learning_rate`

Ví dụ:

```bash
cd mossformer2
python3 train.py \
  --config config/train/MossFormer2_SS_16K.yaml \
  --recurrent-type mamba2 \
  --init_checkpoint_path /path/to/checkpoint.pt
```

### 7.2. Fine-tuning từ baseline FSMN sang Mamba-2

Trường hợp này có thể dùng để tận dụng encoder, decoder, attention branch và các lớp tương thích, nhưng recurrent branch mới sẽ không khớp hoàn toàn checkpoint cũ.

Cách làm:

1. đổi `recurrent_type` sang `mamba2`
2. đặt `init_checkpoint_path` tới checkpoint baseline
3. chấp nhận việc recurrent branch mới không load đầy đủ

Khuyến nghị:

- theo dõi log `missing keys`
- dùng `finetune_learning_rate` nhỏ hơn training from scratch

Ví dụ:

```bash
cd mossformer2
python3 train.py \
  --config config/train/MossFormer2_SS_16K.yaml \
  --recurrent-type mamba2 \
  --init_checkpoint_path /path/to/fsmn_baseline.pt \
  --finetune_learning_rate 5e-5
```

## 8. Cách chạy inference

### 8.1. Inference cho checkpoint baseline FSMN

```bash
cd mossformer2
python3 inference.py --config config/inference/MossFormer2_SS_16K.yaml
```

Miễn là config có:

```yaml
recurrent_type: "fsmn"
```

### 8.2. Inference cho checkpoint Mamba-2

Config inference phải khớp với lúc train:

```yaml
recurrent_type: "mamba2"
recurrent_inner_channels: 256
mamba_d_state: 64
mamba_d_conv: 4
mamba_expand: 2
mamba_headdim: 64
```

Sau đó chạy:

```bash
cd mossformer2
python3 inference.py --config config/inference/MossFormer2_SS_16K.yaml
```

Nếu config inference không khớp recurrent branch của checkpoint, model có thể không load đúng.

## 9. Quy trình khuyến nghị để thử nghiệm

1. Chạy baseline `fsmn` để có mốc so sánh.
2. Chuyển sang `recurrent_type=mamba2` với bộ tham số chuẩn.
3. So sánh chất lượng, thời gian train, memory và tốc độ inference.
4. Chỉ sau đó mới thử tăng `mamba_d_state` hoặc `recurrent_inner_channels`.

## 10. Các lỗi thường gặp

### 10.1. Lỗi import Mamba-2

Nguyên nhân:

- chưa cài `mamba-ssm`
- môi trường CUDA/Triton không tương thích
- local repo `mamba/` chưa đủ dependency

Cách xử lý:

- kiểm tra `torch`
- kiểm tra `mamba-ssm`
- chỉ bật `recurrent_type=mamba2` khi dependency đã sẵn sàng

### 10.2. Lỗi cấu hình `headdim`

Nguyên nhân:

- `mamba_headdim` không chia hết expanded width

Cách xử lý:

- dùng preset chuẩn `64`
- nếu đổi `recurrent_inner_channels` hoặc `mamba_expand`, kiểm tra lại tính chia hết

### 10.3. Load checkpoint bị thiếu key

Nguyên nhân:

- đổi từ FSMN sang Mamba-2 hoặc ngược lại

Cách xử lý:

- đây là điều bình thường khi recurrent branch khác loại
- đảm bảo các phần còn lại load được
- ưu tiên fine-tune thay vì kỳ vọng restore hoàn toàn

