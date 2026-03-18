# Kế hoạch triển khai MossFormer2-Mamba2

## 1. Mục tiêu và phạm vi

Mục tiêu của thay đổi này là thay nhánh recurrent của MossFormer2 từ Dilated FSMN sang Mamba-2, nhưng vẫn giữ nguyên phần attention branch, encoder/decoder, mask head, residual structure, normalization, và output interface của model. Biến thể mới có thể gọi là `MossFormer2-Mamba` hoặc `MossFormer2-SSD`.

Phạm vi triển khai được chốt như sau:

- Giữ nguyên `FLASH_ShareA_FFConvM`.
- Giữ nguyên `Encoder`, `Decoder`, `MossFormer_MaskNet`, output gate, separator head.
- Chỉ thay nhánh recurrent trong `MossformerBlock_GFSMN`.
- Không biến model thành một kiến trúc mới; chỉ thay khối recurrent theo hướng tương thích với kiến trúc gốc.

## 2. Phân tích code hiện tại

Luồng gọi hiện tại của model trong repo:

1. `MossFormer2_SS` trong `mossformer2/models/mossformer2/mossformer2.py`
2. `MossFormer`
3. `MossFormer_MaskNet`
4. `Computation_Block`
5. `MossFormerM`
6. `MossformerBlock_GFSMN`

Trong `MossformerBlock_GFSMN`, mỗi tầng hiện đang thực hiện hai bước theo đúng thứ tự:

1. Chạy `flash(x)` bằng `FLASH_ShareA_FFConvM`
2. Chạy `self.fsmn[i](x)` bằng `Gated_FSMN_Block_Dilated`

FSMN hiện tại gồm hai lớp chính:

- `Gated_FSMN_dilated`: dùng `to_u`, `to_v`, và `UniDeepFsmn_dilated`
- `Gated_FSMN_Block_Dilated`: bọc thêm `conv1`, `CLayerNorm`, `conv2`, và residual ngoài block

Ý nghĩa kiến trúc hiện tại:

- Nhánh attention (`FLASH_ShareA_FFConvM`) học phụ thuộc local-global.
- Nhánh FSMN đóng vai trò memory/sequential modeling.

Trong repo `mamba`, class `Mamba2` nhận input `[B, L, D]` và trả output `[B, L, D]`. Điều này khớp với shape ở giữa recurrent branch hiện tại, nên Mamba-2 phù hợp để thay đúng phần memory branch mà không cần sửa shape contract của separator.

## 3. Kiến trúc thay thế đề xuất

Lựa chọn kiến trúc được chốt:

- Không thay riêng `UniDeepFsmn_dilated`
- Không thay attention branch
- Thay ở mức block: `Gated_FSMN_Block_Dilated` -> `Gated_Mamba2_Block`

### 3.1. Block mới cần có

Block mới vẫn giữ topology của block cũ:

1. `conv1: Conv1d(dim -> inner_channels) + PReLU`
2. `norm1: CLayerNorm`
3. `Gated_Mamba2`
4. `norm2: CLayerNorm`
5. `conv2: Conv1d(inner_channels -> dim)`
6. residual ngoài block

### 3.2. Thiết kế `Gated_Mamba2`

`Gated_Mamba2` giữ vai trò tương đương `Gated_FSMN_dilated`, với cấu trúc:

- `to_u = FFConvM(...)`
- `to_v = FFConvM(...)`
- `mamba = Mamba2(...)`

Forward logic:

1. `x_u = to_u(x)`
2. `x_v = to_v(x)`
3. `x_u = mamba(x_u)`
4. `out = x_v * x_u + x`

Việc giữ gate branch là bắt buộc trong v1 để bảo toàn inductive bias của MossFormer2 và giúp ablation rõ ràng.

### 3.3. Shape contract bắt buộc

- Block input: `[B, L, D]`
- Sau `conv1 + norm1`, trước recurrent core: `[B, L, C]`
- Mamba-2 input/output: `[B, L, C]`
- Block output: `[B, L, D]`

Không được thay đổi thứ tự transpose hiện có ngoài phần recurrent core.

## 4. Các thay đổi code phải thực hiện

### 4.1. Thay đổi trong `mossformer2/models/mossformer2/mossformer2_block.py`

Cần sửa trực tiếp file này ở v1.

Các thay đổi cụ thể:

1. Thêm import `Mamba2`, hoặc một wrapper import an toàn để báo lỗi rõ nếu dependency chưa sẵn sàng.
2. Thêm class `Gated_Mamba2`.
3. Thêm class `Gated_Mamba2_Block`.
4. Đổi `self.fsmn` trong `MossformerBlock_GFSMN` thành `self.recurrent`.
5. Sửa constructor `MossformerBlock_GFSMN` để nhận thêm config:
   - `recurrent_type`
   - `recurrent_inner_channels`
   - `mamba_d_state`
   - `mamba_d_conv`
   - `mamba_expand`
   - `mamba_headdim`
6. Sửa logic khởi tạo:
   - nếu `recurrent_type == "fsmn"` dùng `Gated_FSMN_Block_Dilated`
   - nếu `recurrent_type == "mamba2"` dùng `Gated_Mamba2_Block`
   - ngược lại `raise ValueError(...)`
7. Sửa `forward` của `MossformerBlock_GFSMN`:
   - từ `self.fsmn[ii](x)`
   - thành `self.recurrent[i](x)`

### 4.2. Thay đổi trong `mossformer2/models/mossformer2/mossformer2.py`

Cần propagate config recurrent xuyên suốt constructor chain:

1. `MossFormerM.__init__`
   - nhận thêm config recurrent
   - truyền xuống `MossformerBlock_GFSMN`
2. `Computation_Block.__init__`
   - nhận config recurrent
   - forward tiếp cho `MossFormerM`
3. `MossFormer_MaskNet.__init__`
   - nhận config recurrent
   - forward tiếp cho `Computation_Block`
4. `MossFormer.__init__`
   - nhận config recurrent
   - forward tiếp cho `MossFormer_MaskNet`
5. `MossFormer2_SS.__init__`
   - lấy config từ `args`
   - truyền toàn bộ xuống `MossFormer`

Các phần không được sửa:

- `Encoder`
- `Decoder`
- `MossFormer_MaskNet.forward` logic tách nguồn
- `FLASH_ShareA_FFConvM`

### 4.3. Thay đổi trong `mossformer2/train.py` và `mossformer2/inference.py`

Thêm parser args mới:

- `--recurrent-type`
- `--recurrent-inner-channels`
- `--mamba-d-state`
- `--mamba-d-conv`
- `--mamba-expand`
- `--mamba-headdim`

Default được chốt:

- `recurrent_type: fsmn`
- `recurrent_inner_channels: 256`
- `mamba_d_state: 64`
- `mamba_d_conv: 4`
- `mamba_expand: 2`
- `mamba_headdim: 64`

### 4.4. Thay đổi trong YAML train/inference

Thêm các khóa mới vào toàn bộ config train/inference đang dùng:

- `recurrent_type`
- `recurrent_inner_channels`
- `mamba_d_state`
- `mamba_d_conv`
- `mamba_expand`
- `mamba_headdim`

Mục tiêu là không hard-code các giá trị Mamba trong model constructor.

## 5. Xử lý dependency và rủi ro kỹ thuật

### 5.1. Dependency

Dependency được ưu tiên:

- `mamba-ssm`
- môi trường CUDA/Triton tương thích

Yêu cầu triển khai:

- Nếu import Mamba lỗi, phải fail rõ ràng với thông báo dễ hiểu.
- Không fallback âm thầm sang implementation giả.

Quyết định cho v1:

- Dùng `Mamba2` chuẩn trước.
- Chỉ cân nhắc `Mamba2Simple` nếu môi trường không hỗ trợ fused kernels, và khi đó phải ghi minh bạch quyết định này trong code hoặc tài liệu.

### 5.2. Ràng buộc kỹ thuật của Mamba-2

`headdim` phải chia hết `d_ssm`.

Với cấu hình mặc định:

- `inner_channels = 256`
- `expand = 2`
- `d_inner = 512`
- `headdim = 64`

thì cấu hình hợp lệ.

### 5.3. Rủi ro checkpoint

Checkpoint baseline FSMN sẽ không load hoàn toàn vào recurrent branch mới.

Yêu cầu xử lý:

- Cho phép partial load
- Log rõ `missing keys` / `unexpected keys`
- Không làm hỏng khả năng finetune từ phần còn lại của model

### 5.4. Rủi ro shape và stability

Các lỗi dễ xảy ra nhất:

- nhầm `[B, D, L]` với `[B, L, D]`
- truyền tensor sai shape vào `Mamba2`
- quên transpose sau `CLayerNorm`
- cấu hình `headdim` không tương thích với `d_ssm`
- instability khi train do precision của SSM

Khuyến nghị:

- Giữ parameter ở fp32 khi train AMP
- Không tự ý đổi initialization của Mamba

## 6. Test plan bắt buộc

### 6.1. Unit test cho block mới

Viết test cho `Gated_Mamba2_Block`:

- Input giả: `[2, 100, 512]`
- Kỳ vọng:
  - output shape đúng bằng input shape
  - output toàn giá trị hữu hạn

### 6.2. Unit test cho backbone

Viết test cho `MossformerBlock_GFSMN(recurrent_type="mamba2")`:

- Input giả: `[2, 100, 512]`
- Kỳ vọng:
  - output shape không đổi
  - forward pass chạy được

### 6.3. End-to-end test cho separator

Viết test cho `MossFormer2_SS`:

- Input waveform: `[B, T]`
- Kỳ vọng:
  - output là list
  - độ dài list bằng `num_spks`
  - mỗi phần tử có shape `[B, T]`

### 6.4. Test parser/config

Kiểm tra cả hai mode:

- `recurrent_type=fsmn`
- `recurrent_type=mamba2`

Kỳ vọng:

- model khởi tạo được từ parser/YAML
- không cần sửa code để chuyển giữa baseline và biến thể mới

### 6.5. Test param count và benchmark

Thêm utility hoặc script nhỏ để:

- đếm số tham số baseline FSMN
- đếm số tham số biến thể Mamba-2
- benchmark forward cơ bản trên chuỗi ngắn và dài

### 6.6. Acceptance criteria

Các tiêu chí hoàn thành tối thiểu:

- forward pass chạy không lỗi
- output shape không đổi
- tensor output hữu hạn
- baseline FSMN vẫn chạy được
- config Mamba-2 bật/tắt được mà không sửa code

## 7. Thứ tự triển khai đề xuất

1. Thêm config CLI + YAML
2. Propagate config qua chain constructor trong `mossformer2.py`
3. Thêm `Gated_Mamba2` và `Gated_Mamba2_Block`
4. Refactor `MossformerBlock_GFSMN` sang `self.recurrent`
5. Thêm kiểm tra import/dependency rõ ràng
6. Viết sanity tests shape + end-to-end
7. Chạy so sánh params và benchmark forward
8. Chuẩn bị ablation baseline `fsmn` vs `mamba2`

## 8. Public interface cần ghi nhớ

Các trường config/CLI/YAML mới:

- `recurrent_type`
- `recurrent_inner_channels`
- `mamba_d_state`
- `mamba_d_conv`
- `mamba_expand`
- `mamba_headdim`

Public interface của model không đổi:

- input vẫn là waveform `[B, T]`
- output vẫn là list độ dài `num_spks`, mỗi phần tử shape `[B, T]`

## 9. Assumptions chốt sẵn

- V1 không tạo file mới riêng cho recurrent block; sửa trực tiếp trong `mossformer2_block.py`
- V1 không thay đổi train loop, loss, dataloader, encoder, decoder, hoặc attention branch
- V1 không cố hỗ trợ fallback CPU cho Mamba-2 nếu dependency không đáp ứng
- V1 giữ baseline FSMN làm mặc định để tránh phá workflow hiện tại

