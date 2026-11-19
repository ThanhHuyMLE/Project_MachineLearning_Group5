# ADR-009: Bản MLP PyTorch viết tay (giáo dục) và benchmark chính thức 4 mô hình

- Trạng thái: Đã chấp nhận (19/11/2026)
- Liên quan: ADR-007 (cài đặt bằng thư viện NumPy), ADR-008 (MLP bản
  PyTorch chính thức), `docs/algorithms/mlp.md`,
  `docs/progress_results.md`

## Bối cảnh

ADR-008 đã thêm bản PyTorch chính thức cho MLP, nhưng bản đó dùng
autograd (`loss.backward()`) và `torch.optim.Adam` nên sinh viên
không nhìn thấy đạo hàm và bước cập nhật tham số chạy như thế nào.
Trong buổi vấn đáp, giảng viên có thể hỏi "cái nào mới là backprop
viết tay, PyTorch cung cấp cái gì" - câu hỏi này không trả lời được
chỉ bằng bản chính thức. Đồng thời, benchmark lúc đó vẫn liệt kê bản
NumPy MLP trong bảng chính thức, dù quyết định của ADR-008 là benchmark
chạy cả hai bản cạnh nhau; cần chốt rõ bảng chính thức chỉ có 4 mô hình
(PyTorch MLP + Linear + Decision Tree + Random Forest) và bản NumPy
MLP đứng ngoài bảng, chỉ chạy kèm để đối chiếu.

Vai trò của 4 nhóm mô hình, chốt trong ADR này:

| Nhóm | Công nghệ | Cách tính gradient | Vai trò |
|---|---|---|---|
| Standard PyTorch MLP | PyTorch | Autograd (`loss.backward()`) | Workflow và benchmark chính thức của Lab 4 |
| Manual PyTorch MLP | PyTorch Tensor | Chain rule viết tay | Bằng chứng hiểu thuật toán và để kiểm chứng |
| NumPy MLP | NumPy | Chain rule viết tay | Đối chiếu legacy mang tính giáo dục |
| Các mô hình hồi quy còn lại (Linear, Decision Tree, Random Forest) | NumPy | Thuật toán viết tay | Baseline đối chiếu chính thức |

Bản manual là bằng chứng trung tâm cho thấy nhóm hiểu backprop và
Adam, không phải file phụ; bản PyTorch chính thức không phải "hoàn
toàn viết tay" vì nó dùng autograd và `torch.optim.Adam` của PyTorch.

## Các lựa chọn đã xem xét

1. **Thêm bản PyTorch viết tay làm bản giáo dục, không đưa vào
   benchmark (chọn):** `MLPTorchManualModel` dùng tensor torch cho
   mọi phép tính nhưng tự viết forward, backward theo chain rule và
   Adam, tham số `requires_grad=False`, training path không gọi
   `backward()`, không dùng autograd hay `torch.optim`. Bản này chỉ
   phục vụ học và vấn đáp; benchmark chính thức không đổi.
2. **Mở rộng bản PyTorch chính thức bằng cờ "manual":** làm `fit`
   của bản chính thức có hai nhánh, code khó đọc và dễ lẫn lộn khi
   vấn đáp (không biết nhánh nào đang chạy).
3. **Dùng bản NumPy làm bản "viết tay" duy nhất:** bản NumPy đã có
   sẵn backprop viết tay, nhưng nó dùng mảng NumPy, không trả lời
   được câu hỏi về tensor, dtype, device và autograd của PyTorch
   trong mục 10-11 của spec lab 4.
4. **Đưa bản viết tay vào benchmark chính thức thay bản PyTorch:**
   sai vai trò: benchmark đo mô hình dùng thật (autograd + Adam của
   torch), còn bản viết tay chỉ để chứng minh hiểu cơ chế.

## Quyết định

- Thêm `MLPTorchManualModel` trong
  `src/house_price/models/mlp_torch_manual.py`, cùng kiến trúc
  32-16-1 và cùng cấu hình với hai bản MLP còn lại (Adam lr 0.001,
  beta1 0.9, beta2 0.999, epsilon 1e-8, epochs 100, batch 32, seed
  42), cùng interface `__init__` -> `build_model` -> `fit` ->
  `predict` và cùng `save`/`load` thành `.pt`.
- Bản manual viết tay: `_forward` (lưu cache activations và
  pre-activations), `_backward` (delta đầu ra `(2/n)(y_pred - y)`
  rồi chain rule ngược qua ReLU, đạo hàm ReLU bằng 0 tại 0),
  `_adam_update` (moment m, v và hiệu chỉnh độ lệch, chạy dưới
  `torch.no_grad()`), hàm loss `mse_loss` (trung bình bình phương
  sai số). DataLoader/TensorDataset vẫn dùng để nạp mini-batch
  (batch 32, shuffle bằng generator có seed), vì nạp dữ liệu không
  phải phần "học" cần viết tay.
- Bản manual KHÔNG thuộc benchmark chính thức và KHÔNG có trong
  `src/house_price/train_evaluate.py` (`build_models` chỉ tạo 4 mô
  hình chính thức). Nó được export trong
  `src/house_price/models/__init__.py` để test và demo trực tiếp.
- Benchmark chính thức chốt 4 mô hình: PyTorch MLP Regression,
  Linear Regression, Decision Tree Regression, Random Forest
  Regression. Bản NumPy MLP (`MLP Regression`) đứng ngoài bảng,
  chỉ chạy kèm bằng cờ `--include-numpy-mlp` của
  `train_evaluate.py` để đối chiếu với các lần chạy cũ.
- Checkpoint của bản manual lưu trực tiếp danh sách tensor
  `weights`/`biases` (không dùng `state_dict`, vì bản này không có
  `nn.Module`); checkpoint của bản chính thức lưu `state_dict`.
  Cả hai đều nạp bằng `torch.load(..., map_location="cpu",
  weights_only=True)`.

## Kiểm chứng

- Gradient check (mạng 3 -> 4 -> 1, so với finite difference):
  sai số tương đối tối đa 1,739e-03, xem
  `tests/test_mlp_torch_manual.py::test_manual_gradient_matches_finite_difference`.
  Sai số lớn hơn nhiều so với bản NumPy (1,7e-08) vì tensor float32
  của torch (bản NumPy dùng float64); đây là sai số làm tròn của
  float32, không phải lỗi chain rule - test vẫn pass với ngưỡng
  dành cho float32.
- Học ánh xạ tuyến tính trên dữ liệu synthetic: R2 0,9834, xem
  `tests/test_mlp_torch_manual.py::test_manual_learns_linear_mapping`.
- Training path không dùng autograd: test
  `test_manual_training_path_uses_no_autograd` quét source của
  `mlp_torch_manual.py` và cấm `backward()`, `autograd.grad` và
  `torch.optim` trong training path; tham số có
  `requires_grad=False` (test `test_manual_parameters_shape_dtype_device_and_no_grad`).
- Lưu/nạp: `test_manual_save_and_load_round_trip` cho cùng dự đoán;
  `test_manual_reproducible_with_same_seed` cho cùng kết quả với
  cùng seed.
- Benchmark chính thức ngày 19/11/2026 (test 3.241 dòng, seed 42):
  RF R2 0,8638, PyTorch MLP R2 0,8334, Linear R2 0,8119, DT R2
  0,7314; validation: RF 0,8617, PyTorch MLP 0,8287, Linear 0,8091,
  DT 0,7198. Bản NumPy MLP ngoài bảng: R2 validation 0,8310. Chi
  tiết xem `docs/progress_results.md`.

## Đánh đổi chấp nhận

- Được: có bằng chứng viết tay cho backprop và Adam bằng tensor
  torch; trả lời được câu hỏi vấn đáp "autograd làm gì thay mình";
  benchmark chính thức gọn và đúng vai trò từng bản.
- Mất: thêm một bản cài đặt thứ ba cùng kiến trúc phải duy trì;
  bản manual không hứa parity bit-for-bit với hai bản còn lại
  (thứ tự phép tính float32, cách xáo trộn batch và khởi tạo khác
  nhau nên số có thể lệch nhỏ).
- Rủi ro còn lại: người đọc có thể nhầm bản manual là mô hình
  chính thức; tài liệu (README, `docs/algorithms/mlp.md`,
  `docs/oral-examination.md`) phải luôn ghi rõ "bản giáo dục,
  không benchmark".
