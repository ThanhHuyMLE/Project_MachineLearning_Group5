# ADR-008: MLP có thêm bản PyTorch theo yêu cầu của cô giáo

- Trạng thái: Đã chấp nhận (19/11/2026), ngoại lệ có giới hạn cho ADR-007
- Liên quan: ADR-007 (cài đặt bằng thư viện NumPy), `docs/algorithms/mlp.md`

## Bối cảnh

Ngày 19/11/2026, cô giáo dạy bài "Training a DNN with PyTorch" và yêu
cầu lab 4 dùng PyTorch cho phần DNN/MLP. Yêu cầu này mâu thuẫn với
ADR-007 (10/11/2026, quy tắc của Thanh Huy, chủ nhóm): mọi mô hình
được cài đặt bằng thư viện NumPy, không dùng thư viện mô hình có sẵn
(không dùng class mô hình của scikit-learn, không dùng
TensorFlow/Keras/PyTorch).

Cô giáo có quyền cao hơn quy tắc của nhóm, nên yêu cầu của cô được áp
dụng. Nhưng yêu cầu của cô chỉ nói về DNN/MLP, không nói về các mô
hình còn lại, nên ngoại lệ này chỉ áp dụng cho MLP.

## Các lựa chọn đã xem xét

1. **Dùng PyTorch cho toàn bộ MLP (chọn):** dùng `torch.nn`,
   DataLoader, MSELoss, Adam và autograd. Linear Regression,
   Decision Tree và Random Forest giữ nguyên cách cài đặt hiện tại.
2. **Giữ nguyên ADR-007, không dùng PyTorch:** vi phạm yêu cầu của
   cô giáo cho lab 4.

## Quyết định

- MLP có thêm bản PyTorch `MLPTorchRegressorModel` trong
  `src/house_price/models/mlp_torch.py`, theo đúng 6 bước trong slide
  của cô giáo: (1) chuẩn bị dữ liệu bằng Dataset và DataLoader,
  (2) định nghĩa mô hình bằng `torch.nn.Module`, (3) hàm loss MSE và
  optimizer Adam, (4) training loop, (5) validation theo dõi val_loss
  mỗi epoch, (6) lưu và nạp lại mô hình bằng `state_dict`.
- Bản PyTorch giữ nguyên interface chung `__init__` ->
  `build_model` -> `fit` -> `predict` để pipeline, benchmark và
  notebook dùng thống nhất.
- Linear Regression, Decision Tree và Random Forest giữ nguyên bản
  NumPy theo ADR-007; không dùng PyTorch hay estimator có sẵn cho
  ba mô hình này.
- PyTorch chỉ dùng bản CPU (`torch>=2.9,<3`, cài bằng
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`),
  ghi trong `requirements.txt`. TensorFlow/Keras vẫn không dùng.

## Kiểm chứng

- Chạy ngày 19/11/2026 trên laptop Key (Ubuntu, Python 3.14,
  torch 2.15.0+cpu): benchmark CLI 4 mô hình trên cùng tập test
  3.241 dòng, seed 42 (`PYTHONPATH=src python -m
  house_price.train_evaluate --evaluation-split test`).
- PyTorch MLP: R2 test 0,8334, RMSE 154.778 USD, MAE 89.839 USD,
  MAPE 17,7992%; R2 validation 0,8287.
- 3 mô hình còn lại tái lập đúng số benchmark ngày 17/11/2026
  (Random Forest R2 0,8638, Linear R2 0,8119, Decision Tree R2
  0,7314), chứng tỏ pipeline không đổi.
- Test: 109 passed (thêm 8 test cho `MLPTorchRegressorModel` và 1
  trường hợp hợp đồng mới), `ruff check` và `ruff format` sạch.

> Ghi chú lịch sử (bổ sung sau khi viết ADR): các số trong mục Kiểm
> chứng này là ảnh chụp tại thời điểm viết ADR, ngày 19/11/2026, và
> được giữ nguyên vì lý do lịch sử; không sửa số 109 thành số test
> hiện tại. Sau khi thêm các test cho bản PyTorch viết tay (ADR-009),
> bộ test có 129 test, đều pass trong lần chạy ngày 19/11/2026;
> con số mới nhất xem báo cáo verify gần nhất và
> `docs/progress_results.md`. Bảng benchmark chính thức cũng được
> chốt lại bởi ADR-009: chỉ 4 mô hình (PyTorch MLP + Linear +
> Decision Tree + Random Forest).
- Notebook `notebooks/practice_04_mlp_regression.ipynb` chạy lại
  sạch, metrics trong notebook khớp benchmark CLI.

## Đánh đổi chấp nhận

- Được: tuân thủ yêu cầu của cô giáo và dùng đúng workflow PyTorch.
- Mất: thêm một dependency nặng (torch CPU khoảng 200 MB); benchmark
  chạy thêm khoảng 37 giây cho bản PyTorch; phải duy trì hai bản cài
  đặt cùng một kiến trúc.
- Rủi ro còn lại: trên dữ liệu synthetic nhỏ với ít epoch, bản PyTorch
  hội tụ chậm hơn bản NumPy ở cùng learning rate (đã kiểm chứng:
  200 epoch cho R2 khoảng 0,74-0,91 tùy seed, 500 epoch cho R2
  0,96-0,99); trên dữ liệu thật với 100 epoch (47.300 bước cập nhật)
  chênh lệch không đáng kể (R2 test lệch 0,0020).
