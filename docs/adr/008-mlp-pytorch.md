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

1. **Thêm bản PyTorch cho MLP, giữ bản NumPy làm đối chiếu (chọn):**
   MLP có hai bản cài đặt cùng kiến trúc Dense(32, ReLU) ->
   Dense(16, ReLU) -> Dense(1, Linear) và cùng cấu hình (Adam lr
   0.001, epochs 100, batch 32, seed 42): bản NumPy cũ và bản PyTorch
   mới (`torch.nn`, DataLoader, MSELoss, Adam, autograd). Benchmark
   chạy cả hai và báo cáo cạnh nhau. Linear Regression, Decision Tree
   và Random Forest không phải DNN nên giữ nguyên bản NumPy.
2. **Viết lại toàn bộ MLP bằng PyTorch, xóa bản NumPy:** mất đối
   chiếu đã kiểm chứng (gradient check 1,7e-08) và mất bằng chứng
   rằng hai cách cài đặt cho cùng kết quả.
3. **Giữ nguyên ADR-007, không dùng PyTorch:** vi phạm yêu cầu của
   cô giáo cho lab 4.

## Quyết định

- MLP có thêm bản PyTorch `MLPTorchRegressorModel` trong
  `src/house_price/models/mlp_torch.py`, theo đúng 6 bước trong slide
  của cô giáo: (1) chuẩn bị dữ liệu bằng Dataset và DataLoader,
  (2) định nghĩa mô hình bằng `torch.nn.Module`, (3) hàm loss MSE và
  optimizer Adam, (4) training loop, (5) validation theo dõi val_loss
  mỗi epoch, (6) lưu và nạp lại mô hình bằng `state_dict`.
- Bản PyTorch giữ nguyên interface chung `__init__` ->
  `build_model` -> `fit` -> `predict` và cùng siêu tham số với bản
  NumPy, để pipeline, benchmark và notebook không đổi cách gọi.
- Bản NumPy `MLPRegressorModel` được giữ nguyên, không xóa, làm đối
  chiếu. Benchmark chính thức chạy cả hai bản trên cùng tập test và
  báo cáo cạnh nhau.
- Linear Regression, Decision Tree và Random Forest giữ nguyên bản
  NumPy theo ADR-007; không dùng PyTorch hay estimator có sẵn cho
  ba mô hình này.
- PyTorch chỉ dùng bản CPU (`torch>=2.9,<3`, cài bằng
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`),
  ghi trong `requirements.txt`. TensorFlow/Keras vẫn không dùng.

## Kiểm chứng

- Chạy ngày 19/11/2026 trên laptop Key (Ubuntu, Python 3.14,
  torch 2.15.0+cpu): benchmark CLI 5 mô hình trên cùng tập test
  3.241 dòng, seed 42 (`PYTHONPATH=src python -m
  house_price.train_evaluate --evaluation-split test`).
- PyTorch MLP: R2 test 0,8334, RMSE 154.778 USD, MAE 89.839 USD,
  MAPE 17,7992%; bản NumPy: R2 test 0,8354, RMSE 153.881 USD,
  MAE 90.392 USD, MAPE 18,0595%. Chênh lệch R2 chỉ 0,0020; MAE và
  MAPE của bản PyTorch thấp hơn bản NumPy. Trên validation:
  PyTorch R2 0,8287 so với NumPy R2 0,8310.
- 4 mô hình còn lại tái lập đúng số benchmark ngày 17/11/2026
  (Random Forest R2 0,8638, Linear R2 0,8119, Decision Tree R2
  0,7314), chứng tỏ pipeline không đổi.
- Test: 109 passed (thêm 8 test cho `MLPTorchRegressorModel` và 1
  trường hợp hợp đồng mới), `ruff check` và `ruff format` sạch.
- Notebook `notebooks/practice_04_mlp_regression.ipynb` chạy lại
  sạch, metrics trong notebook khớp benchmark CLI.

## Đánh đổi chấp nhận

- Được: tuân thủ yêu cầu của cô giáo; có hai bản MLP đối chiếu nhau,
  chứng minh kết quả không phụ thuộc backend.
- Mất: thêm một dependency nặng (torch CPU khoảng 200 MB); benchmark
  chạy thêm khoảng 37 giây cho bản PyTorch; phải duy trì hai bản cài
  đặt cùng một kiến trúc.
- Rủi ro còn lại: trên dữ liệu synthetic nhỏ với ít epoch, bản PyTorch
  hội tụ chậm hơn bản NumPy ở cùng learning rate (đã kiểm chứng:
  200 epoch cho R2 khoảng 0,74-0,91 tùy seed, 500 epoch cho R2
  0,96-0,99); trên dữ liệu thật với 100 epoch (47.300 bước cập nhật)
  chênh lệch không đáng kể (R2 test lệch 0,0020).
