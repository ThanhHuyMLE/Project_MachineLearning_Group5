# MLP Regression bằng PyTorch

## 1. Mục đích

Mô hình dự đoán giá nhà `price` từ 88 đặc trưng sau tiền xử lý. Bản
chính thức là `MLPTorchRegressorModel`, dùng `torch.nn`, autograd và
`torch.optim.Adam`. `MLPTorchManualModel` là bản giáo dục dùng tensor
PyTorch nhưng tự cài đặt Backpropagation và Adam.

## 2. Kiến trúc

```text
Input(88) -> Linear(32) -> ReLU -> Linear(16) -> ReLU -> Linear(1)
```

Tầng cuối dùng Linear vì đây là bài toán Regression, đầu ra không bị
giới hạn trong một khoảng cố định. Trọng số được khởi tạo bằng He
initialization, phù hợp với ReLU; bias khởi tạo bằng 0.

## 3. Forward Propagation và loss

Với mỗi tầng ẩn:

```text
z = XW + b
a = ReLU(z) = max(0, z)
```

Tầng cuối không dùng activation:

```text
y_pred = aW + b
```

Hàm mất mát là Mean Squared Error:

```text
MSE = mean((y_true - y_pred)^2)
```

MSE phạt mạnh các sai số lớn và phù hợp để tối ưu mô hình hồi quy.

## 4. Huấn luyện bản chính thức

`MLPTorchRegressorModel` thực hiện đúng workflow PyTorch:

1. Chuyển dữ liệu sang tensor `float32` và tạo `DataLoader`.
2. Xây mạng bằng `torch.nn.Sequential`.
3. Dùng `nn.MSELoss` và `torch.optim.Adam`.
4. Với mỗi mini-batch: xóa gradient, forward, tính loss,
   `loss.backward()` và `optimizer.step()`.
5. Theo dõi `loss` và `val_loss` sau mỗi epoch.
6. Lưu `state_dict` vào checkpoint `.pt` và xuất toàn bộ tham số sang
   JSON để có thể nạp lại.

```python
model = MLPTorchRegressorModel(epochs=100, batch_size=32)
model.build_model(input_size=X_train.shape[1])
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
y_pred = model.predict(X_test)
```

## 5. Bản PyTorch viết tay

`MLPTorchManualModel` giữ cùng kiến trúc nhưng không dùng autograd hay
`torch.optim`. Lớp này tự tính gradient theo chain rule và tự cập nhật
hai moment của Adam. Mục đích là minh họa cơ chế học; lớp này không nằm
trong benchmark chính thức.

| Tiêu chí | PyTorch chính thức | PyTorch viết tay |
|---|---|---|
| Class | `MLPTorchRegressorModel` | `MLPTorchManualModel` |
| Gradient | Autograd | Chain rule tự cài đặt |
| Optimizer | `torch.optim.Adam` | Adam tự cài đặt |
| Vai trò | Huấn luyện và benchmark | Học thuật, kiểm chứng |
| Artifact | `.pt` và JSON | `.pt` |

## 6. Siêu tham số

| Tham số | Giá trị |
|---|---:|
| Hidden layers | `(32, 16)` |
| Learning rate | `0.001` |
| Epochs | `100` |
| Batch size | `32` |
| Random state | `42` |
| Optimizer | Adam |
| Loss | MSE |

Validation chỉ dùng để theo dõi khả năng khái quát, không dùng để cập
nhật trọng số. Tập test chỉ được dùng sau khi đã chốt cấu hình.

## 7. Đánh giá

Pipeline tính MSE, RMSE, MAE, R2 và MAPE. Theo yêu cầu bài thực hành,
notebook trình bày MSE và R2 trong bảng so sánh cuối. Kết quả benchmark
đầy đủ được lưu trong `outputs/metrics.csv` hoặc
`outputs/metrics_val.csv` tùy split đánh giá.

Kết quả test đã ghi nhận cho PyTorch MLP:

| MSE | RMSE (USD) | MAE (USD) | R2 | MAPE |
|---:|---:|---:|---:|---:|
| 23.956.240.614,18 | 154.778,04 | 89.838,73 | 0,8334 | 17,7992% |

## 8. Ưu điểm và hạn chế

Ưu điểm:

- Mô hình hóa quan hệ phi tuyến.
- Workflow PyTorch rõ ràng, dễ lưu và nạp lại.
- Có validation history để theo dõi Overfitting.

Hạn chế:

- Nhạy với chuẩn hóa dữ liệu và siêu tham số.
- Khó giải thích hơn Linear Regression và Decision Tree.
- Không vượt Random Forest trong benchmark hiện tại.
