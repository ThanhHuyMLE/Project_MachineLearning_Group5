# Multilayer Perceptron Regression (MLP)

## 1. Giới thiệu và mục đích

Multilayer Perceptron (MLP) là mạng nơ-ron truyền thẳng (feedforward)
nhiều tầng, dùng để học quan hệ phi tuyến giữa đặc trưng căn nhà và giá bán.
Bài tập dùng MLP làm mô hình chính, đối chiếu với 3 mô hình truyền thống.

## 2. Đầu vào và đầu ra

- Đầu vào: vector `X` gồm 18 đặc trưng đã chuẩn hóa (mean 0, std 1).
- Đầu ra: một số thực `y_hat` là giá nhà dự đoán (USD).
- Kiến trúc theo đề bài: Dense(32, relu) -> Dense(16, relu) -> Dense(1, linear).

## 3. Mô hình hoạt động thế nào

Mỗi tầng tính tổ hợp tuyến tính rồi qua hàm kích hoạt phi tuyến.
Tầng ẩn dùng ReLU để học quan hệ phi tuyến; tầng ra dùng Linear để trả
giá trị liên tục không giới hạn.

## 4. Nền tảng toán học

Lan truyền tiến (Forward Propagation), `l` là chỉ số tầng:

```text
z[l] = W[l] a[l-1] + b[l]
a[l] = ReLU(z[l]) = max(0, z[l])      (tầng ẩn)
y_hat = z[3] = W[3] a[2] + b[3]       (tầng ra, linear)
```

Hàm mất mát MSE trên mini-batch `B` mẫu:

```text
L = (1/B) * sum((y_i - y_hat_i)^2)
```

Lan truyền ngược (Backpropagation) tính gradient bằng quy tắc chuỗi
(chain rule), rồi Adam cập nhật trọng số:

```text
m_t = beta1 m_{t-1} + (1 - beta1) g_t
v_t = beta2 v_{t-1} + (1 - beta2) g_t^2
m_hat = m_t / (1 - beta1^t);  v_hat = v_t / (1 - beta2^t)
theta_t = theta_{t-1} - eta * m_hat / (sqrt(v_hat) + epsilon)
```

Với `eta = 0.001`, `beta1 = 0.9`, `beta2 = 0.999`, `epsilon = 1e-7`.
Chi tiết đầy đủ xem [DESIGN.md](../DESIGN.md) mục 5.2.

## 5. Mã giả

Xem [pseudocode.md](../pseudocode.md) mục MLP Regression. Mã giả trình bày vòng lặp
từng phần tử; code dùng phép tính mảng của thư viện.

## 6. Cài đặt Python

File: `src/house_price/models/mlp.py`, class `MLPRegressorModel`.
Backend mặc định là sklearn `MLPRegressor` (xem `adr/004-mlp-trien-khai.md`).

| Phương thức | Map với mã giả |
|---|---|
| `__init__(hidden_layer_sizes=(32, 16), ...)` | Lưu cấu hình, chưa tạo mạng |
| `build_model()` | Tạo estimator/kiến trúc, reset trạng thái |
| `fit(X_train, y_train)` | Huấn luyện, lưu `history_` (loss/MAE) |
| `predict(X_test)` | Trả giá dự đoán, báo lỗi nếu chưa fit |

Tham số mặc định: hidden (32, 16), activation relu, solver adam,
loss MSE, `max_iter=100`, `batch_size=32`, `random_state=42`.

## 7. Dữ liệu sử dụng

King County, 18 đặc trưng, mục tiêu `price`.
Xem [king-county-dataset.md](../king-county-dataset.md).
Chuẩn hóa StandardScaler chỉ fit trên train.

## 8. Chỉ số đánh giá

MSE, RMSE, MAE, R2 (và MAPE nếu hợp lý). Xem `evaluation.py`.
MSE/RMSE/MAE càng nhỏ càng tốt; R2 càng lớn càng tốt, có thể âm.

## 9. Trực quan hóa và demo

- Đường loss train/validation theo epoch (phát hiện overfitting).
- Biểu đồ Actual vs Predicted: điểm càng sát đường `y = y_hat` càng tốt.
- Hàm vẽ xem `visualization.py`.

## 10. Kết quả thực nghiệm

**CHƯA THỰC HIỆN.** Chưa huấn luyện trong Phase 1.
Sinh viên 2 và 3 điền số thật vào đây và `progress_results.md`
sau khi chạy, kèm theo số epoch thực tế chạy được.

## 11. Hạn chế và thảo luận

- Nhạy cảm với chuẩn hóa: đặc trưng chưa chuẩn hóa làm Adam hội tụ kém.
- Nhạy cảm với siêu tham số (learning rate, epochs, kích thước tầng ẩn).
- Khó giải thích từng dự đoán (mô hình hộp đen) so với Linear Regression.
- Cần dữ liệu đủ lớn; 21.613 mẫu là đủ cho kiến trúc nhỏ này.
- Hiểu lầm thường gặp: tầng ẩn nhiều hơn không phải lúc nào cũng tốt hơn;
  tầng ẩn quá lớn trên dữ liệu bảng dễ gây overfitting.
