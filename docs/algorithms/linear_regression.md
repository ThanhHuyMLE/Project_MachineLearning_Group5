# Linear Regression (Hồi quy tuyến tính)

## 1. Giới thiệu và mục đích

Linear Regression là mô hình cơ sở (baseline) đơn giản nhất: giá nhà được
mô hình hóa bằng tổ hợp tuyến tính của các đặc trưng. Dùng làm mốc để
đánh giá xem mô hình phi tuyến (MLP, cây) có thực sự tốt hơn không.

## 2. Đầu vào và đầu ra

- Đầu vào: vector `X` gồm 18 đặc trưng đã chuẩn hóa.
- Đầu ra: một số thực `y_hat` là giá nhà dự đoán (USD).

## 3. Mô hình hoạt động thế nào

Mô hình tìm một siêu phẳng sao cho dự đoán càng gần giá thật càng tốt.
Mỗi hệ số cho biết giá thay đổi bao nhiêu khi đặc trưng đó tăng 1 đơn vị
(đã chuẩn hóa), giữ các đặc trưng khác không đổi.

## 4. Nền tảng toán học

Mô hình: `y_hat = X beta + b`, với `beta` là vector hệ số, `b` là hệ số chặn.
Mục tiêu bình phương tối thiểu (OLS):

```text
beta* = argmin sum((y_i - X_i beta - b)^2)
```

Nghiệm dạng kín (khi `X^T X` khả nghịch):

```text
beta = (X^T X)^(-1) X^T y
```

Thực tế scikit-learn giải bằng phép biến đổi tuyến tính ổn định số
(SVD/QR) thay vì nghịch đảo trực tiếp.

## 5. Mã giả

Xem [pseudocode.md](../pseudocode.md) mục Linear Regression.

## 6. Cài đặt Python

File: `src/house_price/models/linear_regression.py`,
class `LinearRegressorModel` (đã cài đặt đầy đủ làm mẫu cho 3 class còn lại).

| Phương thức | Công việc |
|---|---|
| `__init__()` | Lưu cấu hình, chưa có hệ số |
| `build_model()` | Tạo `sklearn LinearRegression`, reset trạng thái |
| `fit(X_train, y_train)` | Giải OLS, lưu `coef_` và `intercept_` |
| `predict(X_test)` | Trả `X beta + b`, báo lỗi nếu chưa fit |

## 7. Dữ liệu sử dụng

King County, 18 đặc trưng, mục tiêu `price`.
Xem [king-county-dataset.md](../king-county-dataset.md).

## 8. Chỉ số đánh giá

MSE, RMSE, MAE, R2 trên cùng tập test với 3 mô hình kia.

## 9. Trực quan hóa và demo

Biểu đồ Actual vs Predicted; hệ số `coef_` có thể vẽ để xem đặc trưng
nào ảnh hưởng mạnh nhất (chỉ có ý nghĩa khi đặc trưng đã chuẩn hóa).

## 10. Kết quả thực nghiệm

**CHƯA THỰC HIỆN.** Sinh viên 4 điền số thật sau khi chạy.

## 11. Hạn chế và thảo luận

- Chỉ học được quan hệ tuyến tính; kém khi giá phụ thuộc phi tuyến
  vào diện tích, vị trí, chất lượng.
- Nhạy cảm với ngoại lai (outlier) vì loss bình phương phạt nặng sai số lớn.
- `zipcode` đang xử lý như số liên tục: hệ số của nó không có ý nghĩa
  thứ tự thật; cần ghi rõ giới hạn này.
- Ưu điểm: huấn luyện nhanh, dễ giải thích, ít overfitting, làm mốc tốt.
