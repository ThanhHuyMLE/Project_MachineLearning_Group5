# Linear Regression (Hồi quy tuyến tính)

## 1. Mục đích

Linear Regression là mô hình cơ sở (baseline) của dự án: giá nhà được
mô hình hóa bằng tổ hợp tuyến tính của các đặc trưng. Vai trò của nó là
làm mốc để đánh giá xem các mô hình phi tuyến (MLP, cây quyết định,
rừng ngẫu nhiên) có thực sự học được thêm thông tin hay không. Nếu một
mô hình phức tạp không vượt được baseline này trên cùng tập test thì
độ phức tạp đó chưa đem lại giá trị.

## 2. Đầu vào và đầu ra

- Đầu vào: ma trận `X` có dạng `(N, 88)`: 18 đặc trưng số đã chuẩn hóa
  bằng `StandardScaler` (fit trên train) cộng 70 cột one-hot của
  `zipcode`. Thứ tự cột cố định, lấy từ `output_feature_names`.
- Đầu ra: vector `y_hat` có dạng `(N,)`, giá nhà dự đoán tính bằng USD.
- Mục tiêu huấn luyện: `y` là cột `price`, giữ đơn vị USD; pipeline
  hiện tại không chuẩn hóa mục tiêu.
- Giả định về thang đo: `X` đã chuẩn hóa nên các cột có thang đo cỡ 1.
  Thuật toán gradient descent trong code giả định điều này; đưa `X`
  chưa chuẩn hóa với thang đo lớn có thể làm thuật toán phân kỳ, và
  `fit` sẽ báo lỗi rõ ràng thay vì trả về hệ số vô hạn hoặc NaN.

## 3. Mô hình hoạt động thế nào

Mô hình tìm một siêu phẳng sao cho dự đoán càng gần giá thật càng tốt
theo tiêu chí bình phương tối thiểu (OLS). Mỗi hệ số cho biết giá thay
đổi bao nhiêu khi đặc trưng đó tăng 1 đơn vị (đã chuẩn hóa), giữ các
đặc trưng khác không đổi. Vì `zipcode` đã mã hóa one-hot, mỗi hệ số
của một cột one-hot cho biết giá trung bình của căn nhà ở mã vùng đó
chênh lệch bao nhiêu so với mức chung, khi các đặc trưng khác giữ
nguyên; `zipcode` không còn xử lý như số liên tục.

## 4. Nền tảng toán học

Mô hình:

```text
y_hat = X beta + b
```

Gộp hệ số chặn vào vector tham số bằng ma trận thiết kế mở rộng
`X_aug = [1 | X]` và `beta_aug = [b, beta]`, mô hình viết gọn là
`y_hat = X_aug beta_aug`. Hàm mất mát MSE:

```text
L(beta_aug) = (1/n) * ||X_aug beta_aug - y||^2
```

Gradient của mất mát theo `beta_aug`:

```text
dL/d beta_aug = (2/n) * X_aug^T (X_aug beta_aug - y)
```

Nghiệm dạng đóng của bài toán OLS (khi `X_aug^T X_aug` khả nghịch):

```text
beta_aug = (X_aug^T X_aug)^(-1) X_aug^T y
```

Nghiệm dạng đóng chỉ dùng trong test để đối chiếu tính đúng đắn của
thuật toán huấn luyện; code huấn luyện dùng batch gradient descent
(xem mục 5), không dùng nghiệm dạng đóng để fit.

Quy tắc dừng của gradient descent trong code dùng thay đổi TƯƠNG ĐỐI
của loss giữa hai epoch liên tiếp:

```text
|L_t - L_(t-1)| / max(L_(t-1), 1e-12) < tol
```

Dùng thay đổi tương đối (không phải tuyệt đối) vì `y` là giá nhà thô
(đơn vị USD, MSE cỡ 1e10) nên một ngưỡng tuyệt đối không có ý nghĩa
chung cho mọi thang đo của `y`.

## 5. Mã giả

Mã giả dưới đây khớp với `fit` trong
`src/house_price/models/linear_regression.py`. Quy ước: chỉ số bắt
đầu từ 0; code Python gộp một số vòng lặp thành phép tính mảng.

```text
ĐẦU VÀO: X_train (N x 88), y_train, learning_rate = 0.1,
         max_epochs = 1000, tol = 1e-6
ĐẦU RA: y_pred; coef_ (beta), intercept_ (b), history_ (loss từng epoch),
        n_iter_, converged_

THỦ TỤC INIT(learning_rate, max_epochs, tol):
    lưu cấu hình; coef_ <- RỖNG; intercept_ <- RỖNG
    n_iter_ <- 0; converged_ <- False

THỦ TỤC BUILD_MODEL():
    reset trạng thái đã học; fitted <- False

THỦ TỤC FIT(X_train, y_train):
    kiểm tra model đã build, shape hợp lệ, dữ liệu hữu hạn
    X_aug <- [1 | X_train]                      # thêm cột 1 cho hệ số chặn
    beta_aug <- 0                               # khởi tạo bằng 0
    losses <- []
    LẶP epoch = 1 .. max_epochs:
        residual <- X_aug beta_aug - y_train
        gradient <- (2/n) * X_aug^T residual
        beta_aug <- beta_aug - learning_rate * gradient
        loss <- mean((X_aug beta_aug - y_train)^2)
        NẾU loss không hữu hạn: BÁO LỖI phân kỳ, DỪNG
        NẾU đã có loss trước đó VÀ
           |loss_truoc - loss| / max(loss_truoc, 1e-12) < tol:
            losses.append(loss); converged_ <- True; DỪNG LẶP
        losses.append(loss)
    intercept_ <- beta_aug[0]; coef_ <- beta_aug[1:]
    history_ <- {"loss": losses}; n_iter_ <- len(losses); fitted <- True

HÀM PREDICT(X):
    NẾU model chưa fit: BÁO LỖI, DỪNG
    TRẢ VỀ X coef_ + intercept_

HÀM EXPORT_PARAMS(path, feature_names):
    ghi JSON: intercept, coefficients theo tên đặc trưng,
    learning_rate, n_epochs, final_loss, converged
```

## 6. Cài đặt Python

File: `src/house_price/models/linear_regression.py`,
class `LinearRegressorModel`. Đây là cài đặt bằng thư viện NumPy;
không dùng `sklearn.linear_model.LinearRegression` hay bất kỳ
estimator có sẵn nào. NumPy chỉ làm phép tính mảng. Xem
[adr/007-mo-hinh-code-tay.md](../adr/007-mo-hinh-code-tay.md).

| Phương thức | Công việc |
|---|---|
| `__init__(learning_rate=0.1, max_epochs=1000, tol=1e-6)` | Lưu cấu hình, chưa có hệ số |
| `build_model()` | Reset trạng thái, chưa có hệ số |
| `fit(X_train, y_train)` | Batch gradient descent được cài đặt bằng thư viện NumPy, dừng sớm theo thay đổi tương đối của loss |
| `predict(X)` | Trả `X beta + b`, báo lỗi nếu chưa fit |
| `coef_`, `intercept_` | Hệ số và hệ số chặn, chỉ có sau fit |
| `export_params(path, feature_names=None)` | Xuất tham số đã học ra JSON |
| `history_`, `n_iter_`, `converged_` | Loss từng epoch, số epoch đã chạy, cờ dừng sớm |

Cách gọi theo mẫu chung:

```python
model = LinearRegressorModel()
model.build_model()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model.export_params("outputs/linear_regression_params.json", feature_names)
```

## 7. Siêu tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `learning_rate` | 0.1 | Bước cập nhật của gradient descent. Quá lớn thì phân kỳ, quá nhỏ thì hội tụ chậm |
| `max_epochs` | 1000 | Số epoch tối đa nếu không dừng sớm |
| `tol` | 1e-6 | Ngưỡng dừng sớm theo thay đổi tương đối của loss |

## 8. Dữ liệu sử dụng

King County, 88 đặc trưng sau tiền xử lý, mục tiêu `price`.
Xem [king-county-dataset.md](../king-county-dataset.md).
Pipeline chuẩn hóa `X` trước khi fit (xem mục 2); đây là điều kiện
để `learning_rate = 0.1` ổn định.

## 9. Chỉ số đánh giá

MSE, RMSE, MAE, R2, MAPE tính bằng `regression_metrics` trong
`src/house_price/evaluation.py`, trên cùng tập test với 3 mô hình
kia; cả 5 chỉ số đều có trong bảng so sánh do `compare_models` xuất
ra (`outputs/metrics.csv` có đủ cột MSE, RMSE, MAE, R2, MAPE). Không
chọn siêu tham số dựa trên điểm test.

## 10. Kết quả thực nghiệm

Kết quả trên tập test (3.241 dòng, pipeline chuẩn 88 đặc trưng),
Stage 03, xem [progress_results.md](../progress_results.md):

| Chỉ số | Kết quả |
|---|---:|
| MSE | 26.546.613.099,42 |
| RMSE | 162.931,31 USD |
| MAE | 97.840,49 USD |
| R2 | 0,8154 |

Lưu ý: một lần chạy kiểm chứng pipeline trước đó cho R2 0,8153
(RMSE 162.978, MAE 97.867); chênh lệch nhỏ do lần chạy khác nhau,
không phải hai mô hình khác nhau. Số 0,8154 trong bảng trên là số
chính thức của Stage 03.

## 11. Trực quan hóa và demo

Biểu đồ Actual vs Predicted; hệ số `coef_` có thể vẽ để xem đặc trưng
nào ảnh hưởng mạnh nhất (chỉ có ý nghĩa khi đặc trưng đã chuẩn hóa).
Tham số đã học xem trực tiếp trong file JSON do `export_params` ghi.

## 12. Ưu điểm và hạn chế

Ưu điểm:

- Huấn luyện nhanh, dễ giải thích từng hệ số, ít overfitting.
- Làm mốc tốt: mọi mô hình phức tạp hơn đều phải vượt qua nó.

Hạn chế:

- Chỉ học được quan hệ tuyến tính; kém khi giá phụ thuộc phi tuyến
  vào diện tích, vị trí, chất lượng.
- Nhạy cảm với ngoại lai (outlier) vì loss bình phương phạt nặng
  sai số lớn; dữ liệu giữ lại 1.146 dòng giá trên 1.129.575 USD
  nên cần đọc RMSE song song với MAE và R2.
- Gradient descent phụ thuộc thang đo của `X`: chỉ ổn định khi `X`
  đã chuẩn hóa; đưa `X` thô chưa chuẩn hóa có thể phân kỳ.
- Không học được tương tác giữa các đặc trưng (ví dụ diện tích lớn
  chỉ đắt khi ở khu vực tốt) nếu không tạo thêm đặc trưng tương tác.
