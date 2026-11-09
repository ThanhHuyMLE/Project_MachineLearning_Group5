# Random Forest Regression (Hồi quy rừng ngẫu nhiên)

## 1. Giới thiệu và mục đích

Random Forest là mô hình tập hợp (ensemble): huấn luyện nhiều cây quyết định
độc lập trên các mẫu bootstrap khác nhau, rồi lấy trung bình dự đoán.
Mục đích là giảm phương sai (variance) của cây đơn lẻ mà vẫn giữ khả năng
học quan hệ phi tuyến.

## 2. Đầu vào và đầu ra

- Đầu vào: vector `X` gồm 18 đặc trưng.
- Đầu ra: một số thực `y_hat` là giá nhà dự đoán (USD), bằng trung bình
  dự đoán của tất cả các cây trong rừng.

## 3. Mô hình hoạt động thế nào

Bagging: mỗi cây học trên một mẫu bootstrap (lấy mẫu có hoàn lại) của train.
Tại mỗi nút chia, cây chỉ xét một tập con ngẫu nhiên các đặc trưng, làm các
cây khác nhau hơn. Dự đoán cuối là trung bình cộng dự đoán của các cây.

## 4. Nền tảng toán học

Dự đoán của rừng với `T` cây:

```text
y_hat(x) = (1/T) * sum_t y_hat_t(x)
```

Trung bình làm giảm phương sai: nếu các cây có phương sai `sigma^2` và
tương quan trung bình `rho`, phương sai của trung bình xấp xỉ:

```text
Var(trung bình) ~ rho * sigma^2 + (1 - rho) * sigma^2 / T
```

Giảm tương quan giữa các cây (nhờ lấy mẫu bootstrap và chọn đặc trưng
ngẫu nhiên) làm giảm `rho`, từ đó giảm phương sai chung.

## 5. Mã giả

Xem [pseudocode.md](../pseudocode.md) mục Random Forest Regression.

## 6. Cài đặt Python

File: `src/house_price/models/random_forest.py`,
class `RandomForestRegressorModel` (TODO: Sinh viên 4 điền).

| Phương thức | Công việc |
|---|---|
| `__init__(n_estimators=100, ...)` | Lưu cấu hình, chưa có rừng |
| `build_model()` | Tạo `sklearn RandomForestRegressor`, reset trạng thái |
| `fit(X_train, y_train)` | Huấn luyện `n_estimators` cây |
| `predict(X_test)` | Trung bình dự đoán các cây, báo lỗi nếu chưa fit |

## 7. Dữ liệu sử dụng

King County, 18 đặc trưng, mục tiêu `price`.
Xem [king-county-dataset.md](../king-county-dataset.md).

## 8. Chỉ số đánh giá

MSE, RMSE, MAE, R2 trên cùng tập test với 3 mô hình kia.

## 9. Trực quan hóa và demo

Biểu đồ Actual vs Predicted; độ quan trọng đặc trưng (feature importance)
có thể vẽ để xem đặc trưng nào đóng góp nhiều nhất.

## 10. Kết quả thực nghiệm

**CHƯA THỰC HIỆN.** Sinh viên 4 điền số thật sau khi chạy, kèm theo
`n_estimators` thực tế dùng và thời gian huấn luyện (rừng 100 cây
chậm hơn cây đơn lẻ đáng kể).

## 11. Hạn chế và thảo luận

- Tốn bộ nhớ và thời gian huấn luyện lớn hơn cây đơn lẻ.
- Khó giải thích từng dự đoán riêng lẻ (mô hình hộp đen tập hợp).
- Không ngoại suy tốt ngoài miền giá trị train: dự đoán bị chặn
  trong khoảng giá trị lá của các cây.
- Ưu điểm: giảm overfitting so với cây đơn lẻ, thường chính xác nhất
  trong 3 mô hình truyền thống trên dữ liệu bảng.
- Hiểu lầm thường gặp: nhiều cây hơn không phải lúc nào cũng tốt hơn
  vô hạn; lỗi giảm dần và bão hòa khi `n_estimators` đủ lớn.
