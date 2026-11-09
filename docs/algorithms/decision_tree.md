# Decision Tree Regression (Hồi quy cây quyết định)

## 1. Giới thiệu và mục đích

Decision Tree chia không gian đặc trưng thành các vùng bằng các ngưỡng
(cắt ngang), mỗi vùng dự đoán bằng giá trung bình của các mẫu trong vùng.
Mô hình học được quan hệ phi tuyến và tương tác mà Linear Regression bỏ lỡ.

## 2. Đầu vào và đầu ra

- Đầu vào: vector `X` gồm 18 đặc trưng (không bắt buộc chuẩn hóa, nhưng
  pipeline vẫn chuẩn hóa để dùng chung một tập dữ liệu).
- Đầu ra: một số thực `y_hat` là giá nhà dự đoán (USD).

## 3. Mô hình hoạt động thế nào

Tại mỗi nút, thuật toán thử từng đặc trưng và ngưỡng, chọn cách chia làm
giảm phương sai trong nút nhiều nhất. Lặp lại đệ quy đến khi dừng điều kiện
(hết mẫu, hết đặc trưng, đạt `max_depth`). Dự đoán bằng cách đi từ gốc
xuống lá theo các ngưỡng và lấy giá trị của lá.

## 4. Nền tảng toán học

Độ không thuần khiết tại nút: phương sai của `y` trong nút,
`Var = (1/n) * sum((y_i - y_mean)^2)`.
Mục tiêu khi chia nút thành trái/phải:

```text
Giảm phương sai = Var(nút) - (n_trái/n) Var(trái) - (n_phải/n) Var(phải)
```

Chọn cách chia có mức giảm lớn nhất. Giá trị dự đoán tại lá:

```text
y_hat(lá) = trung bình y của các mẫu trong lá
```

## 5. Mã giả

Xem [pseudocode.md](../pseudocode.md) mục Decision Tree Regression.

## 6. Cài đặt Python

File: `src/house_price/models/decision_tree.py`,
class `DecisionTreeRegressorModel` (TODO: Sinh viên 4 điền).

| Phương thức | Công việc |
|---|---|
| `__init__(max_depth=None, ...)` | Lưu cấu hình, chưa có cây |
| `build_model()` | Tạo `sklearn DecisionTreeRegressor`, reset trạng thái |
| `fit(X_train, y_train)` | Xây cây bằng cách chia đệ quy |
| `predict(X_test)` | Đi từ gốc xuống lá, báo lỗi nếu chưa fit |

## 7. Dữ liệu sử dụng

King County, 18 đặc trưng, mục tiêu `price`.
Xem [king-county-dataset.md](../king-county-dataset.md).

## 8. Chỉ số đánh giá

MSE, RMSE, MAE, R2 trên cùng tập test với 3 mô hình kia.

## 9. Trực quan hóa và demo

Biểu đồ Actual vs Predicted; có thể vẽ cây (giới hạn độ sâu) để minh họa
cách chia, chỉ để học, không vẽ cây đầy đủ vì quá lớn.

## 10. Kết quả thực nghiệm

**CHƯA THỰC HIỆN.** Sinh viên 4 điền số thật sau khi chạy.
Chú ý ghi lại `max_depth` thực tế dùng, vì cây không giới hạn độ sâu
dễ overfitting trên tập train.

## 11. Hạn chế và thảo luận

- Rất dễ overfitting nếu không giới hạn độ sâu hoặc số mẫu tối thiểu
  mỗi lá: cây học thuộc cả nhiễu trong train.
- Nhạy cảm với thay đổi nhỏ trong dữ liệu: đổi một ngưỡng chia có thể
  đổi cả cây con bên dưới.
- Dự đoán là hàm bậc thang (hằng số theo vùng), không mượt như MLP.
- Ưu điểm: không cần giả định phân phối, dễ hiểu từng quyết định chia,
  làm nền tảng cho Random Forest.
