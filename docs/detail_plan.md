# Kế hoạch chi tiết

| Bước | Việc thực hiện | Hoàn thành khi |
|---|---|---|
| 1 | Đọc đề Bài tập 4 và [DESIGN.md](DESIGN.md) | Xác định bài toán Hồi quy, 4 mô hình, 14 bước |
| 2 | Kiểm tra dữ liệu King County | Có nguồn, số mẫu, phân bố giá, kiểm tra thiếu/trùng |
| 3 | Làm sạch và kỹ thuật đặc trưng | `house_age`, `was_renovated`, `sqft_per_bedroom`, loại cột id/date/index |
| 4 | Chia tập | Train/validation/test 70/15/15, seed 42, chia theo nhóm id, không trùng id giữa các tập |
| 5 | Chuẩn hóa | ColumnTransformer chỉ fit trên train (số: median + StandardScaler; zipcode: one-hot), 88 cột đầu ra cho cả 3 tập |
| 6 | Viết class mô hình | Đủ 4 phương thức, build riêng trước fit, predict báo lỗi khi chưa fit |
| 7 | Huấn luyện MLP | Fit trên train, theo dõi MSE/MAE trên validation |
| 8 | Huấn luyện 3 mô hình đối chiếu | Cùng tập train/val/test, cùng quy trình đánh giá |
| 9 | Đánh giá | Có MSE, RMSE, MAE, R2 (và MAPE nếu hợp lý) trên tập test |
| 10 | Trực quan hóa | Actual vs Predicted, loss train/val, biểu đồ so sánh metric |
| 11 | Demo | Nhà mới được tiền xử lý bằng pipeline đã fit và trả giá dự đoán |
| 12 | Viết docs | Các file có liên kết, mỗi file cố gắng dưới 200 dòng |
| 13 | Kiểm tra | `pytest` pass, `ruff check` sạch, pipeline chạy tròn đầu đến cuối |

## Thứ tự sử dụng class mô hình

```python
model = MLPRegressorModel()
model.build_model()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Tham số giữ cố định

- Chia tập: train/validation/test = 70/15/15, `random_state=42`,
  chia theo nhóm `id` bằng `GroupShuffleSplit`.
  Hồi quy không có nhãn lớp nên không stratify.
- Tiền xử lý: kỹ thuật đặc trưng `house_age`, `was_renovated`,
  `sqft_per_bedroom`; bù khuyết `bedrooms`, `bathrooms` bằng median
  của tập train; chuẩn hóa bằng `StandardScaler` fit trên train;
  `zipcode` mã hóa one-hot 70 mã.
- MLP mặc định: hidden layers (32, 16), activation relu, output linear,
  optimizer Adam, loss MSE, epochs 100, batch size 32 (theo đề bài).
- Không chỉnh siêu tham số dựa trên điểm test. Chỉ dùng validation để chọn.

## Thứ tự làm của 4 sinh viên (giảm xung đột)

1. **Sinh viên 1 trước:** hoàn thành `preprocessing.py` và hợp đồng dữ liệu.
   Các bạn khác chỉ đọc hợp đồng, không sửa file của nhau khi chưa thống nhất.
2. **Sinh viên 2 và 4 song song:** Sinh viên 2 làm `models/mlp.py`,
   Sinh viên 4 làm 3 mô hình đối chiếu. Hai nhóm file độc lập nhau.
3. **Sinh viên 3 cuối cùng:** ghép pipeline trong `train_evaluate.py`,
   đánh giá, vẽ biểu đồ, phân tích overfitting.
4. Mỗi người tự cập nhật tài liệu thuật toán của mình và ghi kết quả thật
   vào [progress_results.md](progress_results.md) sau mỗi công đoạn.

## Cập nhật sau mỗi công đoạn

Ghi việc làm, cách kiểm tra và kết quả thật vào
[progress_results.md](progress_results.md).
Nếu file tài liệu vượt 200 dòng thì tách theo chủ đề và bổ sung liên kết.
Không ghi kết quả chưa chạy như kết quả đã kiểm chứng.
