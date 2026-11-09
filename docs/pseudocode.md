# Mã giả thuật toán

Quy ước: chỉ số bắt đầu từ 0. Mã giả trình bày vòng lặp từng phần tử;
code Python có thể gộp một số vòng lặp thành phép tính mảng.
Mỗi mô hình tuân theo: INIT -> BUILD_MODEL -> FIT -> PREDICT.
Chi tiết toán học và giải thích đầy đủ xem `docs/algorithms/`.

## Quy trình chung (4 mô hình)

```text
ĐẦU VÀO: tập dữ liệu King County, 19 cột đầu vào X (18 số + zipcode,
88 cột sau mã hóa), mục tiêu y = price
ĐẦU RA: dự đoán y_pred trên tập test, bảng MSE/RMSE/MAE/R2

1. Nạp CSV, kỹ thuật đặc trưng (house_age, was_renovated,
   sqft_per_bedroom), loại cột [Unnamed: 0, id, date] khỏi X
2. Tách X, y; chia train/validation/test 70/15/15 theo nhóm id, seed 42
3. ColumnTransformer: fit trên train (số: median + StandardScaler;
   zipcode: mode + one-hot), transform val/test
4. Với mỗi mô hình: build_model -> fit(train) -> predict(test)
6. Tính MSE, RMSE, MAE, R2 cho từng mô hình trên cùng tập test
7. So sánh và vẽ biểu đồ Actual vs Predicted
```

## MLP Regression

```text
ĐẦU VÀO: X_train (N x 88), y_train, hidden (32, 16), epochs 100, batch 32
ĐẦU RA: y_pred

THỦ TỤC INIT(hidden_layers, learning_rate):
    lưu cấu hình; model <- RỖNG; history <- RỖNG

THỦ TỤC BUILD_MODEL():
    model <- mạng: Dense(32, relu) -> Dense(16, relu) -> Dense(1, linear)
    biên dịch với optimizer Adam, loss MSE, metric MAE

THỦ TỤC FIT(X_train, y_train, X_val, y_val):
    NẾU model = RỖNG: BÁO LỖI, DỪNG
    Với mỗi epoch:
        xáo trộn train, chia mini-batch 32
        Với mỗi batch: lan truyền tiến -> tính loss MSE
            -> lan truyền ngược -> Adam cập nhật trọng số
        ghi loss/MAE trên train và validation vào history

HÀM PREDICT(X):
    NẾU model chưa fit: BÁO LỖI, DỪNG
    TRẢ VỀ model dự đoán cho từng dòng X
```

## Linear Regression

```text
ĐẦU VÀO: X_train (N x 88), y_train
ĐẦU RA: y_pred

THỦ TỤC INIT(): lưu hệ số beta <- RỖNG; intercept <- RỖNG
THỦ TỤC BUILD_MODEL(): tạo bộ ước lượng LinearRegression; reset trạng thái
THỦ TỤC FIT(X_train, y_train):
    giải bài toán bình phương tối thiểu:
    beta <- argmin ||y - X beta||^2
HÀM PREDICT(X): TRẢ VỀ X beta + intercept
```

## Decision Tree Regression

```text
ĐẦU VÀO: X_train, y_train, max_depth (mặc định: không giới hạn)
ĐẦU RA: y_pred

THỦ TỤC INIT(max_depth): lưu max_depth; tree <- RỖNG
THỦ TỤC BUILD_MODEL(): tạo DecisionTreeRegressor; reset trạng thái
THỦ TỤC FIT(X_train, y_train):
    đệ quy chia nút: chọn đặc trưng và ngưỡng làm giảm MSE nhiều nhất
    dừng khi hết mẫu, hết đặc trưng, hoặc đạt max_depth
    mỗi lá lưu giá trị trung bình của y trong lá đó
HÀM PREDICT(X):
    Với mỗi x: đi từ gốc theo ngưỡng chia đến lá, trả giá trị của lá
```

## Random Forest Regression

```text
ĐẦU VÀO: X_train, y_train, n_estimators (mặc định 100)
ĐẦU RA: y_pred

THỦ TỤC INIT(n_estimators): lưu số cây; forest <- RỖNG
THỦ TỤC BUILD_MODEL(): tạo RandomForestRegressor; reset trạng thái
THỦ TỤC FIT(X_train, y_train):
    Với mỗi cây: lấy mẫu bootstrap từ train, huấn luyện cây quyết định
    (mỗi nút chỉ xét một tập con ngẫu nhiên các đặc trưng)
HÀM PREDICT(X):
    TRẢ VỀ trung bình dự đoán của tất cả các cây
```
