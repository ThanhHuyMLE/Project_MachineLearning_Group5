# Decision Tree Regression (Hồi quy cây quyết định)

## 1. Mục đích

Decision Tree chia không gian đặc trưng thành các vùng bằng các ngưỡng
cắt, mỗi vùng dự đoán bằng giá trung bình của các mẫu trong vùng.
Mô hình học được quan hệ phi tuyến và tương tác mà Linear Regression
bỏ lỡ, đồng thời mỗi quyết định chia đều đọc được trực tiếp nên cây
là mô hình dễ giải thích nhất trong 4 mô hình của dự án. Cây cũng là
khối xây dựng của Random Forest.

## 2. Đầu vào và đầu ra

- Đầu vào: ma trận `X` có dạng `(N, 88)` sau tiền xử lý. Cây không
  yêu cầu chuẩn hóa (ngưỡng chia chỉ so sánh thứ tự), nhưng pipeline
  vẫn chuẩn hóa để dùng chung một tập dữ liệu cho cả 4 mô hình.
- Đầu ra: vector `y_hat` có dạng `(N,)`, giá nhà dự đoán tính bằng USD.
- Mục tiêu huấn luyện: `y` là cột `price`, giữ đơn vị USD.
- Quy ước ngưỡng: mẫu có `x <= threshold` đi nhánh trái,
  `x > threshold` đi nhánh phải. Quy ước này áp dụng giống nhau ở cả
  `fit` và `predict`.

## 3. Mô hình hoạt động thế nào

Tại mỗi nút, thuật toán thử từng đặc trưng ứng viên và từng ngưỡng
chia, chọn cách chia làm giảm MSE có trọng số của hai nút con nhiều
nhất. Lặp lại việc chia cho từng nút con đến khi gặp điều kiện dừng
(đạt `max_depth`, nút còn ít mẫu hơn `min_samples_split`, mọi cách
chia đều vi phạm `min_samples_leaf`, hoặc không còn cách chia nào
giảm MSE quá `min_impurity_decrease`). Giá trị dự đoán tại lá là
trung bình mục tiêu của các mẫu trong lá. Khi dự đoán, mẫu đi từ gốc
xuống lá theo các ngưỡng và lấy giá trị của lá đó.

## 4. Nền tảng toán học

Độ không thuần khiết tại nút: phương sai của `y` trong nút,

```text
MSE(nút) = (1/n) * sum((y_i - y_mean)^2)
```

Mục tiêu khi chia nút thành trái/phải là tối thiểu hóa MSE có trọng
số của hai nút con:

```text
MSE_có_trọng_số = (n_trái * MSE_trái + n_phải * MSE_phải) / n_nút
```

tương đương với tối đa hóa mức giảm phương sai:

```text
Mức giảm = MSE(nút) - MSE_có_trọng_số
```

Ngưỡng chia là trung điểm giữa hai giá trị riêng biệt liên tiếp của
đặc trưng trong nút. Giá trị dự đoán tại lá:

```text
y_hat(lá) = trung bình y của các mẫu trong lá
```

Chiến lược tìm kiếm trong code: với mỗi đặc trưng, sắp xếp các giá
trị (O(n log n)) rồi dùng tổng tích lũy (cumulative sum) của `y` và
`y^2` để tính SSE của mọi vị trí chia ứng viên trong O(n), nên mỗi
nút tốn O(p * n log n) với p là số đặc trưng ứng viên và n là số mẫu
trong nút. Mọi ngưỡng giữa hai giá trị riêng biệt liên tiếp đều được
xét; không giới hạn ngưỡng theo phân vị và không dùng tìm kiếm
exhaustive naive O(p * n^2).

Tính xác định: thứ tự xét đặc trưng theo chỉ số tăng dần, ngưỡng theo
giá trị tăng dần; khi hòa MSE thì cách chia đầu tiên tốt nhất được
giữ (so sánh nghiêm ngặt `<`).

## 5. Mã giả

Mã giả dưới đây khớp với `fit`, `_best_split` và `predict` trong
`src/house_price/models/decision_tree.py`. Quy ước: chỉ số bắt đầu
từ 0; code Python gộp một số vòng lặp thành phép tính mảng và dùng
ngăn xếp thay cho đệ quy.

```text
ĐẦU VÀO: X_train (N x 88), y_train,
         max_depth (mặc định: không giới hạn),
         min_samples_split = 2, min_samples_leaf = 1,
         max_features (mặc định: xét mọi đặc trưng),
         min_impurity_decrease = 0.0, random_state = 42
ĐẦU RA: y_pred; cây lưu trong tree_ (dict lồng nhau)

THỦ TỤC INIT(...):
    lưu cấu hình; tree_ <- RỖNG

THỦ TỤC BUILD_MODEL():
    tạo trạng thái cây rỗng và reset trạng thái đã học

THỦ TỤC FIT(X_train, y_train):
    kiểm tra model đã build, shape hợp lệ, dữ liệu hữu hạn
    rng <- default_rng(random_state)
    stack <- [(gốc, toàn bộ chỉ số mẫu, depth = 0)]
    TRONG KHI stack không rỗng:
        (node, indices, depth) <- lấy ra khỏi stack
        node <- lá tạm: value <- mean(y[indices]),
                mse <- mean((y[indices] - value)^2),
                n_samples <- len(indices)
        NẾU n_samples < min_samples_split: TIẾP TỤC
        NẾU max_depth đặt VÀ depth >= max_depth: TIẾP TỤC
        NẾU mse <= 0: TIẾP TỤC
        best <- BEST_SPLIT(X[indices], y[indices], rng)
        NẾU best là None: TIẾP TỤC
        (feature, threshold, weighted_mse) <- best
        NẾU mse - weighted_mse <= min_impurity_decrease: TIẾP TỤC
        chia indices theo x <= threshold (trái) / x > threshold (phải)
        NẾU một phía rỗng: TIẾP TỤC
        node thành nút trong: lưu feature, threshold, left, right
        đẩy (right, ...) rồi (left, ...) vào stack
    tree_ <- gốc; tính feature_importances_ (chuẩn hóa về tổng 1)
    fitted <- True

HÀM BEST_SPLIT(X_node, y_node, rng):
    best <- None; best_weighted_mse <- +vô cực
    VỚI MỖI feature trong CANDIDATE_FEATURES:
        sắp xếp giá trị (stable) và y theo thứ tự đó
        vị trí chia ứng viên: ngay sau mỗi giá trị riêng biệt
        LOẠI vị trí vi phạm min_samples_leaf ở một phía
        dùng tổng tích lũy của y và y^2 để tính SSE trái/phải
        weighted_mse <- (SSE_trái + SSE_phải) / n_node
        NẾU weighted_mse nhỏ nhất tại nút này < best_weighted_mse:
            best <- (feature, trung điểm hai giá trị kề,
                     weighted_mse)
    TRẢ VỀ best hoặc None

HÀM PREDICT(X):
    NẾU model chưa fit: BÁO LỖI, DỪNG
    VỚI MỖI x: đi từ gốc theo ngưỡng đến lá, TRẢ VỀ value của lá
```

Cấu trúc cây lưu trong `tree_` (dict lồng nhau): nút lá có
`is_leaf=True` và `value`; nút trong có `feature`, `threshold`,
`left` và `right`. Mọi nút đều có `n_samples`, `mse` và `depth`.

## 6. Cài đặt Python

File: `src/house_price/models/decision_tree.py`,
class `DecisionTreeRegressorModel`. Đây là cây CART được cài đặt bằng thư viện NumPy; không dùng `sklearn.tree.DecisionTreeRegressor`. Xem
[adr/007-mo-hinh-code-tay.md](../adr/007-mo-hinh-code-tay.md).

| Phương thức | Công việc |
|---|---|
| `__init__(max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, min_impurity_decrease=0.0, random_state=42)` | Lưu cấu hình, chưa có cây |
| `build_model()` | Tạo trạng thái cây rỗng, reset trạng thái đã học |
| `fit(X_train, y_train)` | Xây cây bằng ngăn xếp (lặp, không đệ quy) để cây sâu không vượt giới hạn đệ quy của Python |
| `predict(X)` | Đi từ gốc xuống lá theo ngưỡng, báo lỗi nếu chưa fit |
| `_best_split(X_node, y_node, rng)` | Cách chia tốt nhất trong nút, hoặc None |
| `_candidate_features(n_features, rng)` | Chỉ số đặc trưng ứng viên tại một nút (có thể lấy mẫu khi `max_features` đặt) |
| `tree_`, `feature_importances_`, `n_features_in_` | Cây đã học, độ quan trọng đặc trưng (chuẩn hóa về tổng 1), số đặc trưng lúc fit |

## 7. Siêu tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `max_depth` | None (không giới hạn) | Độ sâu tối đa của cây. Giới hạn độ sâu là cách chống overfitting trực tiếp nhất |
| `min_samples_split` | 2 | Nút cần ít nhất bao nhiêu mẫu mới được xét chia |
| `min_samples_leaf` | 1 | Mỗi lá sau khi chia phải có ít nhất bao nhiêu mẫu |
| `max_features` | None | Số đặc trưng xét tại mỗi lần chia: None là mọi đặc trưng; `"sqrt"` là căn bậc hai số đặc trưng; int hoặc float theo quy ước trong code |
| `min_impurity_decrease` | 0.0 | Cách chia phải giảm MSE ít nhất bao nhiêu mới được chấp nhận |
| `random_state` | 42 | Hạt giống cho lấy mẫu đặc trưng khi `max_features` đặt; cùng seed cho cùng cây |

## 8. Dữ liệu sử dụng

King County, 88 đặc trưng sau tiền xử lý, mục tiêu `price`.
Xem [king-county-dataset.md](../king-county-dataset.md).
Cây không cần chuẩn hóa nhưng vẫn dùng chung pipeline 88 cột để
so sánh công bằng với 3 mô hình kia.

## 9. Chỉ số đánh giá

MSE, RMSE, MAE, R2, MAPE tính bằng `regression_metrics` trong
`src/house_price/evaluation.py`, trên cùng tập test với 3 mô hình
kia; cả 5 chỉ số đều có trong bảng so sánh do `compare_models` xuất
ra (`outputs/metrics.csv` có đủ cột MSE, RMSE, MAE, R2, MAPE). Không
chọn siêu tham số dựa trên điểm test.

## 10. Kết quả thực nghiệm

Kết quả thăm dò (chưa phải bảng benchmark chính thức của Stage 04;
bảng chính thức xem [progress_results.md](../progress_results.md)
sau khi benchmark chạy xong). Bảng dưới đây là số TRÊN TEST, chỉ
dùng để minh họa overfit, KHÔNG phải căn cứ chọn `max_depth`:

| Cấu hình | Thời gian fit | R2 trên test |
|---|---:|---:|
| `max_depth=8` | 1,12 giây | 0,7314 |
| Không giới hạn độ sâu | 32,29 giây | 0,6880 |

Hai dòng trên cho thấy cây không giới hạn độ sâu vừa chậm hơn nhiều
vừa tệ hơn trên test: cây học thuộc cả nhiễu trong train (overfit).
Căn cứ chọn `max_depth=8` là số trên validation đã xác nhận bằng
thí nghiệm có kiểm soát `scripts/compare_validation.py` (một lần
`prepare_data`, seed 42, 88 cột; kết quả
`outputs/validation_comparison.csv`): `max_depth=8` đạt R2 0,7198
(fit 0,63 giây), không giới hạn độ sâu đạt R2 0,6926 (fit 16,33 giây).

## 11. Trực quan hóa và demo

Biểu đồ Actual vs Predicted; có thể vẽ cây (giới hạn độ sâu) để minh
họa cách chia, chỉ để học, không vẽ cây đầy đủ vì quá lớn.
`feature_importances_` có thể vẽ để xem đặc trưng nào đóng góp nhiều
nhất vào việc giảm MSE (tổng đóng góp chuẩn hóa về 1).

## 12. Ưu điểm và hạn chế

Ưu điểm:

- Không cần giả định phân phối, không cần chuẩn hóa.
- Dễ hiểu từng quyết định chia; làm nền tảng cho Random Forest.
- Học được quan hệ phi tuyến và tương tác.

Hạn chế:

- Rất dễ overfitting nếu không giới hạn độ sâu hoặc số mẫu tối thiểu
  mỗi lá: cây học thuộc cả nhiễu trong train (xem mục 10).
- Nhạy cảm với thay đổi nhỏ trong dữ liệu: đổi một ngưỡng chia có
  thể đổi cả cây con bên dưới.
- Dự đoán là hàm bậc thang (hằng số theo vùng), không mượt như MLP;
  không ngoại suy ngoài miền giá trị train (dự đoán bị chặn trong
  khoảng giá trị lá).
- Cây CART được cài đặt bằng thư viện NumPy chạy chậm hơn nhiều so với cây CART của scikit-learn khi không giới hạn độ sâu (32,29 giây cho một cây trên
  15.129 dòng x 88 cột).
