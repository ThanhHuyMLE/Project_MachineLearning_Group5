# Random Forest Regression (Hồi quy rừng ngẫu nhiên)

## 1. Mục đích

Random Forest là mô hình tập hợp (ensemble): huấn luyện nhiều cây
quyết định trên các mẫu bootstrap khác nhau, rồi lấy trung bình dự
đoán. Mục đích là giảm phương sai (variance) của cây đơn lẻ mà vẫn
giữ khả năng học quan hệ phi tuyến. Đây là mô hình đối chiếu mạnh
nhất trên dữ liệu bảng trong dự án này, nên nó là thước đo quan trọng
nhất để đánh giá xem MLP có thực sự cạnh tranh được hay không.

## 2. Đầu vào và đầu ra

- Đầu vào: ma trận `X` có dạng `(N, 88)` sau tiền xử lý. Như cây đơn
  lẻ, rừng không yêu cầu chuẩn hóa, nhưng pipeline vẫn chuẩn hóa để
  dùng chung một tập dữ liệu cho cả 4 mô hình.
- Đầu ra: vector `y_hat` có dạng `(N,)`, giá nhà dự đoán tính bằng
  USD, bằng trung bình cộng dự đoán của tất cả các cây trong rừng.
- Mục tiêu huấn luyện: `y` là cột `price`, giữ đơn vị USD.

## 3. Mô hình hoạt động thế nào

Bagging: mỗi cây học trên một mẫu bootstrap (lấy mẫu có hoàn lại `n`
dòng từ tập train có `n` dòng), nên mỗi cây thấy một tập dữ liệu hơi
khác nhau và có khoảng 1/3 số dòng không xuất hiện trong mẫu của nó
(out-of-bag). Tại mỗi lần chia, cây chỉ xét một tập con ngẫu nhiên
các đặc trưng khi `max_features` được đặt, làm các cây càng khác nhau
hơn. Dự đoán cuối là trung bình cộng dự đoán của các cây. Trung bình
nhiều cây giảm phương sai mà độ chệch (bias) hầu như không tăng, vì
mỗi cây vẫn học đủ sâu trên dữ liệu của nó.

## 4. Nền tảng toán học

Dự đoán của rừng với `T` cây:

```text
y_hat(x) = (1/T) * sum_t y_hat_t(x)
```

Trung bình làm giảm phương sai: nếu các cây có phương sai `sigma^2`
và tương quan trung bình `rho`, phương sai của trung bình xấp xỉ:

```text
Var(trung bình) ~ rho * sigma^2 + (1 - rho) * sigma^2 / T
```

Giảm tương quan giữa các cây (nhờ lấy mẫu bootstrap và chọn đặc trưng
ngẫu nhiên) làm giảm `rho`, từ đó giảm phương sai chung. Khi `T` đủ
lớn, số hạng `(1 - rho) * sigma^2 / T` tiến về 0 và phương sai còn
lại tiến về `rho * sigma^2`: đó là lý do thêm cây chỉ cải thiện đến
một giới hạn rồi bão hòa.

Xác suất một dòng cụ thể không lọt vào một mẫu bootstrap cỡ `n`
lấy có hoàn lại là `(1 - 1/n)^n`, tiến về `1/e`, khoảng 36,8% khi
`n` lớn. Những dòng này gọi là out-of-bag của cây đó. Code hiện tại
không tính điểm out-of-bag; con số này chỉ để hiểu vì sao mỗi cây
thấy dữ liệu khác nhau.

## 5. Mã giả

Mã giả dưới đây khớp với `fit` và `predict` trong
`src/house_price/models/random_forest.py` và cây trong
`src/house_price/models/decision_tree.py`.

```text
ĐẦU VÀO: X_train (N x 88), y_train, n_estimators = 100,
         max_depth, min_samples_split = 2, min_samples_leaf = 1,
         max_features (mặc định: xét mọi đặc trưng),
         random_state = 42
ĐẦU RA: y_pred; rừng lưu trong trees_ (danh sách cây)

THỦ TỤC INIT(...):
    lưu cấu hình; trees_ <- RỖNG

THỦ TỤC BUILD_MODEL():
    tạo rừng rỗng và reset trạng thái đã học

THỦ TỤC FIT(X_train, y_train):
    kiểm tra model đã build, shape hợp lệ, dữ liệu hữu hạn
    seed_sequence <- SeedSequence(random_state)
    child_sequences <- seed_sequence.spawn(n_estimators)
    trees_ <- []
    VỚI MỖI child_sequence trong child_sequences:
        rng <- default_rng(child_sequence)
        bootstrap_indices <- rng.integers(0, n_samples, size=n_samples)
        tree_seed <- rng.integers(0, 2^31 - 1)
        tree <- DecisionTreeRegressorModel(
            max_depth, min_samples_split, min_samples_leaf,
            max_features, random_state=tree_seed)
        tree.build_model()
        tree.fit(X_train[bootstrap_indices], y_train[bootstrap_indices])
        trees_.append(tree)
    fitted <- True

HÀM PREDICT(X):
    NẾU model chưa fit: BÁO LỖI, DỪNG
    TRẢ VỀ trung bình cộng predict(X) của mọi cây trong trees_
```

Cách sinh số ngẫu nhiên (tái lập được và khác nhau giữa các cây):
`np.random.SeedSequence(random_state)` được `spawn` thành
`n_estimators` dãy con, mỗi cây nhận một dãy con riêng. Từ dãy con
đó tạo một `np.random.Generator`: một phần rút chỉ số bootstrap,
phần còn lại rút hạt giống `random_state` truyền cho cây. Nhờ vậy
hai lần `fit` với cùng seed cho cùng rừng, và thứ tự cây không phụ
thuộc vào trạng thái RNG toàn cục.

## 6. Cài đặt Python

File: `src/house_price/models/random_forest.py`,
class `RandomForestRegressorModel`, dùng lại
`DecisionTreeRegressorModel` trong
`src/house_price/models/decision_tree.py`. Rừng và từng cây đều viết
tay bằng NumPy; không dùng `sklearn.ensemble.RandomForestRegressor`.
Xem [adr/007-mo-hinh-code-tay.md](../adr/007-mo-hinh-code-tay.md).

| Phương thức | Công việc |
|---|---|
| `__init__(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=42)` | Lưu cấu hình, chưa có rừng |
| `build_model()` | Tạo rừng rỗng, reset trạng thái đã học |
| `fit(X_train, y_train)` | Huấn luyện `n_estimators` cây, mỗi cây trên một mẫu bootstrap riêng |
| `predict(X)` | Trung bình dự đoán các cây, báo lỗi nếu chưa fit |
| `trees_`, `n_features_in_` | Danh sách cây đã học, số đặc trưng lúc fit |

Lưu ý về `max_features`: mặc định trong code là `None`, tức mỗi lần
chia của từng cây xét mọi đặc trưng (chỉ bootstrap làm các cây khác
nhau). Muốn bật lấy mẫu đặc trưng theo từng lần chia thì truyền
`max_features="sqrt"` (mỗi lần chia lấy mẫu ngẫu nhiên
`sqrt(n_features)` đặc trưng), hoặc int/float theo quy ước của cây
quyết định. Giá trị mặc định này khác với quy ước của scikit-learn
(mặc định 1.0, tức mọi đặc trưng, đối với hồi quy); khi so sánh với
tài liệu bên ngoài cần ghi rõ mặc định đang dùng trong repo.

## 7. Siêu tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `n_estimators` | 100 | Số cây trong rừng. Nhiều cây hơn thì lỗi giảm dần rồi bão hòa, đổi lại thời gian huấn luyện tăng tuyến tính theo số cây |
| `max_depth` | None | Giới hạn độ sâu của từng cây, truyền thẳng sang cây quyết định |
| `min_samples_split` | 2 | Truyền thẳng sang cây quyết định |
| `min_samples_leaf` | 1 | Truyền thẳng sang cây quyết định |
| `max_features` | None | Số đặc trưng xét tại mỗi lần chia của từng cây (xem mục 6) |
| `random_state` | 42 | Hạt giống điều khiển toàn bộ tính ngẫu nhiên của rừng |

## 8. Dữ liệu sử dụng

King County, 88 đặc trưng sau tiền xử lý, mục tiêu `price`.
Xem [king-county-dataset.md](../king-county-dataset.md).

## 9. Chỉ số đánh giá

MSE, RMSE, MAE, R2, MAPE tính bằng `regression_metrics` trong
`src/house_price/evaluation.py`, trên cùng tập test với 3 mô hình
kia; cả 5 chỉ số đều có trong bảng so sánh do `compare_models` xuất
ra (`outputs/metrics.csv` có đủ cột MSE, RMSE, MAE, R2, MAPE). Không
chọn siêu tham số dựa trên điểm test.

## 10. Kết quả thực nghiệm

Kết quả đo thật ngày 17/11/2026 bằng lệnh `PYTHONPATH=src python -m house_price.train_evaluate --evaluation-split test --output-dir outputs` (tổng thời gian cả 4 mô hình 294,13 giây); môi trường: laptop Key, Windows 10 (bản dựng 10.0.19045.6466), Python 3.13.3, venv `venv-ml-win`, numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2.
Bảng benchmark chính thức của Stage 04 (pipeline chuẩn 88 đặc trưng,
cùng tập test 3.241 dòng, `n_estimators=100`, `max_depth=None`,
`max_features="sqrt"`, `random_state=42`). Xem
[progress_results.md](../progress_results.md).

| Chỉ số | Kết quả |
|---|---:|
| Thời gian fit | 251,47 giây |
| MSE | 19.584.927.856,05 |
| RMSE | 139.946,16 |
| MAE | 77.253,52 |
| R2 | 0,8638 |
| MAPE | 14,6857% |

Con số cũ R2 0,8802 của Random Forest đo trên 19 đặc trưng thô chưa
chuẩn hóa (thí nghiệm trong notebook) KHÔNG thuộc bảng này; xem mục
"Supplementary Experiment: Preprocessing Ablation Study" trong
[progress_results.md](../progress_results.md) và không so sánh trực
tiếp với số trên pipeline 88 đặc trưng.

Căn cứ chọn `max_features="sqrt"` trên validation (không dùng test),
đã xác nhận bằng thí nghiệm có kiểm soát
`scripts/compare_validation.py` (một lần `prepare_data`, seed 42, 88
cột; kết quả `outputs/validation_comparison.csv`, log
`D:\agent-work\qa\validation-comparison.log`): với 10 cây, `"sqrt"`
đạt R2 0,8363 trên validation (fit 12,80 giây), còn
`max_features=None` đạt R2 0,8720 nhưng fit 108,62 giây, ngoại suy
tuyến tính 100 cây khoảng 18 phút nên loại dù R2 cao hơn. Giới hạn
`max_depth=8` cho rừng (`sqrt`, 10 cây) làm R2 validation rơi xuống
0,7839 (fit 1,38 giây) nên cũng loại. Thời gian fit tuyệt đối phụ
thuộc phần cứng và tải máy tại thời điểm đo nên chỉ thứ tự tương đối
giữa các cấu hình mới là bằng chứng.

Giới hạn test-peeking: tập test đã từng bị xem trong quá trình phát triển trước đây nên các số trên là đo thật của lần chạy ngày 17/11/2026 nhưng không được diễn giải như đánh giá độc lập chưa từng bị xem trước.

## 11. Trực quan hóa và demo

Biểu đồ Actual vs Predicted; độ quan trọng đặc trưng (feature
importance) có thể vẽ để xem đặc trưng nào đóng góp nhiều nhất.
Lưu ý: rừng hiện tại không tổng hợp `feature_importances_` từ các
cây; muốn vẽ thì lấy trung bình `feature_importances_` của từng cây
trong `trees_` rồi mới vẽ.

## 12. Ưu điểm và hạn chế

Ưu điểm:

- Giảm overfitting so với cây đơn lẻ, thường chính xác nhất trong
  3 mô hình truyền thống trên dữ liệu bảng.
- Không cần chuẩn hóa, chịu được đặc trưng nhiễu tốt hơn cây đơn lẻ.

Hạn chế:

- Tốn bộ nhớ và thời gian huấn luyện lớn hơn cây đơn lẻ khoảng
  `n_estimators` lần; fit 100 cây CART được cài đặt bằng thư viện NumPy trên 15.129 dòng x 88 cột có thể mất nhiều phút.
- Khó giải thích từng dự đoán riêng lẻ (mô hình hộp đen tập hợp).
- Không ngoại suy tốt ngoài miền giá trị train: dự đoán bị chặn
  trong khoảng giá trị lá của các cây.
- Hiểu lầm thường gặp: nhiều cây hơn không phải lúc nào cũng tốt hơn
  vô hạn; lỗi giảm dần và bão hòa khi `n_estimators` đủ lớn (xem
  công thức phương sai ở mục 4).
