# Kiến trúc dự án

## Cấu trúc thư mục

| Vị trí | Vai trò |
|---|---|
| `data/kc_house_data_NaN/data_raw/` | Dữ liệu King County gốc, không sửa |
| `data/processed/` | Dữ liệu sinh ra khi chạy (gitignored) |
| `src/house_price/preprocessing.py` | Nạp dữ liệu, chia tập, chống rò rỉ dữ liệu |
| `src/house_price/models/` | Lớp mô hình theo mẫu 4 phương thức |
| `src/house_price/evaluation.py` | MSE, RMSE, MAE, R2, MAPE |
| `src/house_price/visualization.py` | Biểu đồ đánh giá và so sánh |
| `src/house_price/model_comparison.py` | Bảng đối chiếu 4 mô hình |
| `src/house_price/train_evaluate.py` | Điều phối tròn pipeline |
| `notebooks/` | Notebook demo 14 bước, báo cáo |
| `models/` | Mô hình đã huấn luyện (gitignored, trừ .gitkeep) |
| `outputs/` | Metric, dự đoán, biểu đồ (gitignored, trừ .gitkeep) |
| `docs/` | Mô tả, thiết kế, kế hoạch, giải thích, tiến trình |
| `docs/algorithms/` | Giải thích từng thuật toán |
| `docs/adr/` | Quyết định kiến trúc đã ghi lại |
| `README.md` | Điểm bắt đầu, hướng dẫn cài và chạy |

Chú ý: thư mục gốc `models/` chứa mô hình **đã huấn luyện** (artifact),
khác với `src/house_price/models/` chứa **code** định nghĩa mô hình.
Không đổi tên hay gộp hai thư mục này.

## Luồng dữ liệu

```text
CSV gốc (21.613 dòng, 22 cột)
  -> kỹ thuật đặc trưng: house_age, was_renovated, sqft_per_bedroom
  -> loại cột id/date/index khỏi X (giữ id làm groups để chia)
  -> tách X (19 cột đầu vào) và y (price)
  -> chia train/validation/test 70/15/15 theo nhóm id (seed 42)
  -> ColumnTransformer fit chỉ trên train: số bù median + StandardScaler,
     zipcode bù mode + one-hot (70 mã) -> 88 cột đầu ra
  -> 4 mô hình huấn luyện trên train, chọn trên validation
  -> đánh giá cuối cùng trên test, vẽ biểu đồ, lưu outputs/
```

Scaler và imputer chỉ học từ train; dùng cùng đối tượng đã fit để biến đổi
validation, test và nhà mới. Xem
[adr/003-chia-train-validation-test.md](adr/003-chia-train-validation-test.md).

## Hợp đồng module

### Preprocessing (`preprocessing.py`)

- `load_raw_dataset(path)` -> DataFrame thô.
- `engineer_features(df)` -> DataFrame đã tạo `house_age`, `was_renovated`,
  `sqft_per_bedroom` (không fit gì, xem `adr/006-feature-engineering-va-zipcode.md`).
- `clean_dataset(df)` -> DataFrame đã kỹ thuật đặc trưng và loại cột thừa,
  CHƯA bù khuyết hay chuẩn hóa (hai việc này nằm trong preprocessor).
- `split_features_target(df)` -> `(X, y)`, `y` là cột `price`.
- `split_train_val_test(X, y, groups=None)` -> 6 đối tượng theo tỷ lệ
  70/15/15, seed 42. Khi truyền `groups` (mã `id`) thì chia theo nhóm
  bằng `GroupShuffleSplit`, xem `adr/005-chia-theo-nhom-id.md`.
- `build_preprocessor(X_train)` -> `ColumnTransformer` đã fit trên train
  (số: median + `StandardScaler`; `zipcode`: mode + one-hot).
- `output_feature_names(preprocessor)` -> tên 88 cột sau mã hóa.
- `prepare_data(path)` -> dict đầy đủ 3 tập đã chuẩn hóa + preprocessor,
  thêm `input_columns` (19 cột đầu vào trước mã hóa) và `id_train`,
  `id_val`, `id_test` (chỉ để kiểm chứng chia theo nhóm).
- Hằng số dùng chung: `TARGET_COLUMN = "price"`,
  `DROP_COLUMNS = ["Unnamed: 0", "id", "date"]`,
  `FEATURE_COLUMNS` (19 cột, xem `docs/king-county-dataset.md`),
  `NUMERIC_COLUMNS` (18 cột số sau kỹ thuật đặc trưng),
  `CATEGORICAL_COLUMNS = ["zipcode"]`.

### Mô hình (`models/`)

Mỗi class tuân theo cùng mẫu:

```python
model = MLPRegressorModel()
model.build_model()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

| Phương thức | Công việc |
|---|---|
| `__init__(...)` | Khai báo cấu hình và biến trạng thái, chưa học |
| `build_model()` | Tạo estimator/kiến trúc, reset trạng thái |
| `fit(X_train, y_train)` | Huấn luyện trên train, lưu lịch sử nếu có |
| `predict(X_test)` | Trả giá trị dự đoán, báo lỗi nếu chưa fit |

`self` là đối tượng hiện tại; các phương thức dùng chung biến `self`.
Phải build trước fit, fit trước predict. Build lại sẽ xóa trạng thái đã học.
`__init__.py` trong package chỉ đánh dấu package và export tên class;
khác với phương thức `__init__()` của class.

### Đánh giá (`evaluation.py`)

- `regression_metrics(y_true, y_pred)` -> dict `mse`, `rmse`, `mae`, `r2`, `mape`.
- `mape` trả `None` khi có giá trị thật bằng 0 (tránh chia cho 0).
- MSE, RMSE, MAE càng nhỏ càng tốt; R2 càng lớn càng tốt, có thể âm.

### So sánh (`model_comparison.py`)

- `compare_models(results)` -> DataFrame sắp theo MSE tăng dần,
  các cột: model, mse, rmse, mae, r2.
- Mỗi mô hình dùng cùng tập test và cùng quy trình đánh giá.

### Điều phối (`train_evaluate.py`)

- `main()` chạy: nạp dữ liệu -> chia tập -> build 4 mô hình -> train ->
  predict -> đánh giá -> so sánh -> lưu `outputs/metrics.csv`.
- Không viết logic mô hình hay metric trực tiếp trong file này;
  chỉ gọi các module chuyên trách.

## Quyết định kiến trúc (tóm tắt)

- **Modular monolith, kiến trúc phân lớp đơn giản:** nhóm 4 người, một
  đơn vị triển khai, dữ liệu dùng chung. Không microservices, không
  database, không API (xem `adr/001-kien-truc-modular-monolith.md`).
- **Giữ cấu trúc gốc có sẵn** (`data/`, `docs/`, `models/`, `notebooks/`,
  `src/`), chỉ sắp xếp lại bên trong `src/` và bổ sung thư mục thiếu.
- **Notebook chỉ demo:** logic dùng lại nằm trong `src/`, notebook gọi
  module đó để demo và thí nghiệm, không viết class riêng trong notebook.
