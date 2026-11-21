# Kiến trúc dự án

## Cấu trúc thư mục

| Vị trí | Vai trò |
|---|---|
| `data/kc_house_data_NaN/data_raw/` | Dữ liệu King County gốc, không sửa |
| `data/processed/` | Dữ liệu sinh ra khi chạy (gitignored) |
| `src/house_price/preprocessing.py` | Nạp dữ liệu, chia tập, chống rò rỉ dữ liệu |
| `src/house_price/models/` | Lớp mô hình theo mẫu 4 phương thức: MLP chính thức dùng PyTorch, 3 mô hình còn lại được cài đặt bằng thư viện NumPy |
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
  -> 4 mô hình chính thức huấn luyện trên train, chọn trên validation
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
  thêm `feature_columns` (tên cột sau mã hóa, 88 cột),
  `input_columns` (19 cột đầu vào trước mã hóa) và `id_train`,
  `id_val`, `id_test` (chỉ để kiểm chứng chia theo nhóm).
- Hằng số dùng chung: `TARGET_COLUMN = "price"`,
  `DROP_COLUMNS = ["Unnamed: 0", "id", "date"]`,
  `FEATURE_COLUMNS` (19 cột, xem `docs/king-county-dataset.md`),
  `NUMERIC_COLUMNS` (18 cột số sau kỹ thuật đặc trưng),
  `CATEGORICAL_COLUMNS = ["zipcode"]`, `RANDOM_STATE = 42`,
  `TEST_SIZE = 0.15`, `VAL_SIZE = 0.15`.

### Mô hình (`models/`)

MLP chính thức dùng PyTorch (`MLPTorchRegressorModel`, xem
`adr/008-mlp-pytorch.md`); Linear, Decision Tree và Random Forest
được cài đặt bằng thư viện NumPy, không dùng estimator có sẵn của
scikit-learn (xem `adr/007-mo-hinh-numpy.md`). scikit-learn chỉ dùng
cho tiền xử lý và metrics. Mỗi class tuân theo cùng mẫu:

```python
model = MLPTorchRegressorModel()
model.build_model(input_size=X_train.shape[1])  # riêng MLP cần input_size
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
y_pred = model.predict(X_test)
```

| Phương thức | Công việc |
|---|---|
| `__init__(...)` | Khai báo cấu hình và biến trạng thái, chưa học |
| `build_model(...)` | Tạo kiến trúc, reset trạng thái. MLP bắt buộc nhận `input_size`; 3 mô hình còn lại không tham số |
| `fit(X_train, y_train)` | Huấn luyện trên train, lưu lịch sử nếu có |
| `predict(X_test)` | Trả giá trị dự đoán, báo lỗi nếu chưa fit |

`self` là đối tượng hiện tại; các phương thức dùng chung biến `self`.
Phải build trước fit, fit trước predict. Build lại sẽ xóa trạng thái đã học.
`__init__.py` trong package chỉ đánh dấu package và export tên class;
khác với phương thức `__init__()` của class.

Hợp đồng riêng từng mô hình:

- `MLPTorchRegressorModel` (`models/mlp_torch.py`): MLP chính thức
  dùng PyTorch, Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1,
  Linear), He init dạng chuẩn, DataLoader batch 32, MSELoss, Adam
  (autograd), lr 0.001, epochs 100. `fit(X_train, y_train, X_val=None,
  y_val=None)` theo dõi `history_` gồm `loss` và `val_loss`; sau fit
  tự lưu state_dict vào `save_path` (mặc định
  `outputs/mlp_torch_parameters.pt`); `load(path)` nạp bằng
  `torch.load(..., map_location="cpu", weights_only=True)`.
- `MLPRegressorModel` (`models/mlp.py`): bản NumPy legacy, cùng kiến
  trúc và cấu hình, forward/backward/Adam viết tay bằng NumPy.
  Ngoài bảng benchmark chính thức; chỉ chạy kèm bằng cờ
  `--include-numpy-mlp`, lưu JSON vào
  `outputs/mlp_parameters.json`.
- `MLPTorchManualModel` (`models/mlp_torch_manual.py`): bản PyTorch
  viết tay mang tính giáo dục, cùng kiến trúc và cấu hình; forward/
  backward/Adam viết tay bằng tensor torch float32,
  `requires_grad=False`, không dùng autograd hay `torch.optim`.
  Không nằm trong benchmark và không có trong `train_evaluate.py`;
  lưu danh sách tensor weights/biases vào
  `outputs/mlp_torch_manual_parameters.pt`.
- `LinearRegressorModel` (`models/linear_regression.py`): batch gradient
  descent được cài đặt bằng thư viện NumPy, `learning_rate=0.1`, `max_epochs=1000`, `tol=1e-6`
  tương đối. `coef_`, `intercept_`, `history_`, `n_iter_`,
  `converged_`; `export_params(path, feature_names)` xuất JSON.
- `DecisionTreeRegressorModel` (`models/decision_tree.py`): cây CART được cài đặt bằng thư viện NumPy, xây bằng ngăn xếp (lặp, không đệ quy), ngưỡng là trung điểm
  giá trị riêng biệt liên tiếp. `tree_` là dict lồng nhau;
  `feature_importances_` chuẩn hóa về tổng 1.
- `RandomForestRegressorModel` (`models/random_forest.py`): bagging trên cây CART, cả hai đều được cài đặt bằng thư viện NumPy. Bootstrap qua `SeedSequence` spawn,
  mỗi cây một dãy con riêng. `max_features` mặc định `"sqrt"` (mỗi
  lần chia lấy mẫu ngẫu nhiên sqrt số đặc trưng);
  `n_estimators=100` mặc định.
  `predict` trả trung bình dự đoán các cây trong `trees_`.
- `BaseRegressorModel` (`models/base.py`): lớp cơ sở nhẹ, chỉ quản lý
  trạng thái `_is_built`/`_is_fitted` và báo lỗi khi gọi sai thứ tự.

### Đánh giá (`evaluation.py`)

- `regression_metrics(y_true, y_pred)` -> dict `mse`, `rmse`, `mae`, `r2`, `mape`.
- `mape` trả `None` khi có giá trị thật bằng 0 (tránh chia cho 0).
- MSE, RMSE, MAE càng nhỏ càng tốt; R2 càng lớn càng tốt, có thể âm.

### So sánh (`model_comparison.py`)

- `compare_models(results)` -> DataFrame sắp theo MSE tăng dần,
  các cột: model, mse, rmse, mae, r2, mape.
- Mọi mô hình dùng cùng tập đánh giá đã chọn và cùng quy trình đánh
  giá (mặc định validation, `--evaluation-split test` cho đánh giá
  cuối trên test).

### Điều phối (`train_evaluate.py`)

- `main(argv=None)` chạy: nạp dữ liệu -> chia tập -> build 4 mô hình
  chính thức -> train -> predict -> đánh giá -> so sánh -> lưu metric,
  dự đoán, biểu đồ và `outputs/mlp_torch_parameters.pt`. Nhận tham số
  dòng lệnh `--data` (đường dẫn CSV), `--output-dir`,
  `--evaluation-split` (`val` mặc định, ghi
  `outputs/metrics_val.csv`, `outputs/predictions_val.csv`, biểu đồ
  hậu tố `_val`; `test` cho đánh giá cuối, ghi `outputs/metrics.csv`,
  `outputs/predictions.csv`, biểu đồ không hậu tố) và
  `--include-numpy-mlp` (thêm bản NumPy MLP ngoài bảng chính thức).
- `build_models(output_dir)` tạo 4 mô hình chính thức; PyTorch MLP
  lưu tham số vào `output_dir/mlp_torch_parameters.pt`.
- `train_and_evaluate(data, output_dir, evaluation_split="val")` fit MLP với `X_val`/`y_val`
  (nếu `prepare_data` không trả validation thì tách 15% từ train),
  3 mô hình còn lại fit trên train; vẽ `actual_vs_predicted_<slug>.png`
  cho từng mô hình, `training_history_<slug>.png` khi có loss history,
  và `comparison_{rmse,mae,r2}.png`.
- Không viết logic mô hình hay metric trực tiếp trong file này;
  chỉ gọi các module chuyên trách.

## Quyết định kiến trúc (tóm tắt)

- **Modular monolith, kiến trúc phân lớp đơn giản:** một đơn vị triển
  khai, dữ liệu dùng chung. Không microservices, không database, không
  API (xem `adr/001-kien-truc-modular-monolith.md`).
- **Giữ cấu trúc gốc có sẵn** (`data/`, `docs/`, `models/`, `notebooks/`,
  `src/`), chỉ sắp xếp lại bên trong `src/` và bổ sung thư mục thiếu.
- **Notebook chỉ demo:** logic dùng lại nằm trong `src/`, notebook gọi
  module đó để demo và thí nghiệm, không viết class riêng trong notebook.
- **MLP chính thức dùng PyTorch, 3 mô hình còn lại được cài đặt bằng
  thư viện NumPy** (xem `adr/008-mlp-pytorch.md` và
  `adr/007-mo-hinh-numpy.md`, thay thế ADR-004). Bản NumPy MLP giữ
  làm đối chiếu legacy, bản PyTorch viết tay chỉ phục vụ học backprop
  (xem `adr/009-mlp-pytorch-manual.md`).
