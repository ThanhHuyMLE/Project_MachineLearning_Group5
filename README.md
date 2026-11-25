# Dự đoán giá nhà King County - Hồi quy MLP (Nhóm 5)

Bài tập thực hành 4: dự đoán giá nhà (Regression - Hồi quy) bằng
Multilayer Perceptron (MLP) và đối chiếu với 3 mô hình truyền thống:
Linear Regression, Decision Tree Regression, Random Forest Regression.
Bản MLP chính thức dùng PyTorch theo yêu cầu của cô giáo cho lab 4
(xem [docs/adr/008-mlp-pytorch.md](docs/adr/008-mlp-pytorch.md));
Linear, Decision Tree và Random Forest được cài đặt thủ công bằng
NumPy, không dùng class mô hình của scikit-learn, TensorFlow hay
Keras (xem [docs/adr/007-mo-hinh-numpy.md](docs/adr/007-mo-hinh-numpy.md)).
MLP còn có bản PyTorch viết tay mang tính giáo dục
(`src/house_price/models/mlp_torch_manual.py`, xem
[docs/adr/009-mlp-pytorch-manual.md](docs/adr/009-mlp-pytorch-manual.md)).

> Trạng thái: Stage 01 đến Stage 04 đã xong. Bảng benchmark chính
> thức của 4 mô hình trên pipeline chuẩn 88 đặc trưng (chạy ngày
> 19/11/2026 trên Ubuntu, tổng thời gian fit 143,58 giây trên test)
> xem ở mục 4 và [docs/progress_results.md](docs/progress_results.md).

## 1. Bài toán

- Đầu vào: 19 cột đầu vào sau kỹ thuật đặc trưng (18 số + mã vùng
  `zipcode`), mô tả một căn nhà ở King County, Washington, Mỹ; sau
  mã hóa one-hot thành 88 cột đưa vào mô hình.
- Đầu ra: giá bán dự đoán `price` (USD), là giá trị liên tục.
- Đây là bài toán **Hồi quy (Regression)**, không phải Phân loại
  (Classification).
- Dữ liệu: `data/kc_house_data_NaN/data_raw/kc_house_data_NaN.csv`,
  21.613 giao dịch từ tháng 5/2014 đến tháng 5/2015.
  Chi tiết: [docs/king-county-dataset.md](docs/king-county-dataset.md).

## 2. Dữ liệu

- Nguồn: King County House Sales (bang Washington, Hoa Kỳ, gồm Seattle).
- 21.613 dòng, 22 cột (18 đặc trưng gốc, `price`, `Unnamed: 0`, `id`, `date`).
- Giá `price`: nhỏ nhất 75.000 USD, trung vị 450.000 USD, trung bình
  540.088 USD, lớn nhất 7.700.000 USD; phân bố lệch phải mạnh
  (skew 4,02).
- Khuyết: `bedrooms` 13 dòng, `bathrooms` 10 dòng; bù bằng median của
  tập train.
- Trùng lặp: 176 `id` lặp (cùng căn nhà bán nhiều lần), không xóa dòng
  nào mà chia tập theo nhóm `id`.
- Kỹ thuật đặc trưng: `house_age`, `was_renovated`, `sqft_per_bedroom`;
  `zipcode` mã hóa one-hot thành 70 cột.
- Chia train/validation/test 70/15/15 theo nhóm `id`, seed 42:
  15.129 / 3.243 / 3.241 dòng, 0 `id` trùng nhau giữa các tập.

## 3. Các Stage

| Stage | Nội dung | File chính |
|---|---|---|
| Stage 01: Data Understanding & Preprocessing | Dữ liệu, EDA, kỹ thuật đặc trưng, chia tập, chuẩn hóa | `src/house_price/preprocessing.py`, `docs/king-county-dataset.md` |
| Stage 02: MLP Architecture & Implementation | Kiến trúc và cài đặt MLP bằng PyTorch | `src/house_price/models/mlp_torch.py`, `mlp_torch_manual.py`, `docs/algorithms/mlp.md` |
| Stage 03: Model Training & Evaluation | Huấn luyện, đánh giá, trực quan hóa | `src/house_price/train_evaluate.py`, `src/house_price/evaluation.py`, `src/house_price/visualization.py` |
| Stage 04: Model Benchmarking & Analysis | 3 mô hình đối chiếu, bảng so sánh 4 mô hình chính thức | `src/house_price/models/linear_regression.py`, `decision_tree.py`, `random_forest.py`, `src/house_price/model_comparison.py` |

Thứ tự làm để ít xung đột nhất: Stage 01 trước (hợp đồng dữ liệu),
sau đó Stage 02 và Stage 04 làm song song (mô hình độc lập),
Stage 03 cuối cùng (ghép pipeline và đánh giá).
Chi tiết: [docs/detail_plan.md](docs/detail_plan.md).

## 4. Kết quả chính thức (Stage 04)

Kết quả benchmark chính thức trên tập test (3.241 dòng, pipeline chuẩn 88 đặc trưng, cùng một lần chạy pipeline ngày 19/11/2026 trên Ubuntu với `--evaluation-split test`, tổng thời gian fit 143,58 giây):

| Mô hình | Thời gian fit | MSE | RMSE (USD) | MAE (USD) | R2 | MAPE |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest Regression | 106,12 giây | 19.584.927.856,05 | 139.946,16 | 77.253,52 | 0,8638 | 14,6857% |
| PyTorch MLP Regression | 35,40 giây | 23.956.240.614,18 | 154.778,04 | 89.838,73 | 0,8334 | 17,7992% |
| Linear Regression | 1,54 giây | 27.053.237.258,13 | 164.478,68 | 99.082,41 | 0,8119 | 20,2726% |
| Decision Tree Regression | 0,52 giây | 38.626.368.259,00 | 196.535,92 | 99.880,12 | 0,7314 | 18,2854% |

3 mô hình NumPy cũ (Random Forest, Linear, Decision Tree) tái lập
đúng số benchmark ngày 17/11/2026 (lệch 0). Thứ hạng không đổi so
với lịch sử 17/11: Random Forest tốt nhất, tiếp theo là PyTorch MLP,
Linear, rồi Decision Tree. Chạy mặc định trên validation
(`--evaluation-split val`) ghi `outputs/metrics_val.csv` (RF R2 0,8617,
PyTorch MLP 0,8287, Linear 0,8091, Decision Tree 0,7198).

Giới hạn test-peeking: tập test đã từng bị xem trong quá trình phát triển trước đây nên không còn nguyên vẹn thống kê; các số trên là đo thật của lần chạy này nhưng không được diễn giải như đánh giá độc lập chưa từng bị xem trước.

Ghi chú về Linear Regression: số R2 0,8154 của Stage 03 là bản sklearn OLS, chỉ giữ làm lịch sử trong [docs/progress_results.md](docs/progress_results.md); số chính thức trong bảng trên là bản được cài đặt bằng thư viện NumPy với gradient descent (R2 0,8119).

## 5. Cấu trúc thư mục

```text
data/
  kc_house_data_NaN/      # Dữ liệu King County (đã có sẵn trong repo)
  processed/              # Dữ liệu sinh ra khi chạy (gitignored)
docs/
  DESIGN.md               # Tài liệu thiết kế gốc của nhóm (giữ nguyên)
  description.md          # Mô tả bài toán và phạm vi
  detail_plan.md          # Kế hoạch từng bước, tiêu chí hoàn thành
  project_architecture.md # Kiến trúc, hợp đồng module, luồng dữ liệu
  king-county-dataset.md  # Mô tả dữ liệu đã kiểm chứng
  pseudocode.md           # Mã giả tổng hợp của 4 thuật toán
  progress_results.md     # Tiến trình và kết quả thật
  oral-examination.md     # Câu hỏi vấn đáp và dàn ý thuyết trình
  oral-questions-preprocessing.md  # Câu hỏi vấn đáp về dataset và tiền xử lý
  commit-convention.md    # Quy tắc commit message
  adr/                    # Architecture Decision Records
  algorithms/             # Giải thích từng thuật toán (mlp, linear, tree, forest)
notebooks/
  practice_04_mlp_regression.ipynb   # Notebook trình bày 6 task của đề bài
  Report.md               # Báo cáo tổng hợp
models/                   # Mô hình đã huấn luyện (.joblib/.json, gitignored)
outputs/                  # Bảng metric, dự đoán, biểu đồ (gitignored)
src/house_price/          # Code dùng chung của cả nhóm
  artifacts.py            # Lưu preprocessor và manifest suy luận
  preprocessing.py        # Nạp dữ liệu, chia tập, chống rò rỉ dữ liệu
  predict.py              # CLI dự đoán CSV mới bằng artifact đã lưu
  report.py               # Sinh báo cáo Markdown/HTML từ metrics
  evaluation.py           # MSE, RMSE, MAE, R2, MAPE
  visualization.py        # Biểu đồ actual vs predicted, loss, so sánh
  model_comparison.py     # Bảng đối chiếu 4 mô hình
  train_evaluate.py       # Điều phối trọn pipeline (entry point)
  models/                 # Lớp mô hình theo mẫu __init__/build_model/fit/predict
scripts/
  validate_commits.py     # Kiểm tra commit message theo quy tắc
  generate_eda_figures.py # Vẽ biểu đồ EDA vào outputs/eda/
tests/                    # Kiểm tra hợp đồng module (pytest)
```

Quy tắc quan trọng nhất: **chia dữ liệu trước, chỉ fit imputer/scaler
trên tập train** (xem [docs/adr/003-chia-train-validation-test.md](docs/adr/003-chia-train-validation-test.md)).
Không dùng tập test để chọn mô hình hay chỉnh siêu tham số.

## 6. Yêu cầu môi trường

- Python >= 3.10 (xem `pyproject.toml`).
- Các thư viện chính trong `requirements.txt`: numpy, pandas,
  scikit-learn (chỉ dùng cho tiền xử lý và metrics), matplotlib,
  seaborn, joblib và torch (CPU-only, xem mục 7).
- Không cần TensorFlow/Keras: Linear, Decision Tree và Random Forest
  đều được cài đặt bằng thư viện NumPy; MLP dùng PyTorch theo
  yêu cầu của cô giáo (xem [docs/adr/008-mlp-pytorch.md](docs/adr/008-mlp-pytorch.md)).

Môi trường đã kiểm chứng: Ubuntu, Python 3.14.4, torch 2.15.0+cpu
(môi trường ảo `venv-ml-linux`). CI chạy Python 3.13 (xem
`.github/workflows/ci.yml`). PyTorch luôn là bản CPU-only.

## 7. Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
pip install -e ".[dev,notebook]"
```

`requirements.txt` cài torch bản CPU-only (khoảng 200 MB) qua
extra-index `https://download.pytorch.org/whl/cpu`; không cần GPU,
không cần CUDA. Muốn dùng GPU (tùy chọn, không bắt buộc): cài lại
torch theo hướng dẫn chính thức tại
https://pytorch.org/get-started/locally/ với đúng bản CUDA của máy,
rồi chạy lại pipeline; code không hard-code thiết bị (checkpoint nạp
bằng `map_location="cpu"` nên chạy được trên máy không có GPU).
Môi trường notebook dùng `requirements-notebook.txt`, file này kéo
toàn bộ `requirements.txt` (gồm torch) rồi thêm jupyterlab/ipykernel.

Lệnh `pip install -e .` cài gói `house_price` ở chế độ editable để
chạy `python -m house_price.train_evaluate` từ thư mục gốc của repo.
Nhóm `[dev]` thêm pytest và ruff; nhóm `[notebook]` thêm jupyterlab
và ipykernel.

## 8. Chạy pipeline huấn luyện và đánh giá

```bash
PYTHONPATH=src python -m house_price.train_evaluate
```

Lệnh này chạy trọn pipeline: nạp dữ liệu, chia train/validation/test,
huấn luyện 4 mô hình chính thức, đánh giá mặc định trên validation
(`--evaluation-split val`) và in bảng so sánh MSE, RMSE, MAE, R2,
MAPE. Kết quả lưu vào `outputs/` (`metrics_val.csv`,
`predictions_val.csv`, biểu đồ `*_val.png`), tham số PyTorch MLP lưu
vào `outputs/mlp_torch_parameters.pt`, tham số Linear Regression lưu
vào `outputs/linear_regression_parameters.json`. Gói `house_price`
chưa cài trong môi trường ảo thì phải đặt `PYTHONPATH=src`; nếu đã
`pip install -e .` thì có thể bỏ `PYTHONPATH=src`.

Bản PyTorch viết tay
(`MLPTorchManualModel`) không nằm trong pipeline benchmark; chạy
riêng qua test `tests/test_mlp_torch_manual.py`.

Đánh giá cuối trên test (chỉ chạy một lần sau khi đã chốt cấu hình):

```bash
PYTHONPATH=src python -m house_price.train_evaluate --evaluation-split test
```

Lệnh này ghi `outputs/metrics.csv`, `predictions.csv` và biểu đồ
không hậu tố `_val`.

Tùy chọn đường dẫn dữ liệu và thư mục đầu ra:

```bash
PYTHONPATH=src python -m house_price.train_evaluate --data data/kc_house_data_NaN/data_raw/kc_house_data_NaN.csv --output-dir outputs --evaluation-split test
```

Lưu ý: huấn luyện rừng 100 cây CART được cài đặt bằng thư viện NumPy có thể mất nhiều phút;
xem thời gian chạy trong bảng benchmark của Stage 04.

### Dự đoán dữ liệu mới

Pipeline huấn luyện lưu thêm `preprocessor.joblib` và
`inference_manifest.json` để dữ liệu mới dùng đúng median, scaler và
one-hot encoder đã học từ train. Dự đoán một file CSV thô bằng MLP:

```bash
PYTHONPATH=src python -m house_price.predict \
  --input data/new_houses.csv \
  --artifact-dir outputs \
  --output outputs/predictions_new.csv
```

CSV đầu vào cần các cột đặc trưng gốc; không bắt buộc có `price`. File
kết quả giữ nguyên dữ liệu đầu vào và thêm `predicted_price`.

### Sinh báo cáo từ kết quả đã lưu

```bash
PYTHONPATH=src python -m house_price.report --metrics outputs/metrics.csv
```

Lệnh này tạo `outputs/report.md` và `outputs/report.html` mà không
huấn luyện lại mô hình.

## 9. Chạy kiểm thử

```bash
pytest
```

Pytest đã cấu hình sẵn trong `pyproject.toml` (`testpaths = ["tests"]`,
`pythonpath = ["src"]`) nên chạy từ thư mục gốc của repo, không cần
cài gói trước. Các test kiểm tra hợp đồng module, tiền xử lý, chỉ số
đánh giá, gradient của MLP PyTorch viết tay và tính đúng đắn của từng
mô hình.

Bộ test có 126 test, đều pass trong lần chạy ngày 25/11/2026 trên
`.venv` (Linux, Python 3.14). Con số này là tại thời điểm đó; nếu bộ
test có thêm test mới thì xem báo cáo verify mới nhất trong
[docs/progress_results.md](docs/progress_results.md).

## 10. Kiểm tra phong cách code

```bash
ruff check
ruff format --check
```

Ruff cấu hình trong `pyproject.toml` (độ dài dòng 100, target
Python 3.10). Sửa định dạng bằng `ruff format` trước khi commit.

## 11. Luồng dữ liệu

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

Hợp đồng đầy đủ của từng module xem
[docs/project_architecture.md](docs/project_architecture.md).

## 12. Mô hình

Mỗi class tuân theo cùng mẫu `__init__` -> `build_model` -> `fit` ->
`predict`. Riêng MLP cần truyền `input_size` khi build:

```python
model = MLPTorchRegressorModel()
model.build_model(input_size=X_train.shape[1])  # pipeline hiện tại: 88
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
y_pred = model.predict(X_test)
```

Vai trò của 4 nhóm mô hình trong dự án:

| Nhóm | Công nghệ | Cách tính gradient | Vai trò |
|---|---|---|---|
| Standard PyTorch MLP | PyTorch | Autograd (`loss.backward()`) | Workflow và benchmark chính thức của Lab 4 |
| Manual PyTorch MLP | PyTorch Tensor | Chain rule viết tay | Bằng chứng hiểu thuật toán và để kiểm chứng |
| Các mô hình hồi quy còn lại (Linear, Decision Tree, Random Forest) | NumPy | Thuật toán viết tay | Baseline đối chiếu chính thức |

Bản PyTorch viết tay là bằng chứng trung tâm cho thấy nhóm hiểu
backprop và Adam, không phải file phụ: nó viết tay toàn bộ forward,
backward theo chain rule và Adam bằng tensor torch
(`MLPTorchManualModel`). Ngược lại, bản PyTorch chính thức KHÔNG
phải "hoàn toàn viết tay": nó dùng autograd và `torch.optim.Adam`
của PyTorch theo yêu cầu của cô giáo cho lab 4. Nếu yêu cầu "viết
tay" của cô chưa rõ ràng, phân biệt này được ghi rõ ở đây thay vì
giả định: phần "viết tay" của dự án nằm ở bản manual và các mô hình
NumPy, còn bản chính thức dùng đúng công cụ PyTorch chuẩn.

| Mô hình | Cài đặt | Siêu tham số mặc định |
|---|---|---|
| PyTorch MLP Regression (chính thức) | PyTorch (`torch.nn`, DataLoader, MSELoss, Adam, autograd): kiến trúc 32-16-1, He init dạng chuẩn | lr 0.001, epochs 100, batch 32, random_state 42 |
| PyTorch MLP viết tay (giáo dục, không benchmark) | Tensor torch, forward/backward/Adam viết tay bằng chain rule, `requires_grad=False`, không autograd, không `torch.optim` | lr 0.001, epochs 100, batch 32, beta1 0.9, beta2 0.999, epsilon 1e-8 |
| Linear Regression | Được cài đặt bằng thư viện NumPy: batch gradient descent trên `X_aug = [1 \| X]` | lr 0.1, max_epochs 1000, tol 1e-6 (tương đối) |
| Decision Tree Regression | CART được cài đặt bằng thư viện NumPy: chia theo MSE có trọng số, ngưỡng là trung điểm giá trị riêng biệt | max_depth không giới hạn, min_samples_split 2, min_samples_leaf 1 |
| Random Forest Regression | Bagging trên cây CART, cả hai đều được cài đặt bằng thư viện NumPy, bootstrap qua SeedSequence | n_estimators 100, max_features mặc định ("sqrt") |

Hai bản MLP dùng cùng kiến trúc 32-16-1 và cùng cấu hình; khác nhau
ở cách tính gradient và cập nhật tham số (xem bảng phân biệt trong
[docs/algorithms/mlp.md](docs/algorithms/mlp.md)).

Giải thích chi tiết từng thuật toán (mục đích, toán, mã giả khớp
source, siêu tham số, metrics, ưu/nhược điểm):
[docs/algorithms/mlp.md](docs/algorithms/mlp.md),
[docs/algorithms/linear_regression.md](docs/algorithms/linear_regression.md),
[docs/algorithms/decision_tree.md](docs/algorithms/decision_tree.md),
[docs/algorithms/random_forest.md](docs/algorithms/random_forest.md).
Mã giả tổng hợp: [docs/pseudocode.md](docs/pseudocode.md).

## 13. Đánh giá

- Chỉ số: MSE, RMSE, MAE, R2, MAPE, tính bằng `regression_metrics`
  trong `src/house_price/evaluation.py` trên cùng tập test cho cả
  4 mô hình chính thức. MAPE trả `None` khi có giá trị thật bằng 0.
- So sánh: `compare_models` sắp bảng theo MSE tăng dần, các cột
  model, mse, rmse, mae, r2, mape.
- Không chọn mô hình hay chỉnh siêu tham số dựa trên điểm test.
- R2 = 0,8334 không có nghĩa là dự đoán đúng 83,34%; đó là chỉ số
  giải thích biến thiên theo định nghĩa R2.

## 14. Outputs

Sau khi chạy pipeline, thư mục `outputs/` chứa:

- `metrics.csv`: bảng MSE, RMSE, MAE, R2, MAPE của 4 mô hình chính
  thức trên test (chạy với `--evaluation-split test`).
- `metrics_val.csv`, `predictions_val.csv`: kết quả đánh giá trên
  validation khi chạy mặc định (`--evaluation-split val`); biểu đồ
  so sánh và actual-vs-predicted tương ứng có hậu tố `_val`.
- `predictions.csv`: giá thật và giá dự đoán của từng mô hình trên test.
- `mlp_torch_parameters.pt`: state_dict, cấu hình và loss history
  của MLP bản PyTorch chính thức.
- `pytorch_mlp_regression_parameters.json`: kiến trúc, weights, biases
  và lịch sử huấn luyện của MLP PyTorch.
- `decision_tree_regression_parameters.json`: cấu hình và toàn bộ node
  của Decision Tree.
- `random_forest_regression_parameters.json`: cấu hình và toàn bộ 100
  cây; trạng thái từng cây được nén trong JSON để giảm dung lượng.
- `preprocessor.joblib`: median, scaler và one-hot encoder đã fit trên train.
- `inference_manifest.json`: hợp đồng file và danh sách cột cho suy luận.
- `predictions_new.csv`: dự đoán cho CSV mới khi chạy module `predict`.
- `report.md`, `report.html`: báo cáo sinh từ bảng metrics đã lưu.
- `mlp_torch_manual_parameters.pt`: weights/biases và cấu hình của
  bản PyTorch viết tay (chỉ có khi chạy bản manual riêng; không do
  pipeline benchmark sinh ra).
- `linear_regression_parameters.json`: hệ số và hệ số chặn của Linear Regression.
- `actual_vs_predicted_<model>.png`, `training_history_<model>.png`,
  `comparison_{rmse,mae,r2}.png`: biểu đồ đánh giá và so sánh.
- `eda/`: 7 biểu đồ EDA mô tả toàn bộ dữ liệu.

## 15. Quy tắc làm việc

- Làm trên branch riêng, nộp qua Pull Request, không commit trực tiếp lên `main`.
- Commit message theo dạng `TYPE(scope): summary`, TYPE viết hoa.
  Xem [docs/commit-convention.md](docs/commit-convention.md) và [AGENTS.md](AGENTS.md).
- Tài liệu viết bằng tiếng Việt, thuật ngữ kỹ thuật giữ nguyên tiếng Anh.
  Tên biến, hàm, class, module viết bằng tiếng Anh.
- Không ghi kết quả chưa chạy như kết quả đã kiểm chứng.
- Linear, Decision Tree và Random Forest được cài đặt bằng thư viện NumPy;
  MLP chính thức dùng PyTorch theo yêu cầu của cô giáo; bản PyTorch
  viết tay chỉ phục vụ
  học backprop; sklearn chỉ dùng cho tiền xử lý và metrics (xem
  [docs/adr/007-mo-hinh-numpy.md](docs/adr/007-mo-hinh-numpy.md),
  [docs/adr/008-mlp-pytorch.md](docs/adr/008-mlp-pytorch.md) và
  [docs/adr/009-mlp-pytorch-manual.md](docs/adr/009-mlp-pytorch-manual.md)).

## 16. Tài liệu tham khảo

- Tài liệu thiết kế gốc: [docs/DESIGN.md](docs/DESIGN.md) (giữ nguyên
  vì lý do lịch sử; một số điểm trong đó đã được sửa bởi các ADR).
- Mô tả bài toán: [docs/description.md](docs/description.md).
- Kế hoạch chi tiết: [docs/detail_plan.md](docs/detail_plan.md).
- Kiến trúc và hợp đồng module: [docs/project_architecture.md](docs/project_architecture.md).
- Mô tả dữ liệu: [docs/king-county-dataset.md](docs/king-county-dataset.md).
- Tiến trình và kết quả: [docs/progress_results.md](docs/progress_results.md).
- Vấn đáp: [docs/oral-examination.md](docs/oral-examination.md),
  [docs/oral-questions-preprocessing.md](docs/oral-questions-preprocessing.md).
- Quyết định kiến trúc: [docs/adr/](docs/adr/).
