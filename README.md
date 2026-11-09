# Dự đoán giá nhà King County - Hồi quy MLP (Nhóm 5)

Bài tập thực hành 4: dự đoán giá nhà (Regression - Hồi quy) bằng
Multilayer Perceptron (MLP) và đối chiếu với 3 mô hình truyền thống:
Linear Regression, Decision Tree Regression, Random Forest Regression.

> Trạng thái: **Phase 1 - Kiến trúc, tài liệu và khung code (scaffolding)**.
> Chưa huấn luyện mô hình, chưa có kết quả thực nghiệm.
> Xem [docs/progress_results.md](docs/progress_results.md).

## Bài toán

- Đầu vào: 19 cột đầu vào sau kỹ thuật đặc trưng (18 số + mã vùng `zipcode`),
  mô tả một căn nhà ở King County, Washington, Mỹ; sau mã hóa one-hot thành
  88 cột đưa vào mô hình.
- Đầu ra: giá bán dự đoán `price` (USD), là giá trị liên tục.
- Đây là bài toán **Hồi quy (Regression)**, không phải Phân loại (Classification).
- Dữ liệu: `data/kc_house_data_NaN/data_raw/kc_house_data_NaN.csv`,
  21.613 giao dịch từ tháng 5/2014 đến tháng 5/2015.
  Chi tiết: [docs/king-county-dataset.md](docs/king-county-dataset.md).

## Cài đặt nhanh

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev,notebook]"
```

## Chạy thử pipeline (sau khi Sinh viên 2 và 4 điền xong TODO)

```bash
python -m house_price.train_evaluate
```

Lệnh này chạy trọn pipeline: nạp dữ liệu, chia train/validation/test,
huấn luyện 4 mô hình, đánh giá và in bảng so sánh MSE, RMSE, MAE, R2.
Kết quả lưu vào `outputs/`, mô hình đã huấn luyện lưu vào `models/`.

## Cấu trúc thư mục

```text
data/
  kc_house_data_NaN/      # Dữ liệu King County (đã có sẵn trong repo)
  processed/              # Dữ liệu sinh ra khi chạy (gitignored)
docs/
  DESIGN.md               # Tài liệu thiết kế gốc của nhóm (giữ nguyên)
  description.md           # Mô tả bài toán và phạm vi
  detail_plan.md     # Kế hoạch từng bước, tiêu chí hoàn thành
  project_architecture.md       # Kiến trúc, hợp đồng module, luồng dữ liệu
  king-county-dataset.md   # Mô tả dữ liệu đã kiểm chứng
  pseudocode.md               # Mã giả tổng hợp của 4 thuật toán
  progress_results.md    # Tiến trình và kết quả thật (chưa có kết quả)
  commit-convention.md     # Quy tắc commit message
  adr/                    # Architecture Decision Records
  algorithms/             # Giải thích từng thuật toán (mlp, linear, tree, forest)
notebooks/
  practice_04_mlp_regression.ipynb   # Notebook demo 14 bước của đề bài
  Report.md               # Báo cáo (TODO: Sinh viên tổng hợp cuối kỳ)
models/                   # Mô hình đã huấn luyện (.joblib/.keras, gitignored)
outputs/                  # Bảng metric, dự đoán, biểu đồ (gitignored)
src/house_price/          # Code dùng chung của cả nhóm
  preprocessing.py        # Nạp dữ liệu, chia tập, chống rò rỉ dữ liệu
  evaluation.py           # MSE, RMSE, MAE, R2, MAPE
  visualization.py        # Biểu đồ actual vs predicted, loss, so sánh
  model_comparison.py     # Bảng đối chiếu 4 mô hình
  train_evaluate.py       # Điều phối trọn pipeline (entry point)
  models/                 # Lớp mô hình theo mẫu __init__/build_model/fit/predict
scripts/
  validate_commits.py     # Kiểm tra commit message theo quy tắc
tests/                    # Kiểm tra hợp đồng module (pytest)
```

Quy tắc quan trọng nhất: **chia dữ liệu trước, chỉ fit imputer/scaler
trên tập train** (xem [docs/adr/003-chia-train-validation-test.md](docs/adr/003-chia-train-validation-test.md)).
Không dùng tập test để chọn mô hình hay chỉnh siêu tham số.

## Phân công 4 sinh viên

| Sinh viên | Phụ trách chính | File chính |
|---|---|---|
| 1 | Dữ liệu và Tiền xử lý (Preprocessing) | `src/house_price/preprocessing.py`, `docs/king-county-dataset.md` |
| 2 | Kiến trúc MLP và lý thuyết | `src/house_price/models/mlp.py`, `docs/algorithms/mlp.md` |
| 3 | Huấn luyện và Đánh giá (Training & Evaluation) | `src/house_price/train_evaluate.py`, `src/house_price/evaluation.py`, `src/house_price/visualization.py` |
| 4 | Mô hình đối chiếu | `src/house_price/models/linear_regression.py`, `decision_tree.py`, `random_forest.py`, `src/house_price/model_comparison.py` |

Thứ tự làm để ít xung đột nhất: Sinh viên 1 trước (hợp đồng dữ liệu),
sau đó Sinh viên 2 và 4 làm song song (mô hình độc lập),
Sinh viên 3 cuối cùng (ghép pipeline và đánh giá).
Chi tiết: [docs/detail_plan.md](docs/detail_plan.md).

## Quy tắc làm việc

- Mỗi sinh viên làm trên branch riêng, nộp qua Pull Request, không commit trực tiếp lên `main`.
- Commit message theo dạng `TYPE(scope): summary`, TYPE viết hoa.
  Xem [docs/commit-convention.md](docs/commit-convention.md) và [AGENTS.md](AGENTS.md).
- Tài liệu viết bằng tiếng Việt, thuật ngữ kỹ thuật giữ nguyên tiếng Anh.
  Tên biến, hàm, class, module viết bằng tiếng Anh.
- Không ghi kết quả chưa chạy như kết quả đã kiểm chứng.

## Tài liệu tham khảo

- Tài liệu thiết kế gốc: [docs/DESIGN.md](docs/DESIGN.md).
- Giải thích thuật toán: [docs/algorithms/mlp.md](docs/algorithms/mlp.md),
  [docs/algorithms/linear_regression.md](docs/algorithms/linear_regression.md),
  [docs/algorithms/decision_tree.md](docs/algorithms/decision_tree.md),
  [docs/algorithms/random_forest.md](docs/algorithms/random_forest.md).
