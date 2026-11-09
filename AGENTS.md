# Hướng dẫn cho AI Agent - Project_MachineLearning_Group5

Tài liệu này là quy tắc bắt buộc cho mọi AI agent làm việc trong repo này.
Đây là bản dịch và điều chỉnh sang tiếng Việt từ quy tắc của dự án trước
(CLASSIFYING-SPAM-EMAILS), áp dụng cho bài toán Hồi quy giá nhà King County.

Ngôn ngữ: tài liệu viết bằng tiếng Việt; tên biến, hàm, class, module,
lệnh và commit message giữ nguyên tiếng Anh theo quy tắc ở mục 1 và 5.

---

## 1. Quy tắc commit message (bắt buộc)

Mỗi commit message do AI agent đề xuất, soạn hoặc thực thi **PHẢI** tuân theo
quy tắc đầy đủ trong [docs/commit-convention.md](docs/commit-convention.md).

### Dạng bắt buộc

```text
TYPE(scope): summary
```

### Quy tắc bắt buộc

1. **`TYPE` PHẢI viết hoa:** các type cho phép là `FEAT`, `FIX`, `REFACT`,
   `DATA`, `PERF`, `TEST`, `DOCS`, `STYLE`, `CHORE`, `SEC`, `OPS`, `REMOVE`.
   Cấm dùng type chữ thường hay chữ hoa lẫn thường (`feat`, `Fix`, `docs`).
   Nếu người dùng viết type chữ thường, agent **PHẢI** chuẩn hóa về chữ hoa
   trước khi đề xuất commit.
2. **`scope` PHẢI viết thường.** Scope chỉ rõ vùng thay đổi. Scope gợi ý:
   - `model` - Các lớp mô hình (`mlp`, `linear_regression`, `decision_tree`,
     `random_forest`), kiến trúc, siêu tham số, lưu/nạp mô hình.
   - `preprocess` - Nạp dữ liệu, làm sạch, bù khuyết, chia tập,
     chuẩn hóa (StandardScaler), chống rò rỉ dữ liệu (Data Leakage).
   - `features` - Chọn đặc trưng, mã hóa, kỹ thuật đặc trưng (Feature Engineering).
   - `eval` - Chỉ số hồi quy (`mse`, `rmse`, `mae`, `r2`, `mape`),
     bảng so sánh mô hình, biểu đồ đánh giá.
   - `train` - Điều phối huấn luyện (`train_evaluate.py`), lịch sử huấn luyện,
     theo dõi validation.
   - `data` - Dữ liệu King County, hợp đồng dữ liệu, mẫu dữ liệu.
   - `artifacts` - Mô hình đã lưu (`.joblib`, `.keras`), file `metrics.json`,
     kết quả trong `outputs/`.
   - `notebooks` - Notebook thực hành và demo.
   - `tests` - Bộ kiểm tra (`tests/`), fixture, dữ liệu giả.
   - `deps` - Thư viện phụ thuộc, `requirements.txt`, `pyproject.toml`.
   - `docs` - Tài liệu dự án, README, hướng dẫn.
   - `tooling` - Linter (`ruff`), test runner (`pytest`), script hỗ trợ.
   - `ci` - CI workflow, kiểm tra tự động, GitHub Actions.
   - `infra` - Cấu hình Git, môi trường ảo, môi trường chạy local.
3. **`summary` viết bằng tiếng Anh, dùng thể mệnh lệnh** (`add`, `fix`, `update`,
   không dùng `added`, `fixes`, `updating`). Ngắn gọn, cụ thể, hướng hành động.
   **KHÔNG** dùng tóm tắt mơ hồ (`update code`, `fix bug`, `changes`, `refactor`).
   **KHÔNG** kết thúc summary bằng dấu chấm (`.`).
4. **Thay đổi phá vỡ (Breaking Change):** thêm `!` ngay sau scope
   (ví dụ `FEAT(preprocess)!: change split ratios`) hoặc thêm footer
   `BREAKING CHANGE:` ở cuối commit body.
5. **Kiểm tra trước khi đề xuất commit:**
   - [ ] TYPE viết hoa đúng danh sách cho phép?
   - [ ] Scope viết thường và khớp vùng thay đổi?
   - [ ] Summary tiếng Anh, thể mệnh lệnh, không có dấu chấm cuối?
   - [ ] Summary cụ thể, không mơ hồ?
   - [ ] Thay đổi phá vỡ đã được đánh dấu (nếu có thay đổi hợp đồng dữ liệu,
         API mô hình, hay định dạng file lưu)?

6. **Kiểm tra local:** agent có thể kiểm tra commit message trước khi đề xuất:
   ```bash
   .venv/bin/python scripts/validate_commits.py --message "FEAT(model): add MLP regressor wrapper"
   ```

---

## 2. Quy trình Git và bảo vệ nhánh main

### Đồng bộ an toàn

- Trước khi tạo branch mới, fetch `origin/main` mới nhất.
- Trước khi sửa repo, kiểm tra branch hiện tại và working tree bằng `git status`.
- Không bao giờ xóa, ghi đè hay reset thay đổi chưa commit.
- Không tự động đổi branch nếu việc đó có thể ảnh hưởng công việc hiện có.
- Không chạy `git reset --hard`, `git clean -fd` hay lệnh phá hủy
  nếu chưa có phê duyệt rõ ràng của người dùng.
- Không rebase branch chung nếu chưa thống nhất với các thành viên khác.

### Quy tắc làm việc nhóm

- Mỗi sinh viên làm trên một feature branch riêng.
- Thay đổi nộp qua Pull Request, review rồi mới merge vào `main`.
- Tên branch:
  - `feat/<ten-ngan>` (ví dụ `feat/mlp-architecture`)
  - `fix/<ten-ngan>` (ví dụ `fix/scaler-leakage`)
  - `docs/<ten-ngan>` (ví dụ `docs/mlp-explanation`)
  - `refact/<ten-ngan>` (ví dụ `refact/extract-evaluation`)
  - `chore/<ten-ngan>` (ví dụ `chore/practice-04-architecture`)
  - `data/<ten-ngan>` (ví dụ `data/add-sample-houses`)
- **Không commit/push trực tiếp lên `main`** cho công việc thường xuyên.
  Nhánh `main` phải luôn ổn định.
- **Tuyệt đối không force push** lên `main` (`git push --force`).
- Branch ngắn, nhỏ, nguyên tử để giảm xung đột giữa 4 sinh viên.

### Ủy quyền commit và push

- **KHÔNG** tự động chạy `git commit` hay `git push` nếu chưa có
  xác nhận rõ ràng của người dùng.
- Luôn trình commit message (theo quy tắc mục 1) và branch đích trước.

### Kiểm tra chất lượng

- Chạy test và lint liên quan sau khi sửa code.
- Chạy full test suite trước khi đề xuất PR nếu làm được.
- Phân biệt lỗi có sẵn từ trước với lỗi do thay đổi hiện tại gây ra.
- Không bao giờ nói test đã pass nếu chưa thực sự chạy; nếu không chạy được thì nói rõ lý do và mục đã bỏ qua.
- Không sửa code không liên quan chỉ để test pass.
- Không cài đặt thư viện hay tạo lại môi trường ảo nếu chưa được phê duyệt.

---

## 3. Kiểm soát phạm vi và thay đổi tối thiểu

- Chỉ sửa file liên quan trực tiếp đến nhiệm vụ được giao.
- Không refactor module không liên quan nếu chưa được phê duyệt.
- Giữ nguyên API công khai hiện có, trừ khi thay đổi phá vỡ được yêu cầu.
- Không thêm thư viện mới nếu không cần thiết.
- Không sửa cấu hình dự án, CI workflow hay Git hook nếu chưa được phê duyệt.
- Luôn tóm tắt file đã sửa và giải thích thay đổi quan trọng.
- Hỏi lại khi yêu cầu mơ hồ và giả định sai có thể gây làm lại lớn.

---

## 4. An toàn dữ liệu

1. **Phân tách thư mục rõ ràng:**

   ```text
   data/
   ├── kc_house_data_NaN/    # Dữ liệu King County đã có trong repo
   │   └── data_raw/         # Dữ liệu gốc, không sửa
   └── processed/            # Dữ liệu sinh ra khi chạy (gitignored)
   models/                   # Mô hình đã huấn luyện (gitignored, trừ .gitkeep)
   outputs/                  # Metric, dự đoán, biểu đồ (gitignored, trừ .gitkeep)
   ```

2. **Không commit dữ liệu nhạy cảm:** dữ liệu King County là dữ liệu công khai.
   Tuyệt đối không commit dữ liệu cá nhân, thông tin tài khoản hay khóa bí mật.
3. **Tuân theo `.gitignore`:** file trong `data/processed/*`, `outputs/*` và
   mô hình trong `models/*` (trừ `.gitkeep`) không bao giờ được add vào git.
4. **Hợp đồng dữ liệu:** file CSV phải có cột mục tiêu `price` và 18 cột đặc trưng
   như mô tả trong [docs/king-county-dataset.md](docs/king-county-dataset.md).
   Cột `id`, `date` và cột index `Unnamed: 0` phải loại khỏi tập huấn luyện.

---

## 5. Quy tắc Machine Learning và code

1. **Tính xác định (Determinism):** luôn đặt `random_state=42` khi chia dữ liệu
   và khi dùng estimator ngẫu nhiên, để kết quả tái lập được.
2. **Chống rò rỉ dữ liệu (Data Leakage):** chia train/validation/test TRƯỚC,
   chỉ fit imputer và scaler trên tập train, rồi transform các tập còn lại.
   Không dùng tập test để chọn mô hình hay chỉnh siêu tham số.
   Xem [docs/adr/003-chia-train-validation-test.md](docs/adr/003-chia-train-validation-test.md).
3. **Chỉ số hồi quy:** đây là bài toán Hồi quy. Dùng MSE, RMSE, MAE, R2, MAPE.
   Không dùng accuracy, precision, recall, F1 (chỉ số phân loại).
   MSE, RMSE, MAE càng nhỏ càng tốt; R2 càng lớn càng tốt và có thể âm.
4. **Mẫu class mô hình:** mỗi mô hình tuân theo 4 phương thức
   `__init__` -> `build_model` -> `fit` -> `predict`.
   Không gọi `build_model` trong `__init__`. `predict` trước `fit`
   phải báo lỗi rõ ràng. `build_model` lại sẽ reset trạng thái đã học.
   Chi tiết: [docs/project_architecture.md](docs/project_architecture.md).
5. **Lưu mô hình an toàn:** kiểm tra và tạo thư mục đích trước khi lưu artifact
   (`target.parent.mkdir(parents=True, exist_ok=True)`).
6. **Không ghi kết quả giả:** không ghi kết quả chưa chạy như kết quả đã kiểm chứng.
   Phần chưa thực hiện ghi rõ `TODO` hoặc `CHƯA THỰC HIỆN`.
7. **Tài liệu tiếng Việt:** mọi tài liệu hướng người đọc viết bằng tiếng Việt,
   thuật ngữ kỹ thuật giữ nguyên tiếng Anh (ví dụ Regression, Data Leakage,
   Overfitting). Tên biến, hàm, class, module viết bằng tiếng Anh.
   Mỗi file tài liệu cố gắng dưới 200 dòng; nếu vượt thì tách theo chủ đề và bổ sung liên kết trong README.

---

## 6. Lệnh phát triển thiết yếu

Luôn dùng môi trường ảo tại `.venv/`:

```bash
# Chạy bộ test
.venv/bin/pytest

# Lint và format
.venv/bin/ruff check .
.venv/bin/ruff format --check .

# Kiểm tra commit message theo quy tắc
.venv/bin/python scripts/validate_commits.py --message "FEAT(model): add MLP regressor wrapper"

# Chạy trọn pipeline (sau khi các TODO mô hình được điền xong)
.venv/bin/python -m house_price.train_evaluate
```
