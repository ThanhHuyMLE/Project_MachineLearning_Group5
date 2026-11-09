# Quy tắc commit message

Dự án này tuân theo quy tắc commit message có cấu trúc để lịch sử git dễ đọc,
dễ tìm kiếm và để cả con người lẫn AI agent đều theo dõi được.
Quy tắc này được kế thừa từ dự án trước CLASSIFYING-SPAM-EMAILS, chỉ đổi
scope và ví dụ sang bài toán Hồi quy giá nhà.

## Dạng chung

Mỗi commit message phải dùng dạng:

```text
TYPE(scope): summary
```

`TYPE` là bắt buộc và PHẢI viết hoa.

### Ví dụ đúng / sai

**Đúng:**
- `FEAT(model): add MLP regressor wrapper`
- `FIX(preprocess): fit scaler on training data only`
- `DOCS(readme): add evaluation instructions and metrics guide`

**Sai:**
- `feat(model): add MLP regressor wrapper`
- `Fix(preprocess): fit scaler on training data only`
- `docs(readme): add evaluation instructions and metrics guide`

### Quy tắc chung

1. **TYPE** là bắt buộc và PHẢI viết hoa. Cấm dùng chữ thường hay chữ hoa
   lẫn thường (ví dụ `feat`, `Fix`).
2. **scope** phải viết thường.
3. **summary** viết bằng tiếng Anh.
4. **summary** ngắn gọn, dùng thể mệnh lệnh (`add`, `fix`, `change`,
   không dùng `added`, `fixes`, `changing`).
5. Không kết thúc summary bằng dấu chấm.
6. Ưu tiên một thay đổi logic trong một commit.

---

## Các TYPE cho phép

- **FEAT**: Tính năng mới (mô hình, pipeline, CLI)
  - Ví dụ: `FEAT(model): add MLP regressor wrapper`
- **FIX**: Sửa lỗi (tiền xử lý, huấn luyện, CLI, metric)
  - Ví dụ: `FIX(preprocess): handle missing bedrooms values with median`
- **REFACT**: Refactor code mà không đổi hành vi hay hợp đồng mô hình
  - Ví dụ: `REFACT(eval): extract metric computation into helpers`
- **DATA**: Thay đổi dữ liệu mẫu, hợp đồng dữ liệu, kiểm tra dữ liệu
  - Ví dụ: `DATA(data): document King County dataset contract`
- **PERF**: Tối ưu hiệu năng hoặc bộ nhớ
  - Ví dụ: `PERF(train): cache scaled features between model runs`
- **TEST**: Thêm, sửa hoặc cập nhật test và fixture
  - Ví dụ: `TEST(preprocess): add test case for leakage-free split`
- **DOCS**: Chỉ thay đổi tài liệu (README, docstring, hướng dẫn)
  - Ví dụ: `DOCS(readme): document training workflow and metrics report`
- **STYLE**: Format code, khoảng trắng, điều chỉnh theo ruff
  - Ví dụ: `STYLE(train): format argument parser definitions`
- **CHORE**: Công cụ, thư viện phụ thuộc, packaging, .gitignore, môi trường
  - Ví dụ: `CHORE(deps): add pytest and ruff to dev dependencies`
- **SEC**: Cải thiện bảo mật (làm sạch input, nạp mô hình an toàn)
  - Ví dụ: `SEC(data): prevent committing sensitive sample data`
- **OPS**: CI/CD workflow, script tự động, cài đặt môi trường
  - Ví dụ: `OPS(ci): add automated test and lint workflow`
- **REMOVE**: Xóa mô hình, tiện ích, dữ liệu hay code không dùng nữa
  - Ví dụ: `REMOVE(model): drop deprecated scratch prototype file`

*Chú ý: không cho phép TYPE chữ thường hay chữ hoa lẫn thường.*

---

## Scope gợi ý

Dùng scope cụ thể nhất mô tả vùng thay đổi chính. Nếu thay đổi lan nhiều vùng,
lấy scope của vùng chính.

- `model` - Lớp mô hình (`mlp`, `linear_regression`, `decision_tree`,
  `random_forest`), kiến trúc, siêu tham số
- `preprocess` - Nạp dữ liệu, làm sạch, bù khuyết, chia tập, chuẩn hóa
- `features` - Chọn đặc trưng, mã hóa, kỹ thuật đặc trưng
- `eval` - Tính MSE, RMSE, MAE, R2, MAPE, bảng so sánh, biểu đồ
- `train` - Điều phối huấn luyện, lịch sử huấn luyện, theo dõi validation
- `cli` - Điểm vào dòng lệnh, phân tích tham số, in kết quả
- `data` - Dữ liệu King County, hợp đồng dữ liệu, mẫu dữ liệu
- `artifacts` - Lưu/nạp mô hình (`.joblib`, `.keras`), file `metrics.json`
- `notebooks` - Notebook thực hành và demo
- `tests` - Unit test, integration test, dữ liệu giả, fixture
- `deps` - Quản lý thư viện, `requirements.txt`, `pyproject.toml`
- `docs` - Tài liệu dự án, README, hướng dẫn môn học
- `infra` - Cấu hình Git, môi trường ảo, môi trường chạy local
- `ci` - Tích hợp liên tục, chạy test tự động, GitHub Actions
- `tooling` - Linter (`ruff`), test runner (`pytest`), script hỗ trợ

### Ví dụ theo scope

- `FEAT(model): add random forest regressor wrapper`
- `FIX(preprocess): impute bathrooms with training median only`
- `FEAT(eval): report RMSE alongside MSE and MAE`
- `TEST(eval): verify R2 computation on known values`
- `DOCS(readme): update quick start instructions`

---

## Kiểu viết summary

Summary phải cụ thể và hướng hành động. Tránh mô tả mơ hồ.

- **Tốt:**
  - `FIX(preprocess): handle missing text fields gracefully`
  - `FEAT(model): add MLP with hidden layers 32 and 16`
  - `DOCS(readme): add local development setup instructions`
- **Tệ:**
  - `FIX(model): fix bug`
  - `CHORE: update`
  - `FEAT: changes`
  - `REFACT(preprocess): improve stuff`

---

## Commit body

Với commit lớn hoặc phức tạp, dùng thêm body để giải thích chi tiết.

```text
FEAT(model): add MLP regressor wrapper

- wrap sklearn MLPRegressor with build_model/fit/predict interface
- add unit test checking predict before fit raises an error
- update README with MLP usage example
```

- Ngắt dòng giữa subject và body bằng một dòng trống.
- Dùng bullet point cho thay đổi nhiều phần.
- Giải thích *vì sao* thay đổi nếu lý do không hiển nhiên.
- Giữ body ngắn gọn.

---

## Thay đổi phá vỡ (Breaking Change)

Để đánh dấu thay đổi phá vỡ, thêm `!` sau scope hoặc thêm footer
`BREAKING CHANGE:` ở cuối commit body.

```text
FEAT(preprocess)!: change split ratios to train/validation/test
```

```text
FEAT(data): enforce strict King County column schema

BREAKING CHANGE: Datasets without a 'price' column are now rejected.
```

Dùng khi: thay đổi hợp đồng dữ liệu, đổi API mô hình, đổi định dạng file
mô hình đã lưu, đổi tên hoặc xóa tham số CLI.

---

## Thay đổi nhiều vùng

Để lịch sử git sạch, ưu tiên tách thành nhiều commit:

1. `DATA(data): document King County feature contract`
2. `FEAT(preprocess): drop id and date columns before split`
3. `TEST(preprocess): add test case for dropped columns`
4. `DOCS(readme): update dataset section`

*Chú ý: nếu thay đổi nhỏ và gắn kết chặt chẽ, một commit duy nhất
dùng scope chính là chấp nhận được.*

---

## Kiểm tra local

```bash
.venv/bin/python scripts/validate_commits.py --message "FEAT(model): add MLP regressor wrapper"
```

## Ghi chú về CI

Dự án trước có workflow GitHub Actions kiểm tra commit message trên mỗi PR.
**Phase 1 chưa thêm workflow CI** (nằm ngoài phạm vi kiến trúc),
chỉ giữ script kiểm tra local `scripts/validate_commits.py`.
Khi nhóm quyết định bật CI, thêm `.github/workflows/commit-validation.yml`
theo mẫu của dự án trước và cập nhật mục này.
