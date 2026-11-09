# Quá trình thực hiện và kết quả

File này ghi lại việc làm, cách kiểm tra và kết quả **thật** theo từng
công đoạn. Không ghi kết quả chưa chạy như kết quả đã kiểm chứng.

## Phase 1 - Kiến trúc, tài liệu, khung code (ngày 09/11/2026)

- Clone repo `ThanhHuyMLE/Project_MachineLearning_Group5` về máy Ubuntu,
  nhánh `main` tại commit `267049b`.
- Kiểm chứng dữ liệu CSV: 21.613 dòng, 22 cột, `bedrooms` khuyết 13,
  `bathrooms` khuyết 10, giá 75.000 - 7.700.000 USD, 177 `id` trùng lặp.
- Phát hiện: 3 notebook (`.ipynb`) đều rỗng 0 byte, file `src/ML_scartch`
  rỗng 0 byte, chưa có README, requirements, `.gitignore`, test, AGENTS.md.
- Tạo branch `chore/practice-04-architecture` (chưa commit, chưa push).
- Thiết lập cấu trúc `src/house_price/`, tài liệu tiếng Việt, quy tắc commit
  kế thừa từ dự án CLASSIFYING-SPAM-EMAILS.
- Sửa điểm yếu của DESIGN.md gốc: đổi chia train/test 80/20 (dùng test làm
  validation) thành train/validation/test 70/15/15, fit scaler chỉ trên train.
  Xem `docs/adr/003-chia-train-validation-test.md`.

## Kết quả thực nghiệm

**CHƯA CÓ KẾT QUẢ.** Chưa huấn luyện mô hình nào trong Phase 1.
Bảng dưới đây chỉ là khung chờ, Sinh viên 3 điền số thật sau khi chạy.

| Mô hình | MSE | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|
| MLP Regression | CHƯA CHẠY | - | - | - |
| Linear Regression | CHƯA CHẠY | - | - | - |
| Decision Tree Regression | CHƯA CHẠY | - | - | - |
| Random Forest Regression | CHƯA CHẠY | - | - | - |

## Giới hạn đã biết

- TensorFlow chưa được cài đặt (môi trường Python 3.14).
  MLP mặc định dùng sklearn MLPRegressor; xem `adr/004-mlp-trien-khai.md`.
- Các số liệu demo giá dự đoán trong DESIGN.md mục 6.2 (ví dụ 542.300 USD)
  là số minh họa, không phải kết quả đo được. Không trích dẫn chúng
  như kết quả thực nghiệm.
- `zipcode` đang xử lý như số liên tục; Sinh viên 1 cần chốt cách xử lý.
