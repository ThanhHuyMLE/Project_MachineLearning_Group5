# Quá trình thực hiện và kết quả

File này ghi lại việc làm, cách kiểm tra và kết quả **thật** theo từng
công đoạn. Không ghi kết quả chưa chạy như kết quả đã kiểm chứng.

## Architecture Setup Milestone - Kiến trúc, tài liệu, khung code (ngày 09/11/2026)

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

## Data Pipeline Implementation Milestone - Stage 01: Data Understanding & Preprocessing (ngày 09/11/2026)

- Viết lại `src/house_price/preprocessing.py`: thêm `engineer_features`
  (`house_age`, `was_renovated`, `sqft_per_bedroom`), `zipcode` mã hóa
  one-hot, chia tập theo nhóm `id` bằng `GroupShuffleSplit`.
- Đính chính số liệu ở Architecture Setup Milestone: số id lặp đúng là 176 (175 id 2 lần,
  1 id 3 lần), 177 là số dòng thừa, không phải số id lặp.
- Hợp đồng dữ liệu đổi từ 18 đặc trưng thành 19 cột đầu vào
  (18 số + `zipcode`); đầu ra preprocessor là 88 cột (18 số + one-hot
  70 zipcode). Xem `docs/adr/005-chia-theo-nhom-id.md` và
  `docs/adr/006-feature-engineering-va-zipcode.md`.
- Chia theo nhóm id: train 15.129, validation 3.243, test 3.241 dòng,
  số id trùng nhau giữa các tập là 0.
- Môi trường kiểm chứng: Python 3.12.3, sklearn 1.9.1, pandas 3.1.1,
  numpy 2.5.4. Test 35/35 pass; `ruff check` và `ruff format` sạch.
- Không xóa dòng nào trong toàn bộ Data Pipeline Implementation Milestone.

### Kiểm chứng tương thích pipeline (không phải kết quả cuối cùng)

Linear Regression chạy trên pipeline mới chỉ để kiểm tra pipeline chạy
tròn đầu đến cuối, KHÔNG phải kết quả cuối cùng của bài tập:

| Mô hình                                 |         MSE |    RMSE |    MAE |     R2 |
| --------------------------------------- | ----------: | ------: | -----: | -----: |
| Linear Regression (kiểm chứng pipeline) | 2,656187e10 | 162.978 | 97.867 | 0,8153 |

Số R2 0,697 của Architecture Setup Milestone đo trên pipeline cũ (18 đặc trưng, chia ngẫu
nhiên, `zipcode` dạng số) và đã lỗi thời, không so sánh với số trên.

## Kết quả thực nghiệm

Dưới đây là kết quả benchmark của 4 mô hình trên cùng một tập test, cùng một bộ tiền xử lý dữ liệu:

| Mô hình                      |          MSE | RMSE (USD) |  MAE (USD) |     R2 |
| ---------------------------- | -----------: | ---------: | ---------: | -----: |
| **Random Forest Regression** | 1.713137e+10 | 130,886.86 |  72,453.31 | 0.8809 |
| **MLP Regression**           | 2.367922e+10 | 153,880.55 |  90,391.93 | 0.8354 |
| **Linear Regression**        | 2.654661e+10 | 162,931.31 |  97,840.48 | 0.8154 |
| **Decision Tree Regression** | 4.427875e+10 | 210,425.17 | 106,562.20 | 0.6921 |

**Phân tích và Nhận xét:**
Random Forest có hiệu năng tốt nhất với R² ≈ 0.8809, MSE ≈ 1.713 × 10¹⁰ và RMSE ≈ 130,887 USD. MLP xếp thứ hai, kế đến là Linear Regression. Decision Tree có kết quả thấp nhất trong bốn mô hình do bị overfitting.

Random Forest phù hợp nhất với dữ liệu King County vì quan hệ giữa giá nhà với diện tích, chất lượng và vị trí (zipcode) có tính phi tuyến tính cao, trong khi giá trải rộng từ 75.000 đến 7,7 triệu USD và phân bố lệch phải mạnh. Việc kết hợp nhiều cây (Ensemble Bagging) giúp mô hình nắm bắt các quan hệ và tương tác phức tạp, đồng thời giảm biến động (variance) triệt để so với một Decision Tree đơn lẻ. Linear Regression khó biểu diễn các quan hệ phi tuyến này, còn MLP tuy linh hoạt nhưng cần tối ưu siêu tham số nhiều hơn mới có thể vượt qua Random Forest trên dữ liệu dạng bảng (tabular data) này.

## Giới hạn đã biết

- TensorFlow chưa được cài đặt (môi trường Python 3.14).
  MLP mặc định dùng sklearn MLPRegressor; xem `adr/004-mlp-trien-khai.md`.
- Các số liệu demo giá dự đoán trong DESIGN.md mục 6.2 (ví dụ 542.300 USD)
  là số minh họa, không phải kết quả đo được. Không trích dẫn chúng
  như kết quả thực nghiệm.
- `zipcode` đã chốt ở Data Pipeline Implementation Milestone: mã hóa one-hot 70 mã, không còn xử lý
  như số liên tục; xem `adr/006-feature-engineering-va-zipcode.md`.
