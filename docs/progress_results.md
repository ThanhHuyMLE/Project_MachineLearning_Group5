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

| Mô hình | MSE | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|
| Linear Regression (kiểm chứng pipeline) | 2,656187e10 | 162.978 | 97.867 | 0,8153 |

Số R2 0,697 của Architecture Setup Milestone đo trên pipeline cũ (18 đặc trưng, chia ngẫu
nhiên, `zipcode` dạng số) và đã lỗi thời, không so sánh với số trên.

## Stage 02 & Stage 03 Milestone - Cài đặt MLP NumPy & Thực nghiệm Đánh giá

- Cài đặt thủ công `MLPRegressorModel` bằng NumPy trong `src/house_price/models/mlp.py` (không dùng thư viện ML/DL):
  - Kiến trúc: `Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear)`, khởi tạo trọng số He initialization.
  - Huấn luyện: Mini-batch Adam (`learning_rate=0.001`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`, `epochs=100`, `batch_size=32`, `random_state=42`), hàm mất mát MSE, theo dõi `loss` và `val_loss` sau mỗi epoch và lưu tham số tại `outputs/mlp_parameters.json`.
- Chạy notebook `notebooks/practice_04_mlp_regression.ipynb` (14 bước) và ghi kết quả chuẩn trên pipeline 88 đặc trưng (18 đặc trưng số đã `StandardScaler` + 70 cột one-hot `zipcode`) ra `metrics.csv`.

## Kết quả thực nghiệm

### 1. Kết quả chính thức trên tập Test (pipeline chuẩn 88 đặc trưng — `metrics.csv`)

Đánh giá trên tập test gồm **3.241 dòng** (không trùng `id` với train):

| Mô hình | MSE | RMSE (USD) | MAE (USD) | R2 |
|---|---:|---:|---:|---:|
| MLP Regression | 23.679.224.804,72 | 153.880,55 | 90.391,93 | 0,8354 |
| Linear Regression | 26.546.613.099,42 | 162.931,31 | 97.840,49 | 0,8154 |
| Decision Tree Regression | CHƯA CHẠY TRÊN 88 ĐẶC TRƯNG | - | - | - |
| Random Forest Regression | CHƯA CHẠY TRÊN 88 ĐẶC TRƯNG | - | - | - |

### 2. Kết quả đối chiếu 4 mô hình trong Notebook (19 đặc trưng thô chưa chuẩn hóa — Cell 44)

Trong `practice_04_mlp_regression.ipynb` (Cell 30–44), cả 4 mô hình được chạy thử trên 19 cột đặc trưng trước bước `StandardScaler` và one-hot `zipcode` (`input_size=19`):

| Mô hình | MSE | RMSE (USD) | MAE (USD) | R2 |
|---|---:|---:|---:|---:|
| Random Forest Regression (`n_estimators=100`) | 17.230.588.283,39 | 131.265,34 | 72.558,23 | 0,8802 |
| Decision Tree Regression | 36.826.161.724,53 | 191.901,44 | 102.928,24 | 0,7440 |
| Linear Regression | 41.095.334.851,90 | 202.719,84 | 128.919,71 | 0,7143 |
| MLP Regression (`input_size=19`, chưa scale) | 57.180.536.969,47 | 239.124,52 | 158.949,50 | 0,6024 |

- Quá trình huấn luyện MLP (19 đặc trưng chưa scale, chạy 2 lần $\times$ 100 epochs trong Cell 32):
  - Sau 100 epochs đầu: `Train MSE = 55.995.295.165,03` | `Val MSE = 53.024.511.335,43`.
  - Sau 100 epochs tiếp theo: `Train MSE = 50.839.130.222,88` (`MAE = 151.809,05 USD`) | `Val MSE = 49.710.016.546,42` (`MAE = 149.305,61 USD`).

## Giới hạn đã biết

- `MLPRegressorModel` đã được cài đặt thủ công bằng NumPy trong `src/house_price/models/mlp.py` (không phụ thuộc TensorFlow).
- Các số liệu demo giá dự đoán trong DESIGN.md mục 6.2 (ví dụ 542.300 USD)
  là số minh họa, không phải kết quả đo được. Không trích dẫn chúng
  như kết quả thực nghiệm.
- `zipcode` đã chốt ở Data Pipeline Implementation Milestone: mã hóa one-hot 70 mã, không còn xử lý
  như số liên tục; xem `adr/006-feature-engineering-va-zipcode.md`.
- Sự chênh lệch giữa hai bảng kết quả thực nghiệm đến từ việc chuẩn hóa dữ liệu: khi thiếu `StandardScaler` và one-hot `zipcode` (19 cột thô), thang đo các cột lệch nhau lớn (`sqft_lot` lên tới `1.651.359` so với biến nhị phân `0/1`) làm gradient của MLP hội tụ kém (`R2 = 0,6024`, dự đoán mẫu nhà mới bị âm `-380,28 USD`). Khi đưa đủ 88 đặc trưng đã chuẩn hóa, `MLP Regression` cải thiện rõ rệt lên `R2 = 0,8354`.
