# Quá trình thực hiện và kết quả

File này ghi lại việc làm, cách kiểm tra và kết quả **thật** theo từng
công đoạn. Không ghi kết quả chưa chạy như kết quả đã kiểm chứng.
Chỗ nào chưa có số thật thì để lại ghi chú chờ điền số và không điền số ước lượng.

## Điều kiện chung của mọi kết quả trong file này

### Dataset

- King County House Sales, file
  `data/kc_house_data_NaN/data_raw/kc_house_data_NaN.csv`:
  21.613 giao dịch, 22 cột (18 đặc trưng gốc, `price`, `Unnamed: 0`,
  `id`, `date`), từ tháng 5/2014 đến tháng 5/2015.
- Sau kỹ thuật đặc trưng (`house_age`, `was_renovated`,
  `sqft_per_bedroom`): 19 cột đầu vào (18 số + `zipcode`).
- Sau tiền xử lý (chuẩn hóa số + one-hot `zipcode`): 88 cột đưa vào
  mô hình (18 số + 70 mã zipcode). Chi tiết:
  [king-county-dataset.md](king-county-dataset.md).

### Chia tập (split)

- Train/validation/test 70/15/15, chia theo nhóm `id` bằng
  `GroupShuffleSplit`, `random_state=42`: train 15.129 dòng,
  validation 3.243 dòng, test 3.241 dòng, 0 `id` trùng nhau giữa
  các tập.
- Hồi quy không có nhãn lớp nên không stratify.
- Quy tắc: chọn mô hình và siêu tham số chỉ trên validation; tập test
  chỉ dùng một lần cuối để đánh giá.

### Môi trường

- Môi trường kiểm chứng Stage 01: Python 3.12.3, sklearn 1.9.1,
  pandas 3.1.1, numpy 2.5.4. Test 35/35 pass ở thời điểm đó;
  `ruff check` và `ruff format` sạch.
- Môi trường chạy benchmark Stage 04:
  Laptop Key, Windows 10 (bản dựng 10.0.19045.6466), Python 3.13.3, môi trường ảo `venv-ml-win`; numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2, joblib 1.6.0.

### Seed

- Mọi seed mặc định là 42: chia tập (`RANDOM_STATE = 42` trong
  `preprocessing.py`), khởi tạo MLP, xáo trộn batch, bootstrap và lấy
  mẫu đặc trưng của cây/rừng. Chạy lại cùng seed cho cùng kết quả.

### Siêu tham số (benchmark chính, Stage 04)

| Mô hình | Siêu tham số dùng trong benchmark chính |
|---|---|
| MLP Regression | Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear); He init; Adam lr 0.001, beta1 0.9, beta2 0.999, epsilon 1e-8; epochs 100; batch size 32; random_state 42 |
| Linear Regression | Batch gradient descent viết tay; lr 0.1; max_epochs 1000; tol 1e-6 (tương đối) |
| Decision Tree Regression | max_depth=8, random_state=42; các tham số còn lại giữ mặc định (min_samples_split=2, min_samples_leaf=1) |
| Random Forest Regression | n_estimators=100; max_depth=None; max_features="sqrt" (mỗi lần chia lấy mẫu ngẫu nhiên sqrt(n_features) đặc trưng); min_samples_split=2; min_samples_leaf=1; random_state=42 |

### Căn cứ chọn siêu tham số (trên validation, không dùng test)

- Decision Tree `max_depth=8` đạt R2 0,7198 trên validation (fit 1,81 giây), tốt hơn không giới hạn độ sâu (R2 0,6926, fit 48,31 giây).
- Random Forest `max_features="sqrt"` với 10 cây đạt R2 0,8363 trên validation (fit 29,56 giây); `max_features=None` đạt R2 0,8720 nhưng fit 276,71 giây với 10 cây, ngoại suy 100 cây khoảng 46 phút nên loại dù R2 cao hơn.
- Linear Regression đạt R2 0,8091 trên validation.

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

## Stage 01: Data Understanding & Preprocessing (ngày 09/11/2026)

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
- Không xóa dòng nào trong toàn bộ Stage 01.

### Kiểm chứng tương thích pipeline (không phải kết quả cuối cùng)

Linear Regression chạy trên pipeline mới chỉ để kiểm tra pipeline chạy
tròn đầu đến cuối, KHÔNG phải kết quả cuối cùng của bài tập:

| Mô hình | MSE | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|
| Linear Regression (kiểm chứng pipeline) | 2,656187e10 | 162.978 | 97.867 | 0,8153 |

Số R2 0,697 của Architecture Setup Milestone đo trên pipeline cũ (18 đặc trưng, chia ngẫu
nhiên, `zipcode` dạng số) và đã lỗi thời, không so sánh với số trên.

## Stage 02 và Stage 03 - Cài đặt MLP NumPy và thực nghiệm đánh giá

- Cài đặt thủ công `MLPRegressorModel` bằng NumPy trong `src/house_price/models/mlp.py` (không dùng thư viện ML/DL):
  - Kiến trúc: `Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear)`, khởi tạo trọng số He initialization.
  - Huấn luyện: Mini-batch Adam (`learning_rate=0.001`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`, `epochs=100`, `batch_size=32`, `random_state=42`), hàm mất mát MSE, theo dõi `loss` và `val_loss` sau mỗi epoch và lưu tham số tại `outputs/mlp_parameters.json`.
- Chạy notebook `notebooks/practice_04_mlp_regression.ipynb` (14 bước) và ghi kết quả chuẩn trên pipeline 88 đặc trưng (18 đặc trưng số đã `StandardScaler` + 70 cột one-hot `zipcode`) ra `metrics.csv`.
- Gradient check của MLP: sai số tương đối tối đa khoảng 1,7e-08
  (< 1e-5), xem `tests/test_mlp_gradient.py`.

## Stage 04 - Model Benchmarking & Analysis

### Bảng benchmark chính (pipeline chuẩn 88 đặc trưng, tập test 3.241 dòng)

Chạy ngày 17/11/2026 bằng lệnh `PYTHONPATH=src python -m house_price.train_evaluate --output-dir outputs`, tổng thời gian 294,13 giây; train 15.129 dòng, validation 3.243 dòng, test 3.241 dòng, 88 đặc trưng, seed 42.
Bảng dưới đây là kết quả chính thức của Stage 04: cả 4 mô hình chạy
trên cùng pipeline 88 đặc trưng, cùng tập test, cùng bộ chỉ số
(MSE, RMSE, MAE, R2, MAPE) tính bằng `regression_metrics`. MAPE
không đo trong lần chạy này (không có cột MAPE trong
`outputs/metrics.csv`) nên ghi "không đo", không ước lượng.

| Mô hình | Thời gian fit | MSE | RMSE (USD) | MAE (USD) | R2 | MAPE |
|---|---:|---:|---:|---:|---:|---:|
| MLP Regression | 26,18 giây | 23.679.224.804,72 | 153.880,55 | 90.391,93 | 0,8354 | không đo |
| Linear Regression | 4,77 giây | 27.053.237.258,13 | 164.478,68 | 99.082,41 | 0,8119 | không đo |
| Decision Tree Regression | 2,15 giây | 38.626.368.259,00 | 196.535,92 | 99.880,12 | 0,7314 | không đo |
| Random Forest Regression | 251,47 giây | 19.584.927.856,05 | 139.946,16 | 77.253,52 | 0,8638 | không đo |

Giới hạn test-peeking: tập test đã từng bị xem trong quá trình phát triển trước đây nên không còn nguyên vẹn thống kê; các số trong bảng trên là đo thật của lần chạy ngày 17/11/2026 nhưng không được diễn giải như đánh giá độc lập chưa từng bị xem trước.

Đối chiếu với Stage 03: MLP được benchmark mới xác nhận lại đúng số cũ (R2 0,8354, cùng MSE/RMSE/MAE). Linear Regression ở Stage 03 là bản sklearn OLS (R2 0,8154, MSE 26.546.613.099,42); bản viết tay bằng gradient descent trong benchmark này đạt R2 0,8119, MSE 27.053.237.258,13. Số Stage 03 giữ nguyên trong mục lịch sử bên dưới, không dùng làm số benchmark chính.

### Kết quả Stage 03 (lịch sử, giữ nguyên để đối chiếu)

Đánh giá trên tập test gồm **3.241 dòng** (không trùng `id` với train):

| Mô hình | MSE | RMSE (USD) | MAE (USD) | R2 |
|---|---:|---:|---:|---:|
| MLP Regression | 23.679.224.804,72 | 153.880,55 | 90.391,93 | 0,8354 |
| Linear Regression | 26.546.613.099,42 | 162.931,31 | 97.840,49 | 0,8154 |

Nhận xét (Stage 03): trên tập dữ liệu đã chuẩn hóa đầy đủ 88 đặc
trưng, `MLP Regression` đạt **R2 = 0,8354**, giảm **9.050,76 USD** ở
chỉ số RMSE (-5,56%) và giảm **7.448,55 USD** ở chỉ số MAE (-7,61%)
so với `Linear Regression` (R2 = 0,8154). Các tầng ẩn ReLU đã học
được quan hệ phi tuyến giữa quy mô căn nhà, chất lượng xây dựng và
khu vực `zipcode`.

### Kết quả thăm dò cây quyết định (chưa phải benchmark chính)

- `max_depth=8`: fit 1,12 giây, R2 trên test 0,7314.
- Không giới hạn độ sâu: fit 32,29 giây, R2 trên test 0,6880
  (overfit: chậm hơn nhiều mà tệ hơn trên test).
- Hai số này chỉ để chọn `max_depth` trên validation; cấu hình cuối
  cùng dùng trong bảng benchmark chính đã ghi ở mục Siêu tham số
  (`max_depth=8`).

## Supplementary Experiment: Preprocessing Ablation Study

Thí nghiệm này KHÔNG thuộc bảng benchmark chính và KHÔNG được trộn
vào bảng chính. Nó đo ảnh hưởng của việc bỏ qua bước tiền xử lý:
cả 4 mô hình chạy trên 19 cột đặc trưng thô chưa chuẩn hóa
(`input_size=19`, chưa `StandardScaler`, chưa one-hot `zipcode`).
Số liệu dưới đây là lịch sử từ phiên bản notebook 19 đặc trưng
trước khi viết lại sang pipeline 88 đặc trưng, không còn tái lập
được từ notebook hiện tại.

| Mô hình | MSE | RMSE (USD) | MAE (USD) | R2 |
|---|---:|---:|---:|---:|
| Random Forest Regression (`n_estimators=100`) | 17.230.588.283,39 | 131.265,34 | 72.558,23 | 0,8802 |
| Decision Tree Regression | 36.826.161.724,53 | 191.901,44 | 102.928,24 | 0,7440 |
| Linear Regression | 41.095.334.851,90 | 202.719,84 | 128.919,71 | 0,7143 |
| MLP Regression (`input_size=19`, chưa scale) | 57.180.536.969,47 | 239.124,52 | 158.949,50 | 0,6024 |

Quá trình huấn luyện MLP (19 đặc trưng chưa scale, chạy 2 lần
x 100 epochs trong Cell 32 của phiên bản notebook cũ):

- Sau 100 epochs đầu: `Train MSE = 55.995.295.165,03` |
  `Val MSE = 53.024.511.335,43`.
- Sau 100 epochs tiếp theo: `Train MSE = 50.839.130.222,88`
  (`MAE = 151.809,05 USD`) |
  `Val MSE = 49.710.016.546,42` (`MAE = 149.305,61 USD`).

Nhận xét: khi huấn luyện trên 19 cột thô chưa qua `StandardScaler`,
sự chênh lệch thang đo lớn giữa các biến (`sqft_lot` lên tới
`1.651.359` trong khi các cờ nhị phân chỉ nhận `0/1`) khiến gradient
của MLP hội tụ kém (R2 = 0,6024, RMSE = `239.124,52 USD`). Khi áp
dụng `StandardScaler` và one-hot `zipcode`, R2 tăng từ `0,6024` lên
`0,8354`. `Random Forest` đạt kết quả tốt nhất trên dữ liệu dạng
bảng (R2 = 0,8802, MAE = `72.558,23 USD`) do các thuật toán cây
quyết định không bị ảnh hưởng bởi thang đo của đặc trưng.

### Giới hạn test-peeking của thí nghiệm bổ sung này

Thí nghiệm 19 đặc trưng chạy trong notebook, dùng cùng một tập test
để vừa quan sát kết quả vừa rút ra kết luận về tiền xử lý (ví dụ
"MLP cần chuẩn hóa"). Việc dùng test để ra quyết định như vậy là
test-peeking: điểm số trong bảng bổ sung không còn là phép đo độc
lập hoàn toàn, vì quyết định về pipeline đã chịu ảnh hưởng của test.
Vì vậy:

- Bảng bổ sung chỉ dùng để minh họa ảnh hưởng của tiền xử lý, không
  dùng để xếp hạng mô hình hay báo cáo kết quả cuối cùng.
- Kết luận chính thức của dự án chỉ dựa trên bảng benchmark chính
  (88 đặc trưng), trong đó mọi quyết định về pipeline đã chốt trước
  khi đo trên test.
- Đã xong (2026-11-17): các cell huấn luyện (Cell 30-44) đã dùng đồng nhất ma trận 88 đặc trưng `X_train_prep`, `X_val_prep`, `X_test_prep` cho cả 4 mô hình.

## Giới hạn đã biết

- `MLPRegressorModel` đã được cài đặt thủ công bằng NumPy trong `src/house_price/models/mlp.py` (không phụ thuộc TensorFlow).
- Các số liệu demo giá dự đoán trong DESIGN.md mục 6.2 (ví dụ 542.300 USD)
  là số minh họa, không phải kết quả đo được. Không trích dẫn chúng
  như kết quả thực nghiệm.
- `zipcode` đã chốt ở Stage 01: mã hóa one-hot 70 mã, không còn xử lý
  như số liên tục; xem `adr/006-feature-engineering-va-zipcode.md`.
- Sự chênh lệch giữa bảng benchmark chính và bảng bổ sung đến từ việc
  chuẩn hóa dữ liệu: khi thiếu `StandardScaler` và one-hot `zipcode`
  (19 cột thô), thang đo các cột lệch nhau lớn (`sqft_lot` lên tới
  `1.651.359` so với biến nhị phân `0/1`) làm gradient của MLP hội tụ
  kém (R2 = 0,6024, dự đoán mẫu nhà mới bị âm `-380,28 USD`). Khi đưa
  đủ 88 đặc trưng đã chuẩn hóa, `MLP Regression` cải thiện rõ rệt lên
  R2 = 0,8354.
