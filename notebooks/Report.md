# Báo cáo Bài tập thực hành 4: Hồi quy MLP Dự đoán Giá nhà King County

## 1. Mục tiêu & Tổng quan dữ liệu

Bài tập xây dựng và đánh giá mô hình hồi quy Multi-Layer Perceptron bằng PyTorch để dự đoán giá bán nhà (`price`, đơn vị USD) tại quận King County, bang Washington, Hoa Kỳ.

- **Bộ dữ liệu:** `data/kc_house_data_NaN/data_raw/kc_house_data_NaN.csv` gồm **21.613 giao dịch** (từ `20140502` đến `20150527`) và **22 cột** ban đầu (18 đặc trưng gốc, `price`, `Unnamed: 0`, `id`, `date`).
- **Phân bố giá bán (`price`):**
  - Min: **75.000,00 USD** | Q1: **321.950,00 USD** | Trung vị: **450.000,00 USD** | Trung bình: **540.088,14 USD** | Q3: **645.000,00 USD** | Max: **7.700.000,00 USD**.
  - Phân bố lệch phải mạnh (`skewness = 4,02`, `kurtosis = 34,59`), với P90 = **887.000 USD**, P95 = **1.156.480 USD**, P99 = **1.964.400 USD**. Do đuôi phải dài chịu ảnh hưởng của nhà cao cấp, chỉ số RMSE cần được đọc song song với MAE và $R^2$.
- **Đặc trưng nổi bật:**
  - Tương quan Pearson mạnh nhất với `price`: `sqft_living` (**0,702**), `grade` (**0,667**), `sqft_above` (**0,606**), `sqft_living15` (**0,585**), `sqft_per_bedroom` (**0,580**), `bathrooms` (**0,526**).
  - Nhà giáp mặt nước (`waterfront = 1`, 163 căn - 0,75%) có giá trung bình **1.661.876 USD**, cao gấp **3,13 lần** nhà không giáp nước (**531.564 USD**).

## 2. Tiền xử lý & Kỹ thuật đặc trưng (Stage 01)

Toàn bộ pipeline trong `src/house_price/preprocessing.py` chỉ `fit` trên tập `train` để ngăn rò rỉ dữ liệu:

1. **Làm sạch & Giữ tính toàn vẹn phân bố:**
   - **Giá trị khuyết:** Bù khuyết bằng trung vị (`median`) của tập `train` cho `bedrooms` (khuyết 13 giá trị - 0,06%), `bathrooms` (khuyết 10 giá trị - 0,05%) và `sqft_per_bedroom` (khuyết 13 giá trị).
   - **Bản ghi trùng `id`:** Có 0 dòng trùng hoàn toàn; 176 `id` xuất hiện lặp lại (tổng 353 dòng, cách nhau tối đa 315 ngày). Giữ nguyên toàn bộ 21.613 dòng vì đây là các giao dịch bán lại hợp lệ ở thời điểm khác nhau.
   - **Ngoại lai:** Giữ nguyên 1.146 dòng (5,3%) vượt ngưỡng trên IQR (`1.129.575 USD`) để bảo toàn phân khúc bất động sản cao cấp.
2. **Feature Engineering (19 cột đầu vào):**
   - `house_age`: Tuổi nhà (`năm bán - yr_built`), kẹp tối thiểu về 0 cho 12 dòng có `yr_built > năm bán` (miền `0-115`, trung vị `40`).
   - `was_renovated`: Cờ nhị phân nhà đã từng cải tạo (`908` dòng - `4,20%`).
   - `sqft_per_bedroom`: Diện tích sống trên mỗi phòng ngủ (`49,09-3.420,00`, trung vị `576,67`).
3. **Chia tập theo nhóm `id` (`GroupShuffleSplit`, tỷ lệ 70/15/15, `random_state=42`):**
   - `train`: **15.129 dòng** (`y_train` trung bình **537.298 USD**).
   - `val`: **3.243 dòng** (`y_val` trung bình **547.304 USD**).
   - `test`: **3.241 dòng** (`y_test` trung bình **545.893 USD**).
   - Loại bỏ hoàn toàn rò rỉ `id` giữa các tập (`0` căn nhà trùng giữa `train` với `val`/`test`).
4. **Chuẩn hóa & Mã hóa:**
   - 18 đặc trưng số chuẩn hóa bằng `StandardScaler` + 70 mã `zipcode` mã hóa `OneHotEncoder(handle_unknown="ignore")` $\rightarrow$ Đầu vào chuẩn gồm **88 đặc trưng**.

## 3. Kiến trúc Mô hình MLP (`src/house_price/models/mlp_torch.py`)

- **Cấu trúc mạng:** `Input -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear)` với khởi tạo trọng số **He initialization**.
- **Tối ưu hóa:** PyTorch autograd và **Adam** (`learning_rate = 0.001`, `batch_size = 32`, `epochs = 100`, `random_state = 42`).

## 4. Kết quả Thực nghiệm & Phân tích

### 4.1. Đánh giá trên pipeline chuẩn 88 đặc trưng (`metrics.csv`)

| Mô hình | MSE | RMSE (USD) | MAE (USD) | R2 | MAPE |
|---|---:|---:|---:|---:|---:|
| PyTorch MLP Regression | 23.956.240.614,18 | 154.778,04 | 89.838,73 | 0,8334 | 17,7992% |
| Linear Regression | 27.053.237.258,13 | 164.478,68 | 99.082,41 | 0,8119 | 20,2726% |
| Decision Tree Regression | 38.626.368.259,00 | 196.535,92 | 99.880,12 | 0,7314 | 18,2854% |
| **Random Forest Regression** | **19.584.927.856,05** | **139.946,16** | **77.253,52** | **0,8638** | **14,6857%** |

Thời gian fit trong lần chạy chính thức ngày 19/11/2026: Random Forest 106,12 giây; PyTorch MLP 35,40 giây; Linear Regression 1,54 giây; Decision Tree 0,52 giây.

Giới hạn test-peeking: tập test đã từng bị xem trong quá trình phát triển trước đây nên không còn nguyên vẹn thống kê; các số trong bảng 4.1 là đo thật của lần chạy này nhưng không được diễn giải như đánh giá độc lập chưa từng bị xem trước.

- **Nhận xét:** Trên tập dữ liệu đã chuẩn hóa đầy đủ 88 đặc trưng, `Random Forest Regression` đạt kết quả tốt nhất (**$R^2 = 0,8638$**). `PyTorch MLP Regression` đứng thứ hai với **$R^2 = 0,8334$**, vượt `Linear Regression` ($R^2 = 0,8119$), cho thấy các tầng ẩn ReLU học được quan hệ phi tuyến trong dữ liệu.

### 4.2. Thí nghiệm bổ sung: đối chiếu 4 mô hình trên 19 đặc trưng thô chưa chuẩn hóa

> Bảng này là thí nghiệm bổ sung (preprocessing ablation), KHÔNG so sánh trực tiếp với bảng 4.1 vì khác pipeline (19 cột thô so với 88 cột đã chuẩn hóa). Số liệu trong bảng là lịch sử từ phiên bản notebook 19 đặc trưng trước khi viết lại sang pipeline 88 đặc trưng, không còn tái lập được từ notebook hiện tại. Thí nghiệm dùng cùng tập test để rút ra kết luận về tiền xử lý nên có giới hạn test-peeking; xem chi tiết trong `docs/progress_results.md`.

| Mô hình | MSE | RMSE (USD) | MAE (USD) | R2 |
|---|---:|---:|---:|---:|
| **Random Forest Regression** | **17.230.588.283,39** | **131.265,34** | **72.558,23** | **0,8802** |
| Decision Tree Regression | 36.826.161.724,53 | 191.901,44 | 102.928,24 | 0,7440 |
| Linear Regression | 41.095.334.851,90 | 202.719,84 | 128.919,71 | 0,7143 |
| MLP Regression (`input_size=19`, chưa scale) | 57.180.536.969,47 | 239.124,52 | 158.949,50 | 0,6024 |

- **Nhận xét:**
  - Khi huấn luyện trên 19 cột thô chưa qua `StandardScaler`, sự chênh lệch thang đo lớn giữa các biến (`sqft_lot` lên tới `1.651.359` trong khi các cờ nhị phân chỉ nhận `0/1`) khiến gradient của MLP hội tụ kém ($R^2 = 0,6024$, RMSE = `239.124,52 USD`). Khi áp dụng `StandardScaler` và one-hot `zipcode`, MLP trên pipeline 88 đặc trưng đạt $R^2 = 0,8354$. Hai số 0,6024 và 0,8354 thuộc hai pipeline khác nhau nên không so sánh trực tiếp như một mức tăng; chỉ ghi nhận xu hướng trong thí nghiệm ablation lịch sử này là chuẩn hóa giúp MLP hội tụ tốt hơn hẳn.
  - `Random Forest` đạt kết quả tốt nhất trên dữ liệu dạng bảng ($R^2 = 0,8802$, MAE = `72.558,23 USD`) do các thuật toán cây quyết định không bị ảnh hưởng bởi thang đo của đặc trưng.

## 5. Kết luận & Hướng cải tiến

- **Kết luận:** Pipeline tiền xử lý chống rò rỉ dữ liệu theo nhóm `id` kết hợp chuẩn hóa `StandardScaler` và one-hot `zipcode` đóng vai trò quyết định đến hiệu năng của mạng nơ-ron MLP, giúp mô hình vượt qua hồi quy tuyến tính cơ sở trên toàn bộ các chỉ số đánh giá (`MSE`, `RMSE`, `MAE`, `R2`).
- **Hướng cải tiến:**
  1. Đã xong (2026-11-17): các cell huấn luyện (Cell 30-44) đã dùng đồng nhất ma trận 88 đặc trưng `X_train_prep`, `X_val_prep`, `X_test_prep` cho cả 4 mô hình; bảng so sánh cuối cùng đã hoàn thiện.
  2. Thử nghiệm biến đổi `log1p(price)` cho biến mục tiêu để giảm độ lệch phải (`skew = 4,02`), giúp giảm sai số trên phân khúc nhà cao cấp trên 1 triệu USD, đồng thời bổ sung Early Stopping dựa trên `val_loss`.
```
