# Mô tả dữ liệu King County

## 1. Nguồn gốc và mục tiêu

- Tên: King County House Sales (bang Washington, Hoa Kỳ, gồm Seattle).
- File: `data/kc_house_data_NaN/data_raw/kc_house_data_NaN.csv`.
- Thời gian giao dịch: tháng 5/2014 đến tháng 5/2015.
- Bài toán: Hồi quy có giám sát, dự đoán `price` (USD) từ đặc trưng căn nhà.
- Dự án dùng dữ liệu này thay cho Boston Housing trong đề ví dụ: file đã
  có sẵn trong repo, đủ lớn, có giá trị khuyết thật để xử lý. Xem [adr/002-tap-du-lieu-king-county.md](adr/002-tap-du-lieu-king-county.md).
- Lưu ý: `sklearn.datasets.load_boston()` đã bị xóa khỏi scikit-learn hiện đại, không cố gắng khôi phục API này.
- Câu hỏi vấn đáp về dataset và tiền xử lý: xem [oral-questions-preprocessing.md](oral-questions-preprocessing.md).

## 2. Cấu trúc dataset

- 21.613 dòng, 22 cột (gồm 1 cột index `Unnamed: 0`, 1 biến mục tiêu `price`, 1 mã `id`, 1 cột `date`).
- Sau khi loại `Unnamed: 0`, `id`, `date` và kỹ thuật đặc trưng, còn 19 cột đầu vào (18 số + `zipcode`).
- Sau mã hóa one-hot `zipcode` (70 mã), mô hình nhận 88 cột đầu ra.
- Năm bán: 2014 có 14.633 dòng, 2015 có 6.980 dòng.

## 3. Ý nghĩa từng feature

19 cột đầu vào (`FEATURE_COLUMNS`), theo đúng thứ tự trong hợp đồng dữ liệu:

| Cột | Ý nghĩa | Kiểu |
|---|---|---|
| `bedrooms` | Số phòng ngủ | float (có NaN) |
| `bathrooms` | Số phòng tắm | float (có NaN) |
| `sqft_living` | Diện tích sống (sqft) | int |
| `sqft_lot` | Diện tích đất (sqft) | int |
| `floors` | Số tầng | float |
| `waterfront` | Giáp mặt nước (0/1) | int |
| `view` | Điểm hướng nhìn (0-4) | int |
| `condition` | Tình trạng bảo trì (1-5) | int |
| `grade` | Điểm chất lượng xây dựng (1-13) | int |
| `sqft_above` | Diện tích trên mặt đất (sqft) | int |
| `sqft_basement` | Diện tích tầng hầm (sqft) | int |
| `house_age` | Tuổi nhà tại năm bán (năm) | int, xem mục 9 |
| `was_renovated` | Đã từng cải tạo (0/1) | int, xem mục 9 |
| `sqft_per_bedroom` | Diện tích sống trung bình mỗi phòng ngủ | float, xem mục 9 |
| `lat` | Vĩ độ | float |
| `long` | Kinh độ | float |
| `sqft_living15` | Diện tích sống của 15 căn lân cận (sqft) | int |
| `sqft_lot15` | Diện tích đất của 15 căn lân cận (sqft) | int |
| `zipcode` | Mã bưu điện (mã vùng, không phải số liên tục) | phân loại, xem mục 11 |

Hai cột gốc `yr_built` và `yr_renovated` không đưa trực tiếp vào mô hình mà được thay bằng `house_age` và `was_renovated` (xem mục 9). `id` không phải đặc trưng: nó chỉ dùng để chia tập theo nhóm (xem mục 10).

## 4. Biến mục tiêu price

- Đơn vị USD. Nhỏ nhất 75.000, Q1 321.950, trung vị 450.000, trung bình 540.088, Q3 645.000, lớn nhất 7.700.000.
- Độ lệch chuẩn 367.127, độ lệch (skew) 4,02, độ nhọn (kurtosis) 34,59: phân bố lệch phải mạnh, đuôi phải dài.
- P90 887.000, P95 1.156.480, P99 1.964.400, P99,5 2.453.290, P99,9 3.476.302. Trên 1 triệu USD: 1.465 dòng (6,8%); trên 2 triệu: 198 dòng (0,9%).
- Ngưỡng trên theo IQR là 1.129.575 USD, có 1.146 dòng vượt (5,3%). Đây là đặc trưng của thị trường nhà cao cấp, không phải lỗi nhập liệu.

## 5. Missing values

- `bedrooms` khuyết 13 dòng (0,06%), `bathrooms` khuyết 10 dòng (0,05%).
- Các cột khác không có giá trị khuyết.
- Cách xử lý: bù bằng median của tập train trong preprocessor (xem mục 11). Không xóa dòng khuyết vì số lượng quá nhỏ so với 21.613 dòng và việc xóa không làm thay đổi kết quả đáng kể.
- `sqft_per_bedroom` có thêm 13 NaN do `bedrooms` khuyết, cũng bù median.

## 6. Duplicate

- Không có dòng trùng hoàn toàn (0 dòng giống hệt nhau ở mọi cột).
- Đính chính số liệu ở Architecture Setup Milestone: con số đúng là **176 id lặp** (175 id xuất hiện 2 lần, 1 id xuất hiện 3 lần), liên quan 353 dòng, tức 177 dòng thừa. Architecture Setup Milestone ghi "177 id trùng" là sai: 177 là số dòng thừa, không phải số id lặp.
- 0 nhóm id trùng cùng ngày giao dịch; khoảng cách ngày lớn nhất trong cùng một id là 315 ngày. 173/176 nhóm có giá khác nhau giữa các lần bán.
- 3 nhóm giống hệt nhau ở mọi cột mô hình trừ `id`/`date`, nghi là bản ghi trùng, nhưng vẫn giữ lại vì không chứng minh được là lỗi.
- Kết luận: id lặp phần lớn là cùng căn nhà bán nhiều lần, không phải lỗi trùng dữ liệu. Vì vậy không xóa dòng nào, mà chia tập theo nhóm id (xem mục 10 và [adr/005-chia-theo-nhom-id.md](adr/005-chia-theo-nhom-id.md)).

## 7. Outlier

- 3 căn đắt nhất: 7,7 triệu USD (12.050 sqft, grade 13), 7,06 triệu (waterfront), 6,89 triệu. Cả ba đều nhất quán với diện tích, grade và waterfront nên giữ lại.
- 3 căn rẻ nhất: 75.000, 78.000, 80.000 USD (nhà nhỏ 430-780 sqft, grade 3-5). Giữ lại.
- `bedrooms` 1-33: có 1 dòng 33 phòng ngủ trên 1.620 sqft, giữ lại và ghi chú nghi ngờ nhập liệu. Trên 6 phòng ngủ: 62 dòng.
- `bathrooms` 0,5-8, không có giá trị 0. `sqft_living` 290-13.540, trung vị 1.910. `sqft_lot` 520-1.651.359, trung vị 7.618.
- `yr_built` 1900-2015, 12 dòng có `yr_built` lớn hơn năm bán. `yr_renovated` bằng 0 ở 20.699 dòng (95,8%), khác 0 ở 914 dòng (4,2%), 6 dòng lớn hơn năm bán (đều là cải tạo 2015 trên nhà bán 2014, được tính là chưa cải tạo ở mục 9), 0 dòng nhỏ hơn `yr_built`.
- `floors` chỉ nhận 1/1,5/2/2,5/3/3,5. `waterfront` bằng 1 ở 163 dòng (0,75%). `view` bằng 0 ở 19.489 dòng. `condition` bằng 3 ở 14.031 dòng. `grade` bằng 7 ở 8.981 dòng, bằng 8 ở 6.068 dòng, bằng 1 ở 1 dòng, bằng 13 ở 13 dòng. `sqft_basement` bằng 0 ở 13.126 dòng.
- `lat` 47,156-47,778, `long` -122,519 đến -121,315.
- Không xóa dòng nào trong toàn bộ Data Pipeline Implementation Milestone.

## 8. EDA

- Tương quan với `price` (cao đến thấp): `sqft_living` 0,702, `grade` 0,667,
  `sqft_above` 0,606, `sqft_living15` 0,585, `bathrooms` 0,526, `view` 0,397,
  `sqft_basement` 0,324, `bedrooms` 0,309, `lat` 0,307, `waterfront` 0,266,
  `floors` 0,257, `yr_renovated` 0,126, `sqft_lot` 0,090, `sqft_lot15` 0,082,
  `yr_built` 0,054, `zipcode` -0,053, `condition` 0,036, `long` 0,022.
- Diện tích sống và chất lượng xây dựng là hai tín hiệu mạnh nhất. `zipcode` dạng số gần như không tương quan (-0,053) vì mã vùng không phải đại lượng có thứ tự; xem cách xử lý ở mục 11.
- Giá trung bình theo `grade` (USD): 1: 142.000 (n=1), 3: 205.667 (n=3), 4: 214.381 (n=29), 5: 248.524 (n=242), 6: 301.920 (n=2.038), 7: 402.590 (n=8.981), 8: 542.853 (n=6.068), 9: 773.513 (n=2.615), 10: 1.071.771 (n=1.134), 11: 1.496.842 (n=399), 12: 2.191.222 (n=90), 13: 3.709.615 (n=13). Giá tăng đơn điệu theo grade.
- Waterfront: không giáp 531.564 USD, giáp mặt nước 1.661.876 USD.
- Giá trung bình theo `view` (USD): 0: 496.564, 1: 812.281, 2: 792.401, 3: 971.965, 4: 1.463.711.
- `zipcode` có 70 mã (98.001-98.199), mỗi mã ít nhất 50 dòng, nhiều nhất 602 dòng. Giá trung bình theo zipcode 234.284-2.160.606 USD: vị trí địa lý ảnh hưởng lớn đến giá.

### EDA mô tả và EDA phục vụ phát triển mô hình

- Hai mục đích này khác nhau và phải tách rõ. **EDA mô tả** (descriptive EDA)
  dùng toàn bộ dataset để hiểu dữ liệu nguồn: mọi con số trong mục 8 này
  (tương quan, giá trung bình theo grade/view/waterfront/zipcode) đều tính trên
  toàn bộ 21.613 dòng, và 7 biểu đồ trong `outputs/eda/` cũng vẽ từ toàn bộ dữ liệu.
- **EDA phục vụ phát triển mô hình** (model-development EDA) dùng để quyết định
  chọn feature, loại outlier hay biến đổi dữ liệu. Phần này chỉ được tính trên
  tập train, vì nếu dùng số liệu toàn tập (đã gồm test) để ra quyết định thì
  test đã gián tiếp tham gia vào quyết định mô hình (decision leakage), làm
  điểm đánh giá trên test lạc quan hơn thực tế.
- Kiểm chứng đã làm sau khi chia tập: tính lại tương quan với `price` và giá
  trung bình theo grade chỉ trên train (15.129 dòng) rồi so với số toàn tập.
  Tương quan chênh lệch tối đa 0,007, thứ hạng các feature không đổi. Giá trung
  bình theo grade chênh lệch tối đa 6,5% (grade 4, n=29), còn lại không quá 2%,
  vẫn tăng đơn điệu theo grade. Kết luận: các quyết định ở mục 9 không đổi nếu
  chỉ dùng train. Ngoại lệ cần ghi rõ: grade 1 chỉ có 1 dòng nên không có mặt
  trong train (dòng đó rơi vào validation/test).
- Quy tắc cho Stage 02, Stage 03 và Stage 04 từ đây trở đi: mọi phân tích phục vụ chọn mô hình,
  chọn ngưỡng hay biến đổi dữ liệu phải tính trên train (hoặc train + validation),
  không tính lại trên toàn tập hay trên test.

## 9. Feature engineering

- `house_age = năm bán - yr_built`, kẹp tối thiểu 0. Khoảng -1 đến 115, trung vị 40, trung bình 43,3; 12 giá trị âm được kẹp về 0. Tương quan với `price` là -0,054. Thay thế `yr_built` vì hai cột gần như trùng thông tin trong cùng một năm bán.
- `was_renovated = 1` nếu `0 < yr_renovated <= năm bán`, ngược lại 0. Bằng 1 ở 908 dòng (4,20%), tương quan với `price` là 0,126. Thay thế `yr_renovated` vì giá trị 0 trong cột gốc trộn cờ hiệu "chưa cải tạo" với năm. 6 dòng có `yr_renovated` (2015) lớn hơn năm bán (2014) được tính là chưa cải tạo, vì không thể cải tạo sau khi đã bán.
- `sqft_per_bedroom = sqft_living / bedrooms`. Khoảng 49,1-3.420, trung vị 576,7, trung bình 618,2, 13 NaN (do `bedrooms` khuyết). Khi `bedrooms` bằng 0 hoặc khuyết thì kết quả là NaN và được bù median. Tương quan với `price` là 0,580.
- Không tạo đặc trưng nào từ biến mục tiêu `price` (rò rỉ dữ liệu).
  Xem [adr/006-feature-engineering-va-zipcode.md](adr/006-feature-engineering-va-zipcode.md).

## 10. Chiến lược chia tập

- Chia train/validation/test 70/15/15, `random_state=42`, theo nhóm `id` bằng `GroupShuffleSplit`: mọi giao dịch của cùng một căn nhà nằm trong cùng một partition.
- Kích thước thực: train 15.129, validation 3.243, test 3.241 dòng. Số id trùng nhau giữa các tập: 0.
- Chia ngẫu nhiên theo dòng (cách làm ở Architecture Setup Milestone) để lọt 41 dòng test và 38 dòng validation trùng id với train, làm điểm đánh giá lạc quan giả.
- Hồi quy không có nhãn lớp nên không stratify.

## 11. Chuẩn hóa và encoding

- 18 đặc trưng số: bù khuyết bằng median của tập train, rồi chuẩn hóa bằng `StandardScaler` với công thức:

```text
z = (x - mean_train) / std_train
```

- `mean_train` và `std_train` chỉ tính trên tập train. Mọi tập khác (validation, test, dữ liệu mới) chỉ được transform bằng cùng các thống kê này, không được fit lại.
- `zipcode` là mã vùng: bù bằng giá trị thường gặp nhất của tập train, rồi mã hóa one-hot thành 70 cột (`handle_unknown="ignore"`, mã chưa thấy ở train cho toàn 0). Không dùng `zipcode` dạng số liên tục.
- Tổng số cột đầu ra: 18 số + 70 one-hot = 88 cột, kiểu `float64`, không NaN/inf. Thứ tự cột cố định, lấy từ `output_feature_names`.

## 12. Chống leakage

- Thứ tự bắt buộc: kỹ thuật đặc trưng (không fit gì) -> chia tập theo nhóm id -> fit preprocessor chỉ trên train -> transform val/test.
- Imputer (median), scaler (mean/std) và one-hot encoder (tập mã zipcode) chỉ học từ train. Dữ liệu mới dùng đúng preprocessor đã fit.
- `id` không thành feature. Không dùng `price` để tạo đặc trưng.
- Rò rỉ quyết định: không chỉ thống kê fit (imputer, scaler, encoder) mà cả
  quyết định chọn feature, loại outlier hay biến đổi dữ liệu cũng không được
  dựa trên test. Xem cách tách EDA mô tả và EDA phục vụ phát triển mô hình
  ở mục 8.
- Chọn mô hình và siêu tham số chỉ trên validation. Tập test chỉ dùng
  một lần cuối để đánh giá.

## 13. Workflow tiền xử lý cuối cùng

```text
ĐẦU VÀO: CSV gốc 21.613 dòng, 22 cột
ĐẦU RA: X_train (15.129, 88), X_val (3.243, 88), X_test (3.241, 88),
        y_train, y_val, y_test (giá thật, USD)

1. Nạp CSV, kiểm tra có cột price
2. engineer_features: house_age, was_renovated, sqft_per_bedroom; loại yr_built, yr_renovated
3. Loại cột [Unnamed: 0, id, date] khỏi X (giữ id làm groups để chia)
4. Tách X (19 cột), y (price)
5. GroupShuffleSplit theo id, 70/15/15, random_state=42
6. build_preprocessor fit trên X_train: số: SimpleImputer(median) -> StandardScaler; zipcode: SimpleImputer(most_frequent) -> OneHotEncoder(ignore)
7. Transform X_train, X_val, X_test bằng cùng preprocessor
8. y_train trung bình 537.298 USD, y_test trung bình 545.893 USD
```

## 14. Hạn chế còn lại

- `date` chỉ dùng để tính `house_age` và làm nhóm chia; chưa khai thác xu hướng giá theo tháng hay mùa.
- `lat`/`long` dùng trực tiếp dạng số; chưa tạo đặc trưng khoảng cách đến trung tâm Seattle hay đến mặt nước.
- `zipcode` one-hot 70 cột làm đầu ra tăng lên 88 cột; mô hình tuyến tính vẫn dùng được, nhưng chưa thử gom nhóm zipcode theo giá trung bình.
- 3 nhóm id giống hệt ở mọi cột mô hình trừ `id`/`date` vẫn giữ lại (xem mục 6); nếu là bản ghi trùng thật thì chúng gây nhiễu nhỏ.
- Dòng 33 phòng ngủ trên 1.620 sqft vẫn giữ lại dù nghi ngờ nhập liệu.
- Phân bố `price` lệch phải mạnh; chưa thử biến đổi log cho mục tiêu,
  đó là việc của các Stage mô hình nếu cần.
- Các số EDA mô tả ở mục 8 tính trên toàn bộ dataset; việc kiểm chứng lại
  trên train (xem mục 8) làm sau khi chia tập, chỉ để xác nhận quyết định
  không đổi. Đúng trình tự lý tưởng là tính EDA phục vụ quyết định trên train
  ngay từ đầu, nhóm cần tuân thủ trình tự này ở các Stage sau.
