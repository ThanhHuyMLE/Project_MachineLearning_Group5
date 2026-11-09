# Mô tả dữ liệu King County

## Nguồn dữ liệu

- Tên: King County House Sales (bang Washington, Hoa Kỳ, gồm Seattle).
- File: `data/kc_house_data_NaN/data_raw/kc_house_data_NaN.csv`.
- Thời gian giao dịch: tháng 5/2014 đến tháng 5/2015.
- Biến mục tiêu: `price` (giá bán, USD).
- Dự án dùng dữ liệu này thay cho Boston Housing trong đề ví dụ.
  Lý do: file đã có sẵn trong repo, đủ lớn, có giá trị khuyết thật để xử lý.
  Xem [adr/002-tap-du-lieu-king-county.md](adr/002-tap-du-lieu-king-county.md).
  Lưu ý: `sklearn.datasets.load_boston()` đã bị xóa khỏi scikit-learn hiện đại,
  không cố gắng khôi phục API này.

## Số liệu đã kiểm chứng (đọc trực tiếp từ file CSV)

| Mục | Giá trị |
|---|---|
| Số dòng dữ liệu | 21.613 (không tính dòng tiêu đề) |
| Số cột | 22 (gồm 1 cột index `Unnamed: 0`, 1 biến mục tiêu `price`) |
| Giá trị khuyết `bedrooms` | 13 |
| Giá trị khuyết `bathrooms` | 10 |
| Các cột khác | Không có giá trị khuyết (theo DESIGN.md) |
| Giá `price` nhỏ nhất | 75.000 USD |
| Giá `price` lớn nhất | 7.700.000 USD |
| Số `id` trùng lặp | 177 id xuất hiện nhiều hơn 1 lần |

Ghi chú: 177 `id` trùng không đồng nghĩa 177 dòng trùng hoàn toàn.
Sinh viên 1 kiểm tra các dòng này có phải bản ghi trùng thật hay cùng căn nhà
bán nhiều lần, rồi quyết định giữ hay loại, ghi lại trong tiến trình.

## Danh sách cột

| Cột | Ý nghĩa | Kiểu |
|---|---|---|
| `Unnamed: 0` | Chỉ mục dòng gốc | int - LOẠI |
| `id` | Mã định danh căn nhà | int - LOẠI |
| `date` | Ngày giao dịch | text - LOẠI (hoặc trích năm/tháng nếu cần) |
| `price` | Giá bán (biến mục tiêu) | float |
| `bedrooms` | Số phòng ngủ | float (có NaN) |
| `bathrooms` | Số phòng tắm | float (có NaN) |
| `sqft_living` | Diện tích sống (sqft) | int |
| `sqft_lot` | Diện tích đất (sqft) | int |
| `floors` | Số tầng | float |
| `waterfront` | Giáp mặt nước (0/1) | int |
| `view` | Điểm hướng nhìn (0-4) | int |
| `condition` | Tình trạng bảo trì (1-5) | int |
| `grade` | Điểm chất lượng xây dựng (1-13) | int |
| `sqft_above` | Diện tích trên mặt đất | int |
| `sqft_basement` | Diện tích tầng hầm | int |
| `yr_built` | Năm xây dựng | int |
| `yr_renovated` | Năm cải tạo gần nhất (0 = chưa cải tạo) | int |
| `zipcode` | Mã bưu điện | int |
| `lat` | Vĩ độ | float |
| `long` | Kinh độ | float |
| `sqft_living15` | Diện tích sống của 15 căn lân cận | int |
| `sqft_lot15` | Diện tích đất của 15 căn lân cận | int |

Sau khi loại 3 cột, còn **18 đặc trưng** làm đầu vào mô hình.

## Quyết định xử lý cần Sinh viên 1 chốt

1. **Bù khuyết:** `bedrooms`, `bathrooms` bù bằng median của tập train.
2. **Trùng lặp:** kiểm tra 177 `id` trùng, quyết định giữ/loại có ghi lý do.
3. **`zipcode`:** hiện đang xử lý như số liên tục. Cần nhấn mạnh đây là
   mã vùng, không phải đại lượng có thứ tự; nếu giữ nguyên thì ghi rõ
   giới hạn này trong báo cáo, hoặc thử phương án mã hóa phù hợp.
4. **Giá trị ngoại lai (outlier):** giá lớn nhất 7,7 triệu USD gấp ~100 lần
   giá nhỏ nhất; cần EDA phân bố giá trước khi quyết định có xử lý không.
5. **Kỹ thuật đặc trưng (Feature Engineering):** chỉ thêm nếu có lý do,
   ví dụ tuổi nhà `age = 2015 - yr_built`, có cải tạo hay chưa.
   Không tạo đặc trưng từ biến mục tiêu `price` (rò rỉ dữ liệu).

## Hợp đồng dữ liệu

- File CSV phải có cột `price` và đủ 18 cột đặc trưng ở bảng trên.
- `X` là ma trận số sau khi chuẩn hóa, `y` là vector giá thật.
- Chia train/validation/test 70/15/15, `random_state=42`.
- Imputer và scaler chỉ fit trên train.
