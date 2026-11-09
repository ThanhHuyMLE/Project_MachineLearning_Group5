# Câu hỏi vấn đáp: Dataset & Preprocessing

File này tập hợp câu hỏi vấn đáp về dataset King County và tiền xử lý,
trả lời bám sát số liệu thật trong
[king-county-dataset.md](king-county-dataset.md).
Mỗi câu trả lời ngắn gọn, đủ để trình bày miệng trong vài câu.

## 1. Supervised learning là gì?

Là học có giám sát: mô hình học từ dữ liệu đã có nhãn. Ở đây mỗi dòng có
đầu vào X (đặc trưng căn nhà) và nhãn y là `price` đã biết. Mô hình học
ánh xạ từ X sang y trên tập train, rồi dự đoán giá cho nhà chưa thấy.
Khác với học không giám sát là dữ liệu không có nhãn để học theo.

## 2. Regression là gì?

Là bài toán dự đoán một giá trị số liên tục. Ở đây `price` là số tiền
liên tục (75.000-7.700.000 USD), không phải nhãn rời rạc như phân loại.
Vì vậy dùng MSE, RMSE, MAE, R2 để đánh giá, không dùng accuracy,
precision, recall hay F1 (đó là chỉ số phân loại).

## 3. X và y trong bài này là gì?

X là ma trận đầu vào: 19 cột trước mã hóa (18 đặc trưng số + `zipcode`),
sau mã hóa one-hot thành 88 cột. y là vector mục tiêu `price`, giá bán
thật tính bằng USD, không chuẩn hóa. Mỗi dòng là một giao dịch mua bán
một căn nhà, không phải một căn nhà duy nhất vì có id bán nhiều lần.

## 4. Vì sao cần EDA trước khi xây mô hình?

EDA (khám phá dữ liệu) cho biết dữ liệu thật sự chứa gì trước khi quyết
định xử lý: phân bố `price` lệch phải mạnh (skew 4,02), `bedrooms` khuyết
13 và `bathrooms` khuyết 10, 176 id lặp, `sqft_living` tương quan với giá
mạnh nhất (0,702). Không có EDA thì dễ chọn sai cách bù khuyết, chia tập
sai và hiểu sai kết quả mô hình.

## 5. Missing value là gì và dataset này khuyết ở đâu?

Missing value là ô dữ liệu trống, không ghi nhận được. Dataset này khuyết
rất ít: `bedrooms` 13 dòng (0,06%) và `bathrooms` 10 dòng (0,05%),
các cột khác không khuyết. `sqft_per_bedroom` có thêm 13 NaN vì nó được
tính từ `bedrooms` khuyết. Số khuyết nhỏ nên bù giá trị là đủ, không
cần xóa dòng.

## 6. Vì sao chọn bù median mà không bù mean hay xóa dòng?

Median (trung vị) không nhạy với outlier như mean. Phân bố giá và diện
tích lệch phải mạnh, mean bị kéo lên bởi nhà đắt, median đại diện tốt hơn
cho căn nhà điển hình. Xóa dòng khuyết cũng được vì chỉ mất 23 dòng,
nhưng bù median giữ nguyên 21.613 dòng và giữ đúng quy trình pipeline
cho dữ liệu mới cũng có thể khuyết.

## 7. StandardScaler là gì?

Là chuẩn hóa z-score: trừ đi trung bình rồi chia cho độ lệch chuẩn,
theo công thức `z = (x - mean_train) / std_train`. Sau chuẩn hóa, mỗi
đặc trưng có trung bình 0 và độ lệch chuẩn 1 trên tập train. Việc này
cần cho MLP và Linear Regression vì chúng nhạy với thang đo khác nhau
giữa các đặc trưng (ví dụ `sqft_lot` lên tới 1.651.359 trong khi
`bathrooms` chỉ tới 8).

## 8. Vì sao chia train/validation/test mà không chỉ train/test?

Ba tập có ba vai trò khác nhau: train để học, validation để chọn mô hình
và siêu tham số, test để đánh giá cuối cùng một lần duy nhất. Nếu chỉ
chia train/test rồi dùng test để chọn mô hình thì test không còn độc lập,
điểm đánh giá sẽ lạc quan giả. Tỷ lệ dùng là 70/15/15, `random_state=42`.

## 9. Data leakage là gì?

Là rò rỉ dữ liệu: thông tin từ tập test (hoặc từ mục tiêu) lọt vào quá
trình huấn luyện, làm điểm đánh giá tốt giả tạo. Ví dụ: fit scaler trên
toàn bộ dữ liệu trước khi chia, dùng `price` để tạo đặc trưng, hay chọn
siêu tham số dựa trên điểm test. Cách chống: chia trước, chỉ fit trên
train, transform các tập còn lại bằng cùng đối tượng đã fit.

## 10. Vì sao scaler chỉ fit trên train?

Vì mean và std là thống kê học được từ dữ liệu. Nếu tính chúng trên toàn
bộ dữ liệu thì thông tin phân bố của validation và test đã lọt vào train,
đó là leakage. Fit chỉ trên train rồi transform val/test bằng cùng các
thống kê đó thì phép đo trên test mới phản ánh đúng khả năng dự đoán
dữ liệu chưa thấy. Cùng lý do áp dụng cho imputer median và one-hot
encoder.

## 11. Vì sao duplicate id quan trọng?

Có 176 id lặp (175 id 2 lần, 1 id 3 lần), liên quan 353 dòng. Đó phần lớn
là cùng căn nhà bán nhiều lần với giá khác nhau (173/176 nhóm), không phải
lỗi trùng dữ liệu. Nếu chia ngẫu nhiên theo dòng thì 41 dòng test và
38 dòng validation trùng id với train: mô hình đã thấy chính căn nhà đó,
chỉ cần nhớ giá cũ là đoán gần đúng. Vì vậy phải chia theo nhóm id
(xem [adr/005-chia-theo-nhom-id.md](adr/005-chia-theo-nhom-id.md)).

## 12. Feature engineering là gì?

Là tạo đặc trưng mới từ cột gốc để mô hình học dễ hơn. Bài này tạo ba
đặc trưng: `house_age = năm bán - yr_built` (kẹp tối thiểu 0),
`was_renovated = 1` nếu `yr_renovated > 0`, và
`sqft_per_bedroom = sqft_living / bedrooms`. Ví dụ `house_age` thay
`yr_built` vì tuổi nhà mới là thứ ảnh hưởng trực tiếp đến giá.
Không tạo đặc trưng nào từ `price` vì đó là leakage.

## 13. Vì sao zipcode cần xử lý đặc biệt?

`zipcode` là mã vùng, không phải số liên tục: 98.100 không "lớn hơn"
98.001 theo nghĩa giá trị, và tương quan dạng số của nó với giá chỉ
-0,053, gần như không có thông tin. Nhưng giá trung bình theo zipcode
dao động 234.284-2.160.606 USD nên vị trí rất quan trọng. Giải pháp là
mã hóa one-hot thành 70 cột, mỗi mã một cột 0/1, mã chưa thấy ở train
được bỏ qua (`handle_unknown="ignore"`).

## 14. Outlier là gì và dataset này có outlier nào?

Outlier là giá trị nằm xa phần lớn dữ liệu. Ví dụ: giá 7,7 triệu USD
trong khi trung vị chỉ 450.000, `sqft_lot` tới 1.651.359, 1 dòng có
33 phòng ngủ trên 1.620 sqft. Ngưỡng trên theo IQR là 1.129.575 USD,
có 1.146 dòng vượt (5,3%). Outlier ở đây phần lớn là nhà cao cấp thật,
không phải lỗi nhập liệu.

## 15. Vì sao không xóa hết outlier?

Vì outlier không đồng nghĩa lỗi. Ba căn đắt nhất (7,7 triệu, 7,06 triệu,
6,89 triệu USD) đều nhất quán với diện tích, grade và waterfront; ba căn
rẻ nhất (75.000-80.000 USD) đều là nhà nhỏ grade 3-5. Xóa chúng là xóa
một phần thị trường thật và làm mô hình không dự đoán được nhà cao cấp
hay nhà rẻ. Phase 2 không xóa dòng nào; chỉ ghi chú dòng 33 phòng ngủ
là nghi ngờ nhập liệu và vẫn giữ lại.
