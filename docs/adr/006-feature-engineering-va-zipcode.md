# ADR-006: Feature engineering và xử lý zipcode

- Trạng thái: Đã chấp nhận (Phase 2, 09/11/2026)
- Ưu tiên: đặc trưng đầu vào đúng ý nghĩa, zipcode không còn xử lý sai
  dạng số liên tục

## Bối cảnh

Hợp đồng dữ liệu Phase 1 dùng 18 đặc trưng gốc, trong đó `yr_built` và
`yr_renovated` mang thông tin dưới dạng thô chưa phù hợp, và `zipcode`
đang xử lý như số liên tục. Cụ thể: 12 dòng có `yr_built` lớn hơn năm
bán (tuổi âm, lỗi ghi dữ liệu); `yr_renovated` bằng 0 ở 20.699 dòng
(95,8%) nghĩa là chưa cải tạo, nên giá trị số của nó trộn cờ hiệu với
năm; `zipcode` dạng số tương quan với `price` chỉ -0,053, trong khi giá
trung bình theo zipcode dao động 234.284-2.160.606 USD.

## Quyết định

1. `house_age = năm bán - yr_built`, kẹp tối thiểu 0, thay thế `yr_built`.
   Khoảng -1 đến 115, trung vị 40, trung bình 43,3; 12 giá trị âm kẹp
   về 0 thay vì xóa dòng. Tương quan với `price` là -0,054.
2. `was_renovated = 1` nếu `0 < yr_renovated <= năm bán`, ngược lại 0, thay thế
   `yr_renovated`. Bằng 1 ở 908 dòng (4,20%), tương quan với `price` là 0,126.
   6 dòng có `yr_renovated` lớn hơn năm bán được tính là chưa cải tạo.
3. Thêm `sqft_per_bedroom = sqft_living / bedrooms`. Khoảng 49,1-3.420,
   trung vị 576,7, trung bình 618,2, tương quan với `price` là 0,580.
   Khi `bedrooms` bằng 0 hoặc khuyết thì kết quả là NaN và được bù bằng
   median của tập train, không bao giờ chia cho 0.
4. `zipcode` là mã vùng: bù bằng giá trị thường gặp nhất của tập train,
   rồi mã hóa one-hot thành 70 cột với `handle_unknown="ignore"`
   (mã chưa thấy ở train cho toàn 0). Không dùng `zipcode` dạng số.
5. Ba đặc trưng mới do hàm `engineer_features` tạo ra, không fit gì cả,
   chạy trước khi chia tập. Không tạo đặc trưng nào từ `price`.
6. Hợp đồng dữ liệu mới: 19 cột đầu vào (`FEATURE_COLUMNS`, 18 số +
   `zipcode`), xem `docs/king-county-dataset.md` và
   `docs/project_architecture.md`.

## Đánh đổi chấp nhận

- Được: đặc trưng đúng ý nghĩa hơn (`house_age` thay năm xây thô,
  cờ cải tạo thay năm trộn cờ hiệu), vị trí địa lý được mô hình dùng
  đúng cách qua one-hot thay vì số liên tục vô nghĩa.
- Mất: số cột đầu ra của preprocessor tăng từ 19 lên 88 (18 số + 70
  one-hot), mô hình nhận đầu vào rộng hơn; chấp nhận được vì 70 mã
  zipcode mỗi mã ít nhất 50 dòng, không có mã hiếm gây cột toàn 0.
