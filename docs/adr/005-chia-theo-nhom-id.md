# ADR-005: Chia train/validation/test theo nhóm id

- Trạng thái: Đã chấp nhận (Phase 2, 09/11/2026)
- Ưu tiên: chống rò rỉ dữ liệu do cùng căn nhà xuất hiện nhiều lần

## Bối cảnh

Dataset có 21.613 dòng nhưng chỉ 21.436 id duy nhất: 176 id lặp
(175 id xuất hiện 2 lần, 1 id xuất hiện 3 lần), liên quan 353 dòng,
tức 177 dòng thừa. 173/176 nhóm có giá khác nhau giữa các lần bán,
0 nhóm trùng ngày, khoảng cách ngày lớn nhất trong cùng id là 315 ngày.
Đây phần lớn là cùng căn nhà bán nhiều lần, không phải lỗi trùng dữ liệu.

Chia ngẫu nhiên theo dòng (cách làm Phase 1, xem ADR-003) để lọt 41 dòng
test và 38 dòng validation trùng id với train. Mô hình đã thấy chính căn
nhà đó ở train, chỉ cần nhớ giá cũ là đoán gần đúng, nên điểm đánh giá
lạc quan giả. Đính chính số liệu Phase 1: "177 id trùng" là sai,
177 là số dòng thừa, số id lặp đúng là 176.

## Các lựa chọn đã xem xét

1. **Giữ chia ngẫu nhiên theo dòng (loại):** đơn giản, giữ đúng ADR-003,
   nhưng để rò rỉ 41 dòng test + 38 dòng validation trùng id với train.
2. **Xóa các dòng trùng id (loại):** hết rò rỉ, nhưng xóa 177 dòng dữ liệu
   thật (cùng nhà bán nhiều lần là hiện tượng thị trường bình thường),
   và làm mất thông tin biến động giá theo thời gian của cùng căn nhà.
3. **Chia theo nhóm id (chọn):** `GroupShuffleSplit` theo `id`, mọi giao
   dịch của cùng một căn nhà nằm trong cùng một partition.

## Quyết định

1. Chia train/validation/test 70/15/15 bằng `GroupShuffleSplit` theo `id`,
   `random_state=42`. Kích thước thực: train 15.129, validation 3.243,
   test 3.241 dòng. Số id trùng nhau giữa các tập: 0.
2. `id` chỉ dùng làm `groups` khi chia, không thành feature của mô hình.
3. `split_train_val_test` nhận thêm tham số `groups`; khi `groups` là None
   thì chia ngẫu nhiên như Phase 1 (giữ để tương thích test cũ).
4. `prepare_data` trả thêm `id_train`, `id_val`, `id_test` chỉ để kiểm
   chứng việc chia theo nhóm, không phải đầu vào mô hình.

## Đánh đổi chấp nhận

- Được: đánh giá đo khả năng dự đoán căn nhà chưa thấy, trung thực hơn;
  không mất dòng dữ liệu nào.
- Mất: phép đo khắt khe hơn chia ngẫu nhiên, điểm test có thể thấp hơn
  vì mô hình không còn được "nhớ" căn nhà đã thấy; chấp nhận được vì đó
  mới là khả năng tổng quát thật. Tỷ lệ 70/15/15 giữ nguyên.
