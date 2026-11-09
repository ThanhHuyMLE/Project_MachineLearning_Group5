# ADR-002: Dùng dữ liệu King County thay cho Boston Housing

- Trạng thái: Đã chấp nhận (Phase 1, 09/11/2026)

## Bối cảnh

Đề bài nêu Boston Housing làm ví dụ, nhưng repo đã có sẵn thư mục
`data/kc_house_data_NaN/` với file CSV King County House Sales.

## Các lựa chọn đã xem xét

1. **King County (chọn):** file đã có trong repo, 21.613 mẫu, có giá trị
   khuyết thật (13 bedrooms, 10 bathrooms) để luyện xử lý dữ liệu.
2. **Boston Housing:** `sklearn.datasets.load_boston()` đã bị xóa khỏi
   scikit-learn hiện đại; tải lại từ nguồn ngoài thêm phụ thuộc lỗi thời
   và bộ dữ liệu 506 mẫu nhỏ hơn nhiều. Không có lý do đổi nếu King County
   đáp ứng được đề bài.

## Quyết định

Giữ King County làm dữ liệu chính của Bài tập 4. Ghi rõ trong
`docs/king-county-dataset.md` và không tuyên bố đây là Boston Housing.

## Đánh đổi chấp nhận

- Được: dữ liệu thật, lớn, có khuyết và ngoại lai để phân tích.
- Mất: không khớp từng chữ ví dụ trong đề; cần ghi rõ trong báo cáo.
  Cần quyết định thêm về `zipcode` (mã vùng) và 177 `id` trùng lặp.
