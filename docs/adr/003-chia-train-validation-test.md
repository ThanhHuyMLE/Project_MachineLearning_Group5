# ADR-003: Chia train/validation/test và chống rò rỉ dữ liệu

- Trạng thái: Đã chấp nhận (Phase 1, 09/11/2026)
- Ưu tiên: yêu cầu của giảng viên về chống rò rỉ dữ liệu (Data Leakage)

## Bối cảnh

Tài liệu DESIGN.md gốc chỉ chia train/test 80/20 và dùng tập test làm
validation trong vòng lặp huấn luyện (mục 5.3, bước 5). Điều này làm
điểm validation lạc quan giả và biến test thành công cụ chọn mô hình,
không còn là phép đo độc lập.

## Quyết định

1. Chia thành 3 tập: train/validation/test = 70/15/15, `random_state=42`.
2. Thứ tự bắt buộc: chia trước, fit imputer/scaler chỉ trên train,
   transform validation, test và dữ liệu mới bằng cùng đối tượng đã fit.
3. Chọn mô hình và siêu tham số chỉ trên validation. Tập test chỉ dùng
   một lần cuối để đánh giá.
4. Cả 4 mô hình dùng cùng các tập chia sẵn để so sánh công bằng.
5. Quyết định này thay thế phần chia 80/20 và cách dùng test làm validation
   trong DESIGN.md. File DESIGN.md được giữ nguyên để bảo toàn lịch sử;
   điểm khác biệt được ghi lại tại đây và trong `docs/project_architecture.md`.

## Đánh đổi chấp nhận

- Được: đánh giá trung thực, đúng khoa học, dễ giải thích khi vấn đáp.
- Mất: tập train nhỏ hơn (70% thay vì 80%); chấp nhận được với 21.613 mẫu.
