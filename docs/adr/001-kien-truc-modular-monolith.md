# ADR-001: Kiến trúc modular monolith phân lớp đơn giản

- Trạng thái: Đã chấp nhận (Phase 1, 09/11/2026)
- Nguồn: senior_arch, workflow chọn kiểu kiến trúc (nhóm 4 người -> modular monolith)

## Bối cảnh

Dự án là bài tập môn học của nhóm 4 sinh viên, một bài tập Hồi quy duy nhất
trong Phase 1, có thể thêm bài tập khác sau này. Cần kiến trúc dễ hiểu,
dễ giải thích khi vấn đáp, dễ chia việc mà không thêm hạ tầng vận hành.

## Các lựa chọn đã xem xét

1. **Modular monolith phân lớp (chọn):** một package Python, các module
   tách theo trách nhiệm (preprocessing, models, evaluation, train_evaluate).
2. **Microservices:** mỗi mô hình một service. Loại: nhóm chỉ 4 người,
   ranh giới miền chưa rõ, thêm vận hành mà không thêm giá trị học tập.
3. **Notebook đơn khối:** toàn bộ code trong một notebook. Loại: khó review,
   khó test, khó tái sử dụng, dễ rò rỉ dữ liệu khi fit scaler trước khi chia.

## Quyết định

Dùng modular monolith: package `src/house_price/` với các module độc lập,
notebook chỉ demo, không có database, API, Docker hay CI/CD trong Phase 1.

## Đánh đổi chấp nhận

- Được: đơn giản, chạy được ngay trên máy sinh viên, dễ vấn đáp từng module.
- Mất: không mở rộng độc lập từng mô hình; chấp nhận được vì đây là bài tập,
  không phải hệ thống production.
