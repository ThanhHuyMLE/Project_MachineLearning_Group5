# ADR-004: MLP triển khai bằng sklearn MLPRegressor trước, Keras là lựa chọn sau

- Trạng thái: Đã chấp nhận tạm thời (Phase 1, 09/11/2026), cần Sinh viên 2 xác nhận

## Bối cảnh

Đề bài và DESIGN.md mô tả MLP bằng TensorFlow/Keras (Sequential Dense).
Kiểm tra ngày 09/11/2026: môi trường Python 3.14 trên máy nhóm
(venv của dự án CLASSIFYING-SPAM-EMAILS) có scikit-learn 1.9.1,
pandas 3.0.6, numpy 2.5.4 nhưng **không có TensorFlow** và chưa thử cài đặt.

## Các lựa chọn đã xem xét

1. **sklearn MLPRegressor (chọn làm mặc định):** cùng kiến trúc
   hidden (32, 16), relu, solver adam, loss MSE; cài sẵn, chạy được ngay,
   đủ để học lý thuyết MLP. Giới hạn: ít tùy chỉnh kiến trúc sâu như Keras.
2. **TensorFlow/Keras:** đúng kiến trúc trong đề bài. Rủi ro: cài đặt nặng,
   có thể chưa hỗ trợ Python 3.14, cần Sinh viên 2 kiểm tra trước khi chọn.
3. **Viết MLP từ đầu bằng NumPy:** chỉ làm nếu giảng viên yêu cầu rõ ràng.
   Hiện không có yêu cầu đó; không làm.

## Quyết định

- MLP được tự triển khai bằng NumPy trong
  `src/house_price/models/mlp.py`.
- Các phép Forward Propagation, Backpropagation, MSE và Adam nằm trong
  class MLP; không dùng `sklearn.MLPRegressor`, TensorFlow hoặc Keras
  để huấn luyện MLP.
- scikit-learn vẫn được dùng trong pipeline tiền xử lý và đánh giá
  theo các module tương ứng.
- MLP dùng hidden layers `(32, 16)`, ReLU, output Linear, Adam,
  learning rate `0.001`, tối đa 100 epoch và batch size 32.
- Sau khi `fit()` hoàn tất, cấu hình, loss history, weights và biases
  được lưu vào `outputs/mlp_parameters.json`.

## Đánh đổi chấp nhận

- Được: pipeline chạy được ngay, không bị kẹt ở cài đặt.
- Mất: mất một số tùy chỉnh của Keras (callback, kiến trúc tự do);
  nếu cần, bổ sung sau mà không phá vỡ interface.
