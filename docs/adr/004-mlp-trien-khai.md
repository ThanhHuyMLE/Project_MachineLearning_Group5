# ADR-004: MLP được cài đặt bằng thư viện NumPy (thay thế quyết định sklearn tạm thời)

- Trạng thái: Đã thay thế bởi [ADR-007](007-mo-hinh-numpy.md)
  (10/11/2026). Quyết định gốc dưới đây chỉ giữ lại vì lý do lịch sử;
  không dùng làm căn cứ cho code hiện tại.
- Trạng thái gốc: Đã chấp nhận tạm thời (Architecture Setup Milestone,
  09/11/2026), cần xác nhận

## Bối cảnh gốc (09/11/2026)

Đề bài và DESIGN.md mô tả MLP bằng TensorFlow/Keras (Sequential Dense).
Kiểm tra ngày 09/11/2026: môi trường Python 3.14 trên máy nhóm
(venv của dự án CLASSIFYING-SPAM-EMAILS) có scikit-learn 1.9.1,
pandas 3.0.6, numpy 2.5.4 nhưng **không có TensorFlow** và chưa thử cài đặt.

## Các lựa chọn đã xem xét (gốc)

1. **sklearn MLPRegressor (đã chọn làm mặc định tạm thời lúc đó):**
   cùng kiến trúc hidden (32, 16), relu, solver adam, loss MSE; cài
   sẵn, chạy được ngay, đủ để học lý thuyết MLP. Giới hạn: ít tùy
   chỉnh kiến trúc sâu như Keras.
2. **TensorFlow/Keras:** đúng kiến trúc trong đề bài. Rủi ro: cài đặt
   nặng, có thể chưa hỗ trợ Python 3.14, cần xác nhận trước khi chọn.
3. **Viết MLP từ đầu bằng NumPy:** chỉ làm nếu giảng viên yêu cầu rõ
   ràng. Lúc đó không có yêu cầu đó; không làm.

## Quyết định gốc (09/11/2026, đã thay thế)

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

## Vì sao tiêu đề gốc mâu thuẫn với nội dung

Tiêu đề gốc nói "sklearn MLPRegressor trước, Keras là lựa chọn sau",
nhưng mục Quyết định gốc đã ghi "MLP được tự triển khai bằng NumPy"
và không dùng `sklearn.MLPRegressor`. Mâu thuẫn này xuất phát từ việc
file được cập nhật nội dung quyết định sau khi mô hình MLP được cài đặt bằng thư viện NumPy hoàn thành, nhưng tiêu đề không được sửa theo. Ngày 10/11/2026, quy tắc
dùng thư viện NumPy để cài đặt mô hình của nhóm (xem ADR-007) đã thay thế hoàn toàn lựa chọn
sklearn tạm thời, nên tiêu đề cũ không còn đúng ở bất kỳ phần nào.

## Quyết định hiện tại

- Tiêu đề file này đã sửa thành "MLP được cài đặt bằng thư viện NumPy
  (thay thế quyết định sklearn tạm thời)" để khớp với thực tế.
- Trạng thái chuyển thành "Đã thay thế bởi ADR-007"; ADR-007 là căn
  cứ hiện hành cho mọi mô hình, không chỉ MLP.
- Code MLP hiện tại (`src/house_price/models/mlp.py`) tuân theo
  ADR-007, không dùng `sklearn.MLPRegressor`, TensorFlow hay Keras.
