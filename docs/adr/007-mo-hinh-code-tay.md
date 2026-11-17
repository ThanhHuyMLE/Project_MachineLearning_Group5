# ADR-007: Mọi mô hình code tay bằng NumPy, không dùng thư viện mô hình

- Trạng thái: Đã chấp nhận (10/11/2026), thay thế ADR-004
- Cập nhật: 17/11/2026, cả 4 mô hình đã code tay xong (xem mục Tiến độ)

## Bối cảnh

Ngày 10/11/2026, Thanh Huy (chủ nhóm) đặt quy tắc: mọi mô hình phải
CODE TAY để thể hiện cách tính của thuật toán, KHÔNG dùng thư viện hỗ
trợ cho model (không dùng class mô hình của scikit-learn, không dùng
TensorFlow/Keras). Thư viện vẫn dùng bình thường cho phần còn lại:
pandas/NumPy đọc dữ liệu, sklearn cho tiền xử lý (preprocessing) và
đo chỉ số (metrics), thư viện vẽ biểu đồ.

ADR-004 (09/11/2026) trước đây chọn sklearn MLPRegressor làm mặc định
tạm thời, với lý do "chỉ viết tay nếu giảng viên yêu cầu rõ ràng". Yêu
cầu đó giờ đã có, nên ADR-004 bị thay thế. `docs/description.md` từng
ghi "dùng thư viện ML có sẵn" cũng được sửa theo quy tắc mới.

## Các lựa chọn đã xem xét

1. **Code tay bằng NumPy (chọn):** tự viết công thức của từng thuật
   toán, NumPy chỉ làm phép tính mảng (ma trận, lấy mẫu ngẫu nhiên).
   Thể hiện được cách tính khi giảng viên hỏi, tham số học được đọc
   và xuất ra được.
2. **Giữ sklearn làm backend:** chạy nhanh, ít code, nhưng vi phạm quy
   tắc mới của nhóm và không thể hiện được cách tính.
3. **TensorFlow/Keras:** đúng kiến trúc trong đề bài cho MLP, nhưng vẫn
   là thư viện mô hình nên vi phạm quy tắc mới; ngoài ra chưa cài được
   trên Python 3.14.

## Quyết định

- Mọi mô hình trong `src/house_price/models/` đều code tay bằng NumPy,
  không import class mô hình của sklearn hay TensorFlow/Keras.
- Giữ nguyên interface chung `__init__` -> `build_model` -> `fit` ->
  `predict` (xem `docs/project_architecture.md`), để pipeline và code
  gọi không đổi.
- Mô hình nào có `export_params` thì xuất tham số đã học (trọng số,
  bias) ra file JSON trong thư mục outputs, để đọc và đối chiếu khi
  giảng viên hỏi công thức và model param.
- sklearn chỉ còn dùng cho tiền xử lý và metrics, không dùng cho model.
- Cụ thể từng mô hình:
  - MLP: lan truyền tiến/ngược và Adam viết tay, khởi tạo He.
  - Linear Regression: batch gradient descent viết tay trên ma trận
    thiết kế mở rộng `X_aug = [1 | X]`.
  - Decision Tree: cây CART viết tay, chia theo MSE có trọng số.
  - Random Forest: bagging viết tay trên cây CART viết tay, trung
    bình dự đoán các cây.

## Tiến độ thực hiện

- **Linear Regression (xong, 10/11/2026; cập nhật 17/11/2026):**
  `fit` học hệ số bằng batch gradient descent viết tay
  (`learning_rate=0.1`, `max_epochs=1000`, `tol=1e-6` tương đối) trên
  ma trận thiết kế `X_aug = [1 | X]`, gradient
  `(2/n) X_aug^T (X_aug beta - y)`, khởi tạo beta bằng 0. Có
  `export_params` xuất trọng số và bias ra JSON. Bản nháp ADR-007
  ngày 10/11/2026 ghi `fit` giải OLS bằng `numpy.linalg.lstsq`;
  cách đó đã được thay bằng gradient descent viết tay để thể hiện
  rõ quá trình học theo từng epoch và khớp với `history_` loss.
  Nghiệm dạng đóng chỉ còn dùng trong test để đối chiếu tính đúng
  đắn, không dùng để fit.
- **MLP (xong, Stage 02 và Stage 03):** `MLPRegressorModel` trong
  `src/house_price/models/mlp.py`: Dense(32, ReLU) -> Dense(16, ReLU)
  -> Dense(1, Linear), He init, Adam có bias-correction, mini-batch
  + shuffle, theo dõi `val_loss`, lưu JSON tham số. Gradient check:
  sai số tương đối tối đa khoảng 1,7e-08 (< 1e-5).
- **Decision Tree (xong, Stage 04):** `DecisionTreeRegressorModel`
  trong `src/house_price/models/decision_tree.py`: CART viết tay,
  ngưỡng là trung điểm giá trị riêng biệt liên tiếp, tìm kiếm bằng
  sort + cumulative sum, `x <= threshold` đi trái, lá là trung bình
  target. Fit thăm dò: `max_depth=8` hết 1,12 giây (R2 test 0,7314),
  không giới hạn hết 32,29 giây (R2 test 0,6880, overfit).
- **Random Forest (xong, Stage 04):** `RandomForestRegressorModel`
  trong `src/house_price/models/random_forest.py`: bootstrap qua
  `SeedSequence` spawn, mỗi cây một dãy con riêng, lấy mẫu đặc trưng
  theo từng lần chia khi `max_features` được đặt (mặc định `None`,
  tức xét mọi đặc trưng), dự đoán là trung bình các cây,
  `n_estimators=100` mặc định.

## Kiểm chứng (Linear Regression)

- Chạy trên VM và laptop Windows: pytest 39/39 ở thời điểm
  10/11/2026, ruff check và format sạch, không đổi logic tiền xử lý
  hay kết quả số của Stage 01.
- Test đối chiếu kết quả `fit` với nghiệm dạng đóng (closed-form).
- Chỉ số trên tập test (pipeline chuẩn 88 đặc trưng): R2 0,8154,
  RMSE 162.931, MAE 97.840 (xem `docs/progress_results.md`).

## Đánh đổi chấp nhận

- Được: thể hiện đúng cách tính của thuật toán, trả lời được câu hỏi
  của giảng viên, tham số học được xuất ra JSON để đối chiếu.
- Mất: tự viết và tự kiểm chứng nhiều code hơn; mỗi mô hình cần test
  đối chiếu với nghiệm chuẩn (như test closed-form của Linear, test
  gradient check của MLP) để chứng minh tính đúng đắn thay vì dựa
  vào thư viện.
- Rủi ro còn lại: cây CART và rừng viết tay bằng Python thuần chạy
  chậm (một cây không giới hạn độ sâu fit 32,29 giây; rừng 100 cây
  có thể mất nhiều phút), cần ghi thời gian chạy trong bảng benchmark
  của Stage 04 để giảng viên thấy rõ đánh đổi này.
