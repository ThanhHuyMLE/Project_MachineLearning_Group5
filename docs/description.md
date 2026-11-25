# Mô tả dự án

## Bài toán

Dự đoán giá bán nhà ở King County, Washington, Mỹ, từ 19 cột đầu vào
sau kỹ thuật đặc trưng (18 số + mã vùng `zipcode`, từ 18 cột gốc trong
file CSV) mô tả căn nhà (diện tích, số phòng, chất lượng xây dựng, vị trí...).
Biến mục tiêu là `price` (USD), một giá trị số liên tục.

Đây là bài toán **Hồi quy (Regression)**, không phải Phân loại (Classification).
Mô hình nhận vector đặc trưng `X` và trả về một giá trị dự đoán `y_hat`.

Dữ liệu: King County House Sales, 21.613 giao dịch từ tháng 5/2014
đến tháng 5/2015. Chi tiết đã kiểm chứng:
[king-county-dataset.md](king-county-dataset.md).

## Mục tiêu học tập

- Hiểu quy trình Học có giám sát (Supervised Learning):
  chuẩn bị dữ liệu -> huấn luyện -> dự đoán -> đánh giá.
- Hiểu và giải thích được Multilayer Perceptron (MLP):
  lan truyền tiến (Forward Propagation), lan truyền ngược
  (Backpropagation), optimizer Adam, hàm mất mát MSE.
- So sánh công bằng MLP với 3 mô hình truyền thống trên cùng tập test:
  Linear Regression, Decision Tree Regression, Random Forest Regression.
- Viết code Python theo mẫu class `__init__` -> `build_model` -> `fit` -> `predict`.
- Có mã giả (pseudocode), tài liệu giải thích thuật toán và demo dự đoán.

## Phạm vi

- Chỉ làm bài toán Hồi quy giá nhà King County, theo đề Bài tập thực hành 4.
- MLP chính thức dùng PyTorch theo yêu cầu của cô giáo cho lab 4
  (xem [adr/008-mlp-pytorch.md](adr/008-mlp-pytorch.md)); Linear
  Regression, Decision Tree và Random Forest được cài đặt bằng thư
  viện NumPy theo quy tắc của nhóm (xem
  [adr/007-mo-hinh-numpy.md](adr/007-mo-hinh-numpy.md)): không dùng
  class mô hình của scikit-learn, không dùng TensorFlow/Keras.
  NumPy chỉ được dùng cho phép tính mảng của ba mô hình truyền thống;
  MLP được triển khai bằng PyTorch. Bản PyTorch viết tay chỉ phục vụ
  học backprop và vấn đáp
  (xem [adr/009-mlp-pytorch-manual.md](adr/009-mlp-pytorch-manual.md)).
- scikit-learn chỉ dùng cho tiền xử lý (imputer, scaler, encoder, chia
  tập) và đo chỉ số (metrics).
- Không thêm web app, backend service, database, Docker hay CI/CD.

## Căn cứ kiến thức

- Đề Bài tập thực hành 4: Hồi quy MLP dự đoán giá nhà (14 bước công việc,
  kiến trúc Dense(32, relu) -> Dense(16, relu) -> Dense(1, linear),
  optimizer Adam, loss MSE, epochs 100, batch size 32).
- Tài liệu thiết kế gốc của nhóm: [DESIGN.md](DESIGN.md).
  Lưu ý: DESIGN.md dùng tập test làm validation và chỉ chia train/test 80/20.
  Thiết kế mới sửa lại thành train/validation/test, xem
  [adr/003-chia-train-validation-test.md](adr/003-chia-train-validation-test.md).
- Quy ước làm việc của dự án trước CLASSIFYING-SPAM-EMAILS:
  mẫu class 4 phương thức, quy tắc commit, tài liệu ngắn gọn có liên kết.
  Chỉ kế thừa quy ước kỹ thuật, không kế thừa nội dung phân loại email.

## Các Stage

- Stage 01: Data Understanding & Preprocessing - dữ liệu, kỹ thuật đặc
  trưng, chia tập, chuẩn hóa.
- Stage 02: MLP Architecture & Implementation - kiến trúc và cài đặt MLP.
- Stage 03: Model Training & Evaluation - huấn luyện, đánh giá, trực quan hóa.
- Stage 04: Model Benchmarking & Analysis - 3 mô hình đối chiếu và bảng
  so sánh 4 mô hình chính thức (PyTorch MLP + 3 mô hình NumPy).

## Sản phẩm

- Code dùng chung trong `src/house_price/`, notebook demo 14 bước trong
  `notebooks/practice_04_mlp_regression.ipynb`.
- Tài liệu giải thích 4 thuật toán trong `docs/algorithms/`.
- Bảng so sánh MSE, RMSE, MAE, R2 trong `outputs/` (sau khi huấn luyện).
- Hướng dẫn chạy: [README](../README.md).
