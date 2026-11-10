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
- Không xây engine neural network hay cây quyết định từ đầu nếu giảng viên
  không yêu cầu rõ ràng.
- Không thêm web app, backend service, database, Docker hay CI/CD trong Phase 1.
- MLP được cài đặt thủ công bằng NumPy theo yêu cầu dự án.
  Các mô hình đối chiếu có thể dùng scikit-learn.
- NumPy chỉ được dùng cho phép tính mảng; không dùng thư viện
  triển khai MLP, Backpropagation hoặc optimize

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

## Sản phẩm

- Code dùng chung trong `src/house_price/`, notebook demo 14 bước trong
  `notebooks/practice_04_mlp_regression.ipynb`.
- Tài liệu giải thích 4 thuật toán trong `docs/algorithms/`.
- Bảng so sánh MSE, RMSE, MAE, R2 trong `outputs/` (sau khi huấn luyện).
- Hướng dẫn chạy: [README](../README.md).
