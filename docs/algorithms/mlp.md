# Multilayer Perceptron Regression (MLP)

## 1. Giới thiệu và mục đích

Multilayer Perceptron (MLP) là mạng nơ-ron truyền thẳng, gồm các tầng neuron nối tiếp nhau. Mô hình học quan hệ phi tuyến giữa các đặc trưng của căn nhà và giá bán.

Trong dự án, MLP được cài đặt thủ công bằng NumPy: phép tính forward, gradient backpropagation và cập nhật Adam được viết trong class, không dùng `sklearn.MLPRegressor`, TensorFlow hoặc Keras để huấn luyện.

Kiến trúc mặc định:

```
88 đặc trưng → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Linear) → giá dự đoán
```

MLP là một trong các mô hình dự đoán giá nhà của dự án; kết quả cần được đánh giá trên tập test riêng và so sánh với các mô hình còn lại bằng cùng cách chia dữ liệu, tiền xử lý và chỉ số.

## 2. Đầu vào và đầu ra

- Đầu vào: ma trận `X`có dạng (N, 88). Mỗi hàng là một căn nhà; mỗi cột là một đặc trưng sau tiền xử lý.
- Đầu ra: vector `y_hat` có dạng (N,), chứa giá nhà dự đoán bằng USD.
- Mục tiêu huấn luyện: y là cột price, giữ đơn vị USD; pipeline hiện tại không chuẩn hóa mục tiêu.
- Kiến trúc: Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Linear).
  Cụm “18 đặc trưng” chỉ số cột số trước khi mã hóa zipcode; đó không phải số chiều cuối cùng đưa vào MLP. Sau khi one-hot encoding 70 mã ZIP xuất hiện trong train, đầu vào gồm 18 cột số và 70 cột ZIP, tổng cộng 88 đặc trưng. Nhãn price không tính là feature.

## 3. Mô hình hoạt động thế nà

Mỗi neuron nhận các giá trị từ tầng trước, nhân từng giá trị với một trọng số, cộng các kết quả và bias, rồi áp dụng hàm kích hoạt. Dense Layer thực hiện phép tính đó đồng thời cho nhiều neuron.

Hai tầng ẩn dùng ReLU để mô hình học quan hệ phi tuyến. Tầng cuối dùng Linear để trả ra giá trị liên tục, không giới hạn trong khoảng 0–1. Trong lúc huấn luyện, MSE đo sai số dự đoán; Backpropagation tính gradient của sai số theo từng tham số; Adam dùng gradient để cập nhật weights và biases.

## 4. Nền tảng toán học

Ký hiệu `A⁽⁰⁾ = X`; `W⁽ˡ⁾` và `b⁽ˡ⁾` lần lượt là weights và biases ở tầng `l`. Với batch có `B` mẫu:

**Neuron và Dense Layer**

```text
z = Σⱼ wⱼxⱼ + b
a = f(z)

Z⁽ˡ⁾ = A⁽ˡ⁻¹⁾ W⁽ˡ⁾ + b⁽ˡ⁾
```

Với 88 đầu vào, shapes của weights là `(88, 32)`, `(32, 16)`, `(16, 1)`. Bias có shapes tương ứng `(1, 32)`, `(1, 16)`, `(1, 1)`.

**Forward Propagation**

```text
Z⁽¹⁾ = X W⁽¹⁾ + b⁽¹⁾       A⁽¹⁾ = ReLU(Z⁽¹⁾)
Z⁽²⁾ = A⁽¹⁾ W⁽²⁾ + b⁽²⁾   A⁽²⁾ = ReLU(Z⁽²⁾)
ŷ     = A⁽²⁾ W⁽³⁾ + b⁽³⁾   (Linear)
```

Forward Propagation đi từ input tới output để tạo dự đoán. Các giá trị trung gian `Z` và `A` được giữ lại để tính gradient ở bước ngược.

**ReLU**

```text
ReLU(z)  = max(0, z)
ReLU'(z) = 0 nếu z ≤ 0; 1 nếu z > 0
```

ReLU đưa tính phi tuyến vào hidden layers. Nếu mọi tầng chỉ dùng phép tuyến tính thì xếp nhiều tầng vẫn tương đương một phép biến đổi tuyến tính.

**Loss Function: MSE**

```
L = (1/B) Σᵢ (yᵢ - ŷᵢ)²
∂L/∂ŷ = (2/B)(ŷ - y)
```

MSE bình phương sai số nên lỗi lớn bị phạt mạnh hơn lỗi nhỏ. Do giá nhà tính bằng USD, giá trị MSE có đơn vị USD²; RMSE sau này có đơn vị USD.

**Backpropagation**

Backpropagation dùng quy tắc chuỗi để truyền đạo hàm từ output về các hidden layers:

```
δ⁽³⁾ = (2/B)(ŷ - y)
δ⁽²⁾ = (δ⁽³⁾ (W⁽³⁾)ᵀ) ⊙ ReLU'(Z⁽²⁾)
δ⁽¹⁾ = (δ⁽²⁾ (W⁽²⁾)ᵀ) ⊙ ReLU'(Z⁽¹⁾)

dW⁽ˡ⁾ = (A⁽ˡ⁻¹⁾)ᵀ δ⁽ˡ⁾
db⁽ˡ⁾ = tổng δ⁽ˡ⁾ theo các mẫu trong batch
```

Ký hiệu `⊙` là nhân từng phần tử. Gradient cho biết loss đổi thế nào khi weights hoặc biases đổi. Mỗi batch dùng các gradient này để cập nhật tham số.

**Adam Optimizer**

Với gradient `gₜ` tại lần cập nhật thứ `t`, Adam duy trì trung bình động bậc nhất `mₜ` và bậc hai `vₜ`, hiệu chỉnh độ lệch ban đầu rồi cập nhật tham số:

```
mₜ = β₁ mₜ₋₁ + (1 - β₁) gₜ
vₜ = β₂ vₜ₋₁ + (1 - β₂) gₜ²
m̂ₜ = mₜ / (1 - β₁ᵗ)
v̂ₜ = vₜ / (1 - β₂ᵗ)
θₜ = θₜ₋₁ - η m̂ₜ / (sqrt(v̂ₜ) + ε)
```

Trong đó `θ` là weights hoặc biases; `η` là learning rate. Cấu hình trong class: `learning_rate=0.001`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`. Bộ đếm `t` tăng theo mỗi lần cập nhật mini-batch.

## 5. Mã giả

```
BUILD_MODEL(input_size):
    khởi tạo weights cho input_size → 32 → 16 → 1 bằng He initialization
    khởi tạo biases bằng 0
    khởi tạo các moment Adam bằng 0
    đặt trạng thái fitted = false

FIT(X_train, y_train):
    kiểm tra model đã build, shape hợp lệ và dữ liệu hữu hạn
    với mỗi epoch:
        xáo trộn thứ tự các mẫu train
        chia dữ liệu thành mini-batch, tối đa 32 mẫu
        với mỗi batch:
            tính dự đoán bằng forward propagation
            tính gradient của MSE
            lan truyền gradient ngược qua các tầng
            cập nhật weights và biases bằng Adam
        tính MSE trên toàn bộ train và ghi vào history
    đặt fitted = true
    tự ghi cấu hình, history, weights và biases vào outputs/mlp_parameters.json

PREDICT(X):
    kiểm tra model đã fit, shape hợp lệ và số feature bằng input_size
    chạy forward propagation
    trả về vector giá dự đoán
```

Validation không được dùng để cập nhật weights trong class hiện tại. Dùng validation để theo dõi/chọn cấu hình; chỉ dùng test để báo cáo đánh giá cuối cùng.

## 6. Cài đặt Python

File: `src/house_price/models/mlp.py`, class `MLPRegressorModel`. Đây là MLP viết tay bằng NumPy; không dùng estimator `sklearn.MLPRegressor`.

| Thành phần                | Vai trò                                                                                   |
| --------------------------- | ------------------------------------------------------------------------------------------ |
| `__init__(...)`           | Lưu cấu hình tầng ẩn, learning rate, batch size, số epoch và trạng thái ban đầu |
| `build_model(input_size)` | Khởi tạo weights, biases và trạng thái Adam theo số feature                          |
| `_forward(X)`             | Tính các giá trị qua Dense, ReLU và tầng output Linear                               |
| `_backward(...)`          | Tính gradient weights, biases từ loss bằng quy tắc chuỗi                              |
| `_adam_update(...)`       | Cập nhật tham số bằng Adam                                                             |
| `fit(X_train, y_train)`   | Huấn luyện theo mini-batch, ghi loss từng epoch và tự lưu tham số JSON              |
| `predict(X)`              | Trả về giá dự đoán; yêu cầu model đã fit và đầu vào có đúng số cột      |

Cách gọi phải truyền số chiều feature khi build:

```
model = MLPRegressorModel()
model.build_model(input_size=X_train.shape[1])  # pipeline hiện tại: 88
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## 7. Dữ liệu sử dụng

Nguồn là bộ King County House Sales: [CSV dữ liệu gốc](../../data/kc_house_data_NaN/data_raw/kc_house_data_NaN.csv), gồm 21.613 dòng. Pipeline xử lý trong [preprocessing.py](../../src/house_price/preprocessing.py).

Các bước chính:

1. Tạo house_age, was_renovated và sqft_per_bedroom.
2. Bỏ id, date và Unnamed: 0 khỏi X. id chỉ dùng để chia nhóm, tránh cùng căn nhà lọt vào nhiều tập.
3. Chia train/validation/test theo nhóm id, seed 42; số dòng thực tế lần lượt là 15.129 / 3.243 / 3.241.
4. Fit imputer, StandardScaler và OneHotEncoder trên train; dùng các bộ đã fit để transform validation và test.
5. Có 18 đặc trưng số sau feature engineering và cột zipcode. Train có 70 mã ZIP được mã hóa one-hot; do đó đầu vào cuối là 18 + 70 = 88 cột.
   Dữ liệu đã xử lý được prepare_data() tạo thành các mảng trong bộ nhớ; pipeline hiện không ghi bộ dữ liệu sau xử lý thành file trong data/processed. Mục tiêu price tách riêng khỏi X và không tính trong 88 feature.

## 8. Chỉ số đánh giá

Các chỉ số dùng để đánh giá hồi quy:

- MSE: trung bình bình phương sai số; nhỏ hơn tốt hơn, đơn vị USD².
- RMSE: căn bậc hai MSE; nhỏ hơn tốt hơn, cùng đơn vị USD với giá.
- MAE: trung bình độ lớn sai số tuyệt đối; nhỏ hơn tốt hơn, đơn vị USD.
- R²: mức độ biến thiên giá được mô hình giải thích theo định nghĩa của chỉ số; lớn hơn thường tốt hơn. R² có thể âm nếu dự đoán tệ hơn baseline dự đoán trung bình.
- MAPE: sai số tuyệt đối trung bình theo phần trăm; chỉ nên diễn giải khi giá thật khác 0 và không quá gần 0.
  Dùng evaluation.py/pipeline đánh giá của dự án để tính cùng một bộ chỉ số cho các mô hình. Không chọn cấu hình theo kết quả test.

## 9. Trực quan hóa và demo

Các biểu đồ hữu ích khi đánh giá:

- Loss train theo epoch để xem quá trình tối ưu có giảm lỗi hay không.
- Loss train và validation theo epoch, nếu pipeline có ghi validation loss, để phát hiện dấu hiệu overfitting. Class hiện tại chỉ ghi loss train; muốn có đường validation phải tính và lưu riêng.
- Actual vs Predicted trên test: điểm càng gần đường y = ŷ càng tốt.
  Khi predict dữ liệu mới, phải dùng đúng thứ tự cột và cùng preprocessor đã fit trên train. Không fit lại scaler hoặc encoder riêng trên test hay dữ liệu mới.

## 10. Kết quả thực nghiệm

| Chỉ số                           | Kết quả              |
| ---------------------------------- | ---------------------- |
| Số feature                        | 88                     |
| Số mẫu train / validation / test | 15.129 / 3.243 / 3.241 |
| Train MSE epoch đầu              | 421.221.930.605,59     |
| Train MSE epoch cuối              | 23.154.158.391,79      |
| Test MSE                           | 23.679.224.804,72      |
| Test RMSE                          | 153.880,55 USD         |
| Test MAE                           | 90.391,93 USD          |
| Test R²                           | 0,8354                 |
| Test MAPE                          | 18,06%                 |

Kết quả này được chạy bằng script kiểm tra riêng sau khi gọi preprocessing và truyền đúng `input_size=88`; chưa phải kết quả chạy trọn `train_evaluate.py`, vì lời gọi `build_model()` trong entry point cần được sửa như mục 6. R²=0,8354 không có nghĩa là dự đoán đúng 83,54%; đó là chỉ số giải thích biến thiên theo định nghĩa R². Số liệu có thể đổi khi source, dữ liệu hoặc cấu hình thay đổi.

## 11. Hạn chế và thảo luận

* MLP nhạy với thang đo đầu vào; chuẩn hóa các đặc trưng số giúp việc tối ưu ổn định hơn. Bộ chuẩn hóa phải chỉ fit trên train để tránh rò rỉ dữ liệu.
* Kết quả phụ thuộc learning rate, số epoch, batch size, kích thước tầng ẩn và cách khởi tạo.
* Loss train giảm không đảm bảo mô hình tổng quát tốt; cần theo dõi validation và chỉ đánh giá cuối trên test.
* Mô hình khó giải thích từng dự đoán hơn Linear Regression.
* 21.613 giao dịch đủ để thử kiến trúc MLP nhỏ, nhưng chưa chứng minh mô hình không overfit hoặc sẽ hoạt động tốt trên khu vực/thời điểm khác.
* JSON chứa weights và biases nhưng không chứa scaler, imputer, encoder hay thứ tự feature. Muốn dùng lại model độc lập cần lưu và nạp cả preprocessor cùng metadata thứ tự cột.
* Một lần chạy và một cấu hình chưa đủ để kết luận MLP tốt hơn các mô hình đối chứng; cần so sánh cùng split, cùng preprocessing và cùng metrics.
