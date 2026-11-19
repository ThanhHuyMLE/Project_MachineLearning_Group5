# Mã giả thuật toán

Quy ước: chỉ số bắt đầu từ 0. Mã giả trình bày vòng lặp từng phần tử;
code Python có thể gộp một số vòng lặp thành phép tính mảng.
Mỗi mô hình tuân theo: INIT -> BUILD_MODEL -> FIT -> PREDICT.
Chi tiết toán học và giải thích đầy đủ xem `docs/algorithms/`.
Linear, Decision Tree và Random Forest được cài đặt bằng thư viện
NumPy, không dùng estimator có sẵn của scikit-learn (xem
`docs/adr/007-mo-hinh-numpy.md`); MLP chính thức dùng PyTorch (xem
`docs/adr/008-mlp-pytorch.md`), bản NumPy MLP chỉ giữ làm đối chiếu
legacy, bản PyTorch viết tay chỉ phục vụ học backprop (xem
`docs/adr/009-mlp-pytorch-manual.md`).

## Quy trình chung (4 mô hình chính thức)

```text
ĐẦU VÀO: tập dữ liệu King County, 19 cột đầu vào X (18 số + zipcode,
88 cột sau mã hóa), mục tiêu y = price
ĐẦU RA: dự đoán y_pred trên tập test, bảng MSE/RMSE/MAE/R2

1. Nạp CSV, kỹ thuật đặc trưng (house_age, was_renovated,
   sqft_per_bedroom), loại cột [Unnamed: 0, id, date] khỏi X
2. Tách X, y; chia train/validation/test 70/15/15 theo nhóm id, seed 42
3. ColumnTransformer: fit trên train (số: median + StandardScaler;
   zipcode: mode + one-hot), transform val/test
4. Với mỗi mô hình chính thức: build_model -> fit(train) -> predict(test)
   (riêng PyTorch MLP: build_model(input_size) rồi fit(train, val))
   Bản NumPy MLP chỉ chạy kèm bằng cờ `--include-numpy-mlp`.
5. Tính MSE, RMSE, MAE, R2 cho từng mô hình trên cùng tập test
6. So sánh và vẽ biểu đồ Actual vs Predicted
```

## MLP Regression (bản NumPy legacy, ngoài bảng chính thức)

Mã giả dưới đây khớp với `MLPRegressorModel` trong
`src/house_price/models/mlp.py`. Bản này chỉ chạy kèm bằng cờ
`--include-numpy-mlp` để đối chiếu với các lần chạy cũ.

```text
ĐẦU VÀO: X_train (N x 88), y_train, X_val, y_val,
         hidden (32, 16), epochs 100, batch 32,
         learning_rate 0.001, beta1 0.9, beta2 0.999, epsilon 1e-8
ĐẦU RA: y_pred; history_ (loss và val_loss từng epoch);
        tham số lưu vào outputs/mlp_parameters.json

THỦ TỤC INIT(hidden_layer_sizes, learning_rate, epochs, batch_size,
             random_state, beta1, beta2, epsilon, save_path):
    lưu cấu hình; weights_ <- RỖNG; biases_ <- RỖNG

THỦ TỤC BUILD_MODEL(input_size):
    layer_sizes <- (input_size, 32, 16, 1)
    VỚI MỖI (fan_in, fan_out) trong layer_sizes:
        W <- normal(0, sqrt(2/fan_in), size=(fan_in, fan_out))   # He init
        b <- 0
    khởi tạo các moment Adam (m, v cho W và b) bằng 0

THỦ TỤC FIT(X_train, y_train, X_val=None, y_val=None):
    NẾU model chưa build: BÁO LỖI, DỪNG
    Với mỗi epoch:
        xáo trộn train, chia mini-batch 32
        Với mỗi batch: lan truyền tiến -> tính gradient MSE
            -> lan truyền ngược -> Adam cập nhật trọng số
            (Adam có hiệu chỉnh độ lệch m_hat, v_hat)
        tính MSE trên toàn bộ train, ghi vào history_["loss"]
        NẾU có validation: chỉ forward trên validation,
            ghi MSE vào history_["val_loss"]
    tự ghi cấu hình, history, weights và biases ra JSON

HÀM PREDICT(X):
    NẾU model chưa fit: BÁO LỖI, DỪNG
    TRẢ VỀ forward propagation cho từng dòng X
```

## PyTorch MLP Regression (bản chính thức, 6 bước của cô giáo)

```text
ĐẦU VÀO: X_train (N x 88), y_train, X_val, y_val,
         hidden (32, 16), epochs 100, batch 32,
         learning_rate 0.001, random_state 42
ĐẦU RA: y_pred; history_ (loss và val_loss từng epoch);
        tham số lưu vào outputs/mlp_torch_parameters.pt (state_dict)

THỦ TỤC INIT(hidden_layer_sizes, learning_rate, epochs, batch_size,
             random_state, save_path):
    lưu cấu hình; network <- RỖNG

THỦ TỤC BUILD_MODEL(input_size):                    # bước 2
    seed NumPy/Python/torch bằng random_state
    network <- _MLPNetwork(input_size, (32, 16)):
        Linear(88, 32) -> ReLU -> Linear(32, 16) -> ReLU -> Linear(16, 1)
        mọi Linear: He init dạng chuẩn, bias 0

THỦ TỤC FIT(X_train, y_train, X_val=None, y_val=None):
    NẾU model chưa build: BÁO LỖI, DỪNG
    dataset <- TensorDataset(X float32, y float32)  # bước 1
    loader <- DataLoader(dataset, batch 32, shuffle=True,
                         generator có seed cố định)
    criterion <- nn.MSELoss()                       # bước 3
    optimizer <- torch.optim.Adam(network.parameters(), lr 0.001)
    Với mỗi epoch:                                  # bước 4
        network.train()
        Với mỗi (X_batch, y_batch) trong loader:
            optimizer.zero_grad()                   # xóa gradient cũ
            y_pred <- network(X_batch)              # forward
            loss <- criterion(y_pred, y_batch)      # MSE
            loss.backward()                         # autograd: chain rule
            optimizer.step()                        # cập nhật tham số
        tính MSE trên toàn bộ train, ghi history_["loss"]
        NẾU có validation: chỉ forward trên validation,
            ghi MSE vào history_["val_loss"]        # bước 5
    tự lưu state_dict + cấu hình + history ra .pt  # bước 6

HÀM PREDICT(X):
    NẾU model chưa fit: BÁO LỖI, DỪNG
    network.eval(); với torch.no_grad(): TRẢ VỀ network(X) dạng NumPy
```

Bản PyTorch viết tay (`MLPTorchManualModel`,
`src/house_price/models/mlp_torch_manual.py`) thay bước 4 bằng
`_forward` -> `_backward` (chain rule viết tay) -> `_adam_update`
viết tay, tham số `requires_grad=False`, không gọi `backward()`
hay dùng `torch.optim`; còn lại (DataLoader, validation, lưu/nạp)
giữ nguyên vai trò. Bản này không nằm trong benchmark chính thức.

## Linear Regression

```text
ĐẦU VÀO: X_train (N x 88), y_train,
         learning_rate 0.1, max_epochs 1000, tol 1e-6
ĐẦU RA: y_pred; coef_ (beta), intercept_ (b),
        history_ (loss từng epoch), n_iter_, converged_

THỦ TỤC INIT(learning_rate, max_epochs, tol):
    lưu cấu hình; coef_ <- RỖNG; intercept_ <- RỖNG

THỦ TỤC BUILD_MODEL():
    reset trạng thái đã học; fitted <- False

THỦ TỤC FIT(X_train, y_train):
    X_aug <- [1 | X_train]                    # thêm cột 1 cho hệ số chặn
    beta_aug <- 0                             # khởi tạo bằng 0
    losses <- []
    LẶP epoch = 1 .. max_epochs:
        residual <- X_aug beta_aug - y_train
        gradient <- (2/n) * X_aug^T residual
        beta_aug <- beta_aug - learning_rate * gradient
        loss <- mean((X_aug beta_aug - y_train)^2)
        NẾU loss không hữu hạn: BÁO LỖI phân kỳ, DỪNG
        NẾU đã có loss trước đó VÀ
           |loss_truoc - loss| / max(loss_truoc, 1e-12) < tol:
            losses.append(loss); converged_ <- True; DỪNG LẶP
        losses.append(loss)
    intercept_ <- beta_aug[0]; coef_ <- beta_aug[1:]

HÀM PREDICT(X): TRẢ VỀ X coef_ + intercept_

HÀM EXPORT_PARAMS(path, feature_names):
    ghi JSON: intercept, coefficients, learning_rate,
    n_epochs, final_loss, converged
```

## Decision Tree Regression

```text
ĐẦU VÀO: X_train, y_train, max_depth (mặc định: không giới hạn),
         min_samples_split = 2, min_samples_leaf = 1,
         max_features (mặc định: xét mọi đặc trưng),
         min_impurity_decrease = 0.0, random_state = 42
ĐẦU RA: y_pred; cây lưu trong tree_ (dict lồng nhau)

THỦ TỤC INIT(...): lưu cấu hình; tree_ <- RỖNG
THỦ TỤC BUILD_MODEL(): tạo trạng thái cây rỗng, reset trạng thái đã học

THỦ TỤC FIT(X_train, y_train):
    dùng ngăn xếp thay cho đệ quy (cây sâu không vượt giới hạn đệ quy)
    stack <- [(gốc, toàn bộ chỉ số mẫu, depth = 0)]
    TRONG KHI stack không rỗng:
        (node, indices, depth) <- lấy ra khỏi stack
        node <- lá tạm: value <- mean(y[indices]),
                mse <- mean((y[indices] - value)^2)
        NẾU số mẫu < min_samples_split: TIẾP TỤC
        NẾU max_depth đặt VÀ depth >= max_depth: TIẾP TỤC
        NẾU mse <= 0: TIẾP TỤC
        best <- BEST_SPLIT(X[indices], y[indices])
        NẾU best là None: TIẾP TỤC
        NẾU mse - weighted_mse <= min_impurity_decrease: TIẾP TỤC
        chia indices theo x <= threshold (trái) / x > threshold (phải)
        node thành nút trong: lưu feature, threshold, left, right
        đẩy (right, ...) rồi (left, ...) vào stack

HÀM BEST_SPLIT(X_node, y_node):
    VỚI MỖI đặc trưng ứng viên:
        sắp xếp giá trị (stable) và y theo thứ tự đó
        vị trí chia ứng viên: ngay sau mỗi giá trị riêng biệt
        LOẠI vị trí vi phạm min_samples_leaf ở một phía
        dùng tổng tích lũy của y và y^2 để tính SSE trái/phải
        weighted_mse <- (SSE_trái + SSE_phải) / n_node
        giữ cách chia có weighted_mse nhỏ nhất
        (hòa thì giữ đặc trưng xét trước; ngưỡng = trung điểm
        hai giá trị riêng biệt liên tiếp)
    TRẢ VỀ (feature, threshold, weighted_mse) hoặc None

HÀM PREDICT(X):
    Với mỗi x: đi từ gốc theo ngưỡng chia đến lá,
    TRẢ VỀ giá trị trung bình y của lá đó
```

## Random Forest Regression

```text
ĐẦU VÀO: X_train, y_train, n_estimators (mặc định 100),
         max_depth, min_samples_split = 2, min_samples_leaf = 1,
         max_features (mặc định: xét mọi đặc trưng),
         random_state = 42
ĐẦU RA: y_pred; rừng lưu trong trees_ (danh sách cây)

THỦ TỤC INIT(...): lưu cấu hình; trees_ <- RỖNG
THỦ TỤC BUILD_MODEL(): tạo rừng rỗng, reset trạng thái đã học

THỦ TỤC FIT(X_train, y_train):
    seed_sequence <- SeedSequence(random_state)
    child_sequences <- seed_sequence.spawn(n_estimators)
    VỚI MỖI child_sequence trong child_sequences:
        rng <- default_rng(child_sequence)
        bootstrap_indices <- rng.integers(0, n_samples, size=n_samples)
        tree_seed <- rng.integers(0, 2^31 - 1)
        tree <- DecisionTreeRegressorModel(
            max_depth, min_samples_split, min_samples_leaf,
            max_features, random_state=tree_seed)
        tree.build_model()
        tree.fit(X_train[bootstrap_indices], y_train[bootstrap_indices])
        trees_.append(tree)

HÀM PREDICT(X):
    TRẢ VỀ trung bình cộng predict(X) của tất cả các cây
```
