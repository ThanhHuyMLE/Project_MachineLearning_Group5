# Câu hỏi vấn đáp: Bài tập thực hành 4 - Hồi quy MLP dự đoán giá nhà

File này tập hợp câu hỏi vấn đáp cho buổi thuyết trình và vấn đáp
của Bài tập thực hành 4, bao phủ cả 4 Stage: Stage 01 (dữ liệu và
tiền xử lý), Stage 02 (kiến trúc và cài đặt MLP), Stage 03 (huấn
luyện và đánh giá), Stage 04 (benchmark 4 mô hình).

## Cách dùng file này

Mỗi câu hỏi quan trọng trình bày đúng 4 phần:

1. **Đáp ngắn (10-20 giây):** câu trả lời miệng gọn nhất, đủ để
   giảng viên thấy nắm được ý chính.
2. **Giải thích sâu:** công thức, ví dụ số thật trong dự án, hoặc
   cách code thực hiện.
3. **Tham chiếu source:** tên file và tên hàm/class thật trong repo
   để mở code đối chiếu ngay khi giảng viên hỏi sâu.
4. **Hiểu lầm thường gặp:** một cách hiểu sai phổ biến và vì sao sai.

Quy ước ghi rõ trong từng câu: cái nào **đã implement trong repo**,
cái nào **chỉ là lý thuyết** (chưa implement, chỉ nhắc để thể hiện
hiểu biết). Không trình bày phần lý thuyết như thể đã có trong code.

Câu hỏi ngắn về dataset và tiền xử lý (dạng trả lời vài câu) đã có
trong [oral-questions-preprocessing.md](oral-questions-preprocessing.md);
file này đi sâu hơn và bao phủ cả 4 Stage.

---

## A. Nền tảng (Fundamentals)

### A1. Supervised learning là gì? Bài này thuộc loại nào?

1. **Đáp ngắn:** Học có giám sát là học từ dữ liệu đã có nhãn: mỗi
   dòng có đầu vào X và nhãn y đã biết; mô hình học ánh xạ từ X sang
   y trên tập train rồi dự đoán cho dữ liệu chưa thấy.
2. **Giải thích sâu:** Bài này: X là 19 cột đặc trưng căn nhà (sau
   mã hóa thành 88 cột), y là `price` đã biết. Mô hình học hàm
   `f: X -> y` bằng cách tối thiểu hóa sai số giữa dự đoán và giá
   thật trên train. Ngược lại, học không giám sát không có nhãn y
   (ví dụ phân cụm nhà theo đặc trưng mà không dùng giá).
3. **Tham chiếu source:** `src/house_price/preprocessing.py`,
   `split_features_target` tách X và y; `src/house_price/train_evaluate.py`,
   `train_and_evaluate` fit mô hình trên train rồi predict trên test.
4. **Hiểu lầm thường gặp:** "Có nhãn là phân loại." Sai: nhãn có thể
   là số liên tục (hồi quy, như giá nhà) hoặc nhãn rời rạc (phân loại).

### A2. Regression khác Classification ở điểm nào? Vì sao bài này là hồi quy?

1. **Đáp ngắn:** Regression dự đoán giá trị số liên tục; classification
   dự đoán nhãn rời rạc. Bài này dự đoán `price` (75.000-7.700.000 USD)
   nên là hồi quy.
2. **Giải thích sâu:** Vì đầu ra liên tục nên dùng loss MSE và chỉ số
   RMSE, MAE, R2. Nếu ép thành phân loại (ví dụ "đắt/rẻ") thì mất
   thông tin về mức giá cụ thể. Tầng output của MLP dùng Linear để
   trả giá trị không giới hạn, không dùng Softmax như phân loại.
3. **Tham chiếu source:** `src/house_price/models/mlp.py`,
   `MLPRegressorModel._forward` (tầng cuối Linear);
   `src/house_price/evaluation.py`, `regression_metrics`.
4. **Hiểu lầm thường gặp:** "Dùng accuracy để đánh giá hồi quy." Sai:
   accuracy đếm đúng/sai trên nhãn rời rạc, không có ý nghĩa khi đầu
   ra là số liên tục; phải dùng MSE, RMSE, MAE, R2.

### A3. Bias-variance tradeoff là gì?

1. **Đáp ngắn:** Bias là sai số do mô hình quá đơn giản (không học
   được quan hệ thật); variance là sai số do mô hình quá nhạy với
   dữ liệu train cụ thể (học cả nhiễu). Tổng lỗi kỳ vọng gồm bias²,
   variance và nhiễu không tránh được; giảm một bên thường tăng bên kia.
2. **Giải thích sâu:** Linear Regression bias cao (chỉ học tuyến tính)
   nhưng variance thấp. Cây không giới hạn độ sâu bias thấp nhưng
   variance cao: fit 32,29 giây, R2 test chỉ 0,6880, tệ hơn cây
   `max_depth=8` (R2 0,7314) vì học thuộc nhiễu train. (Cặp số test
   này chỉ minh họa overfit; việc chọn `max_depth=8` dựa trên số
   validation đã xác nhận: R2 0,7198 so với 0,6926, xem
   `scripts/compare_validation.py`.) Random Forest
   giảm variance bằng trung bình nhiều cây mà bias hầu như không tăng.
3. **Tham chiếu source:** `src/house_price/models/decision_tree.py`,
   `DecisionTreeRegressorModel.fit` (tham số `max_depth`,
   `min_samples_leaf` kiềm chế variance);
   `src/house_price/models/random_forest.py`,
   `RandomForestRegressorModel.fit`.
4. **Hiểu lầm thường gặp:** "Mô hình càng phức tạp càng tốt." Sai:
   phức tạp hơn chỉ tốt khi nó giảm bias nhiều hơn mức tăng variance;
   cây không giới hạn là phản ví dụ ngay trong dự án này.

### A4. Overfitting và underfitting nhận biết thế nào?

1. **Đáp ngắn:** Overfitting: lỗi train thấp nhưng lỗi validation/test
   cao (học thuộc train). Underfitting: lỗi train đã cao (mô hình quá
   đơn giản hoặc chưa huấn luyện đủ).
2. **Giải thích sâu:** So sánh `loss` và `val_loss` theo epoch trong
   `history_` của MLP: nếu train loss tiếp tục giảm mà val loss đi
   ngang hoặc tăng thì đang overfit. MLP 19 đặc trưng chưa scale:
   train MSE 50,8 tỷ sau 200 epoch nhưng R2 test chỉ 0,6024 là dấu
   hiệu học kém chứ không phải overfit đơn thuần; còn cây không giới
   hạn (R2 test 0,6880 < cây depth 8 là 0,7314) là overfit điển hình.
   (Cặp số test này chỉ minh họa overfit; việc chọn `max_depth=8`
   dựa trên số validation: R2 0,7198 so với 0,6926, xem
   `scripts/compare_validation.py`.)
3. **Tham chiếu source:** `src/house_price/models/mlp.py`,
   `MLPRegressorModel.fit` (ghi `history_` gồm `loss` và `val_loss`);
   `src/house_price/visualization.py`, `plot_training_history`.
4. **Hiểu lầm thường gặp:** "Train loss giảm là mô hình tốt." Sai:
   train loss giảm không đảm bảo tổng quát tốt; phải xem validation
   và chỉ đánh giá cuối trên test.

### A5. Vì sao cần 3 tập train/validation/test mà không chỉ train/test?

1. **Đáp ngắn:** Ba tập có ba vai trò khác nhau: train để học,
   validation để chọn mô hình và siêu tham số, test để đánh giá cuối
   cùng một lần duy nhất.
2. **Giải thích sâu:** Nếu dùng test để chọn mô hình thì test không
   còn độc lập, điểm đánh giá sẽ lạc quan giả vì mô hình đã được
   "chỉnh" theo test. Dự án chia 70/15/15 (15.129 / 3.243 / 3.241
   dòng). DESIGN.md gốc chỉ chia train/test 80/20 và dùng test làm
   validation; ADR-003 đã sửa điểm này.
3. **Tham chiếu source:** `src/house_price/preprocessing.py`,
   `split_train_val_test`; `docs/adr/003-chia-train-validation-test.md`.
4. **Hiểu lầm thường gặp:** "Validation và test là một, chia 2 tập là
   đủ." Sai: gộp hai vai trò vào một tập thì mọi quyết định chọn mô
   hình đều làm nhiễm phép đo cuối cùng.

### A6. Test-peeking là gì? Vì sao nguy hiểm?

1. **Đáp ngắn:** Test-peeking là dùng tập test để ra quyết định (chọn
   mô hình, chỉnh siêu tham số, chọn pipeline) rồi vẫn báo điểm trên
   chính tập test đó như kết quả độc lập.
2. **Giải thích sâu:** Thí nghiệm 19 đặc trưng trong notebook là ví dụ:
   cả 4 mô hình chạy trên 19 cột thô, rồi từ kết quả test (MLP R2
   0,6024) rút ra kết luận "MLP cần chuẩn hóa". Vì quyết định về
   pipeline đã chịu ảnh hưởng của test nên bảng đó chỉ có giá trị
   minh họa (supplementary), không dùng để xếp hạng mô hình. Kết luận
   chính thức chỉ dựa trên bảng benchmark chính 88 đặc trưng, trong
   đó mọi quyết định đã chốt trước khi đo trên test.
3. **Tham chiếu source:** `docs/progress_results.md`, mục
   "Supplementary Experiment: Preprocessing Ablation Study" (ghi rõ
   giới hạn test-peeking); `src/house_price/train_evaluate.py`,
   `train_and_evaluate` (chỉ predict test sau khi fit xong).
4. **Hiểu lầm thường gặp:** "Xem điểm test vài lần để chỉnh cho tốt
   rồi báo điểm cuối là bình thường." Sai: mỗi lần xem test để ra
   quyết định đều làm phép đo cuối kém độc lập đi; đó là rò rỉ quyết
   định (decision leakage).

---

## B. Stage 01: Data Understanding & Preprocessing

### B1. Dataset King County có gì đặc biệt? Vì sao chọn nó?

1. **Đáp ngắn:** 21.613 giao dịch nhà ở King County, Washington, từ
   5/2014 đến 5/2015, 22 cột; có giá trị khuyết thật, id bán lặp và
   phân bố giá lệch phải mạnh. Dự án dùng nó thay cho Boston Housing
   trong đề ví dụ vì file đã có sẵn trong repo và đủ lớn.
2. **Giải thích sâu:** Giá trung vị 450.000 USD, trung bình 540.088
   USD, lớn nhất 7.700.000 USD, skew 4,02, kurtosis 34,59. Tương quan
   mạnh nhất với giá: `sqft_living` 0,702, `grade` 0,667. Có 176 id
   lặp (cùng nhà bán nhiều lần) và 1.146 dòng giá trên ngưỡng IQR
   1.129.575 USD (nhà cao cấp thật, không phải lỗi).
3. **Tham chiếu source:** `docs/king-county-dataset.md` (toàn bộ số
   liệu đã kiểm chứng); `docs/adr/002-tap-du-lieu-king-county.md`.
4. **Hiểu lầm thường gặp:** "Boston Housing vẫn dùng được qua
   `sklearn.datasets.load_boston()`." Sai: API này đã bị xóa khỏi
   scikit-learn hiện đại, không cố khôi phục.

### B2. Missing value ở đâu và xử lý thế nào? Vì sao bù median?

1. **Đáp ngắn:** Chỉ `bedrooms` khuyết 13 dòng (0,06%) và `bathrooms`
   khuyết 10 dòng (0,05%). Bù bằng median của tập train trong
   preprocessor; `sqft_per_bedroom` có thêm 13 NaN do tính từ
   `bedrooms` khuyết, cũng bù median.
2. **Giải thích sâu:** Median không nhạy với outlier như mean; phân
   bố diện tích và giá lệch phải mạnh nên mean bị kéo lên bởi nhà đắt.
   Số khuyết quá nhỏ nên xóa dòng cũng được, nhưng bù median giữ
   nguyên 21.613 dòng và giữ đúng quy trình cho dữ liệu mới cũng có
   thể khuyết. Median chỉ tính trên train để tránh leakage.
3. **Tham chiếu source:** `src/house_price/preprocessing.py`,
   `build_preprocessor` (numeric: `SimpleImputer(strategy="median")`
   rồi `StandardScaler`).
4. **Hiểu lầm thường gặp:** "Bù mean hay median cũng như nhau vì
   khuyết ít." Sai về nguyên tắc: với phân bố lệch phải, mean không
   đại diện cho căn nhà điển hình; chọn median là quyết định có lý do,
   không phải ngẫu nhiên.

### B3. StandardScaler là gì và vì sao bắt buộc với MLP?

1. **Đáp ngắn:** Chuẩn hóa z-score: `z = (x - mean_train) / std_train`,
   đưa mỗi đặc trưng về trung bình 0, độ lệch chuẩn 1 trên train.
   MLP và Linear Regression nhạy với thang đo nên bắt buộc chuẩn hóa.
2. **Giải thích sâu:** `sqft_lot` lên tới 1.651.359 trong khi cờ nhị
   phân chỉ 0/1: không chuẩn hóa thì gradient bị chi phối bởi đặc
   trưng lớn, MLP hội tụ kém (R2 0,6024 trên 19 cột thô so với 0,8354 của bản NumPy trên pipeline 88 cột đã chuẩn hóa; hai số thuộc hai pipeline khác nhau nên chỉ mang tính minh họa ablation, không phải so sánh trực tiếp có kiểm soát). Cây quyết định không cần chuẩn hóa vì ngưỡng chia
   chỉ so sánh thứ tự, nhưng pipeline vẫn chuẩn hóa để dùng chung một
   tập dữ liệu cho cả 4 mô hình.
3. **Tham chiếu source:** `src/house_price/preprocessing.py`,
   `build_preprocessor` (`StandardScaler` trong numeric pipeline).
4. **Hiểu lầm thường gặp:** "Chuẩn hóa là fit trên toàn bộ dữ liệu rồi
   mới chia." Sai: mean/std là thống kê học được từ dữ liệu; tính
   chúng trên toàn tập là leakage, phải fit chỉ trên train.

### B4. Data leakage là gì? Dự án chống leakage bằng những cách nào?

1. **Đáp ngắn:** Rò rỉ dữ liệu là thông tin từ test (hoặc từ mục tiêu)
   lọt vào quá trình huấn luyện, làm điểm đánh giá tốt giả tạo.
2. **Giải thích sâu:** Ba lớp chống leakage trong dự án: (1) thứ tự
   bắt buộc: kỹ thuật đặc trưng (không fit gì) -> chia tập theo nhóm
   id -> fit preprocessor chỉ trên train -> transform val/test;
   (2) `id` không thành feature, không tạo đặc trưng nào từ `price`;
   (3) chống rò rỉ quyết định: chọn feature, loại outlier hay biến
   đổi dữ liệu không được dựa trên test (EDA phục vụ quyết định chỉ
   tính trên train).
3. **Tham chiếu source:** `src/house_price/preprocessing.py`,
   `prepare_data` (thứ tự nạp -> engineer -> tách X/y -> chia theo
   nhóm -> fit preprocessor trên train -> transform);
   `docs/king-county-dataset.md` mục 12.
4. **Hiểu lầm thường gặp:** "Chỉ cần không để test lọt vào train là
   đủ, còn EDA tính trên toàn tập thì không sao." Sai: EDA phục vụ
   quyết định tính trên toàn tập (đã gồm test) cũng là leakage dạng
   quyết định; dự án đã kiểm chứng lại tương quan và giá theo grade
   chỉ trên train (chênh lệch tối đa 0,007) để xác nhận quyết định
   không đổi.

### B5. Vì sao phải chia theo nhóm id? Chia ngẫu nhiên theo dòng thì sao?

1. **Đáp ngắn:** Có 176 id lặp (175 id 2 lần, 1 id 3 lần), liên quan
   353 dòng: cùng căn nhà bán nhiều lần với giá khác nhau (173/176
   nhóm). Chia ngẫu nhiên theo dòng để cùng căn nhà lọt vào cả train
   lẫn test, mô hình chỉ cần nhớ giá cũ là đoán gần đúng.
2. **Giải thích sâu:** Chia ngẫu nhiên theo dòng ở milestone cũ để lọt
   41 dòng test và 38 dòng validation trùng id với train, làm điểm
   đánh giá lạc quan giả. `GroupShuffleSplit` theo `id` đảm bảo mọi
   giao dịch của cùng một căn nhà nằm trong cùng một partition, nên
   test đo khả năng dự đoán căn nhà chưa thấy, không phải giao dịch
   chưa thấy của căn nhà đã thấy.
3. **Tham chiếu source:** `src/house_price/preprocessing.py`,
   `split_train_val_test` (nhánh `groups` dùng `GroupShuffleSplit`);
   `docs/adr/005-chia-theo-nhom-id.md`.
4. **Hiểu lầm thường gặp:** "176 id lặp là dữ liệu trùng, phải xóa."
   Sai: phần lớn là bán lại hợp lệ ở thời điểm khác nhau (giá khác
   nhau, cách nhau tối đa 315 ngày); dự án không xóa dòng nào.

### B6. Feature engineering đã tạo 3 đặc trưng nào? Vì sao?

1. **Đáp ngắn:** `house_age = năm bán - yr_built` (kẹp tối thiểu 0),
   `was_renovated = 1` nếu `0 < yr_renovated <= năm bán`,
   `sqft_per_bedroom = sqft_living / bedrooms`.
2. **Giải thích sâu:** `house_age` thay `yr_built` vì tuổi nhà mới là
   thứ ảnh hưởng trực tiếp đến giá; 12 dòng `yr_built` lớn hơn năm
   bán được kẹp về 0. `was_renovated` thay `yr_renovated` vì giá trị 0
   trong cột gốc trộn cờ hiệu "chưa cải tạo" với năm; 6 dòng cải tạo
   2015 trên nhà bán 2014 tính là chưa cải tạo. `sqft_per_bedroom`
   tương quan với giá 0,580, mạnh hơn nhiều đặc trưng gốc.
3. **Tham chiếu source:** `src/house_price/preprocessing.py`,
   `engineer_features` (trả bản sao, không fit gì).
4. **Hiểu lầm thường gặp:** "Càng nhiều đặc trưng tự tạo càng tốt, kể
   cả tạo từ `price`." Sai: tạo đặc trưng từ biến mục tiêu là leakage
   trực tiếp; dự án không tạo đặc trưng nào từ `price`.

### B7. Vì sao zipcode phải mã hóa one-hot mà không dùng như số?

1. **Đáp ngắn:** `zipcode` là mã vùng, không phải số liên tục: 98.100
   không "lớn hơn" 98.001 theo nghĩa giá trị. Dùng như số thì mô hình
   hiểu sai thứ tự giả tạo.
2. **Giải thích sâu:** Tương quan dạng số của `zipcode` với giá chỉ
   -0,053, gần như không có thông tin, nhưng giá trung bình theo
   zipcode dao động 234.284-2.160.606 USD nên vị trí rất quan trọng.
   One-hot thành 70 cột, mỗi mã một cột 0/1; mã chưa thấy ở train
   được bỏ qua (`handle_unknown="ignore"`, cho toàn 0). Tổng đầu vào
   thành 18 + 70 = 88 cột.
3. **Tham chiếu source:** `src/house_price/preprocessing.py`,
   `build_preprocessor` (categorical pipeline:
   `SimpleImputer(strategy="most_frequent")` rồi `OneHotEncoder`);
   `docs/adr/006-feature-engineering-va-zipcode.md`.
4. **Hiểu lầm thường gặp:** "One-hot làm tăng chiều dữ liệu nên lúc
   nào cũng xấu." Sai: 70 cột one-hot là cái giá phải trả để mô hình
   học được hiệu ứng vị trí; Linear Regression vẫn dùng được và MLP
   bản NumPy đạt R2 0,8354 trên 88 cột này.

### B8. Outlier trong dataset này có xóa không? Vì sao?

1. **Đáp ngắn:** Không xóa dòng nào. Outlier ở đây phần lớn là nhà
   cao cấp thật, không phải lỗi nhập liệu.
2. **Giải thích sâu:** Ba căn đắt nhất (7,7 triệu, 7,06 triệu, 6,89
   triệu USD) đều nhất quán với diện tích, grade và waterfront; ba
   căn rẻ nhất (75.000-80.000 USD) đều là nhà nhỏ grade 3-5. Ngưỡng
   trên theo IQR là 1.129.575 USD, có 1.146 dòng vượt (5,3%). Xóa
   chúng là xóa một phần thị trường thật và làm mô hình không dự đoán
   được nhà cao cấp hay nhà rẻ. Chỉ ghi chú dòng 33 phòng ngủ trên
   1.620 sqft là nghi ngờ nhập liệu và vẫn giữ lại.
3. **Tham chiếu source:** `docs/king-county-dataset.md` mục 7;
   `src/house_price/preprocessing.py`, `clean_dataset` (không xóa dòng).
4. **Hiểu lầm thường gặp:** "Outlier là lỗi nên phải xóa hết." Sai:
   outlier chỉ là giá trị xa phần lớn dữ liệu; phải kiểm tra tính
   nhất quán trước khi quyết định, và với loss MSE thì outlier làm
   RMSE tăng nhưng đó là tín hiệu thật của thị trường.

---

## C. Stage 02: MLP Architecture & Implementation

### C1. Perceptron và Dense layer là gì?

1. **Đáp ngắn:** Perceptron (neuron) tính tổng có trọng số của đầu
   vào cộng bias rồi áp hàm kích hoạt: `a = f(sum(w_j x_j) + b)`.
   Dense layer thực hiện phép tính đó đồng thời cho nhiều neuron bằng
   phép nhân ma trận `Z = A W + b`.
2. **Giải thích sâu:** Với 88 đầu vào, tầng ẩn đầu có 32 neuron nên
   `W` có shape `(88, 32)` và `b` có shape `(1, 32)`; toàn mạng có
   3 ma trận trọng số `(88, 32)`, `(32, 16)`, `(16, 1)`. Viết dạng
   ma trận thay vì vòng lặp từng neuron là lý do code dùng NumPy.
3. **Tham chiếu source:** `src/house_price/models/mlp.py`,
   `MLPRegressorModel._forward` (`z = current @ weight + bias`).
4. **Hiểu lầm thường gặp:** "Mỗi neuron học một đặc trưng riêng."
   Sai: mỗi neuron học một tổ hợp tuyến tính của toàn bộ đầu vào
   tầng trước, không gắn với một đặc trưng gốc cụ thể nào.

### C2. Vì sao cần hàm kích hoạt phi tuyến? Không có thì sao?

1. **Đáp ngắn:** Không có phi tuyến thì xếp bao nhiêu tầng cũng chỉ
   tương đương một phép biến đổi tuyến tính duy nhất, mô hình không
   học được quan hệ phi tuyến như giá tăng nhanh theo diện tích ở
   khu vực tốt.
2. **Giải thích sâu:** Hợp của các phép tuyến tính vẫn là tuyến tính:
   `W2(W1 x) = (W2 W1) x`. ReLU `max(0, z)` bẻ cong không gian đầu
   vào tại mỗi neuron, và nhiều neuron với nhiều ngưỡng bẻ khác nhau
   tạo ra hàm phi tuyến từng đoạn đủ linh hoạt để xấp xỉ quan hệ giá
   nhà. Đó là lý do MLP vượt Linear Regression (bản PyTorch chính
   thức R2 0,8334 so với Linear 0,8119 trên test chính thức).
3. **Tham chiếu source:** `src/house_price/models/mlp.py`,
   `MLPRegressorModel._relu` và `_relu_derivative`.
4. **Hiểu lầm thường gặp:** "Thêm tầng là đủ, hàm kích hoạt không quan
   trọng." Sai: chính hàm kích hoạt mới tạo ra phi tuyến; tầng tuyến
   tính thêm vào không thêm khả năng biểu diễn.

### C3. ReLU là gì? Ưu điểm và nhược điểm?

1. **Đáp ngắn:** `ReLU(z) = max(0, z)`; đạo hàm là 1 khi `z > 0`, 0 khi
   `z <= 0`. Ưu điểm: tính nhanh, không bão hòa ở phía dương nên
   gradient không biến mất như Sigmoid; nhược điểm: neuron có thể
   "chết" nếu `z` luôn âm.
2. **Giải thích sâu:** Đạo hàm đơn giản (0 hoặc 1) làm backprop rẻ và
   giữ gradient nguyên vẹn qua các tầng ẩn, khác với Sigmoid có đạo
   hàm tối đa 0,25 nên gradient co lại theo từng tầng. Nhược điểm
   "dying ReLU": nếu trọng số cập nhật làm `z` luôn âm thì gradient
   bằng 0 mãi và neuron không hồi phục. Dự án dùng ReLU ở cả hai tầng
   ẩn và khởi tạo He để giảm rủi ro này.
3. **Tham chiếu source:** `src/house_price/models/mlp.py`,
   `MLPRegressorModel._relu`, `_relu_derivative`, `build_model`
   (He init).
4. **Hiểu lầm thường gặp:** "ReLU không có đạo hàm tại 0 nên không
   huấn luyện được." Sai: code định nghĩa đạo hàm tại 0 bằng 0 (quy
   ước `(values > 0)`), và điểm không khả vi đơn lẻ không ảnh hưởng
   thực tế vì xác suất `z` đúng bằng 0 là rất nhỏ.

### C4. Sigmoid cho giá trị trong khoảng nào? Có bao giờ bằng 0 hay 1 không?

1. **Đáp ngắn:** Sigmoid `σ(z) = 1 / (1 + e^(-z))` cho giá trị trong
   khoảng mở (0, 1), không bao giờ đúng bằng 0 hay 1 với input hữu hạn.
2. **Giải thích sâu:** Vì `e^(-z)` luôn dương và hữu hạn khi `z` hữu
   hạn nên mẫu số `1 + e^(-z)` luôn lớn hơn 1 và hữu hạn, do đó
   `σ(z)` luôn lớn hơn 0 và nhỏ hơn 1. Chỉ khi `z -> +vô cực` thì
   Sigmoid mới tiến về 1, và `z -> -vô cực` mới tiến về 0; đó là giới
   hạn, không phải giá trị đạt được. Trong dự án, Sigmoid KHÔNG dùng
   trong mạng (tầng ẩn dùng ReLU, tầng cuối dùng Linear); nó chỉ xuất
   hiện ở đây để trả lời lý thuyết cho đúng.
3. **Tham chiếu source:** Không có trong code (chỉ lý thuyết). Mạng
   thật xem `src/house_price/models/mlp.py`,
   `MLPRegressorModel._forward` (ReLU ở tầng ẩn, Linear ở tầng cuối).
4. **Hiểu lầm thường gặp:** "Sigmoid cho ra 0 hoặc 1 nên dùng làm
   đầu ra phân loại nhị phân." Sai: Sigmoid cho ra xác suất trong
   (0, 1), không bao giờ đúng bằng 0 hay 1; muốn nhãn 0/1 phải đặt
   thêm ngưỡng (thường 0,5) sau Sigmoid.

### C5. Softmax là gì? Mỗi thành phần của Softmax có bằng 1 không?

1. **Đáp ngắn:** Softmax biến một vector điểm số thành một vector xác
   suất: `softmax(z)_i = e^(z_i) / Σ_j e^(z_j)`. Các thành phần CỘNG
   LẠI bằng 1; mỗi thành phần riêng lẻ nằm trong (0, 1) và KHÔNG nhất
   thiết bằng 1.
2. **Giải thích sâu:** Vì mẫu số là tổng của mọi `e^(z_j)` nên tổng
   các thành phần luôn đúng bằng 1 theo định nghĩa, còn mỗi thành
   phần chỉ bằng 1 khi mọi thành phần khác bằng 0, điều không xảy ra
   với input hữu hạn (vì `e^(z_j)` luôn dương). Ví dụ `z = [2, 1, 0]`
   cho Softmax khoảng `[0,665; 0,245; 0,090]`, tổng bằng 1, không
   thành phần nào bằng 1. Softmax dùng cho phân loại đa lớp; bài hồi
   quy này KHÔNG dùng Softmax (tầng cuối là Linear, một đầu ra).
3. **Tham chiếu source:** Không có trong code (chỉ lý thuyết). Tầng
   cuối thật xem `src/house_price/models/mlp.py`,
   `MLPRegressorModel._forward` (nhánh `layer_index == len(weights)-1`
   dùng Linear).
4. **Hiểu lầm thường gặp:** "Softmax cho ra vector mà mỗi phần tử
   bằng 1." Sai: chỉ TỔNG các phần tử bằng 1; mỗi phần tử là một xác
   suất trong (0, 1), và chính việc cộng lại bằng 1 mới làm nó thành
   phân phối xác suất hợp lệ.

### C6. Forward Propagation chạy thế nào trong code?

1. **Đáp ngắn:** Đưa batch đầu vào qua từng tầng: nhân ma trận trọng
   số, cộng bias, áp ReLU ở tầng ẩn, giữ nguyên ở tầng cuối (Linear)
   để ra giá dự đoán; đồng thời lưu các giá trị trung gian `Z` và `A`
   để dùng ở bước ngược.
2. **Giải thích sâu:** Với batch `B` mẫu: `Z⁽¹⁾ = X W⁽¹⁾ + b⁽¹⁾`,
   `A⁽¹⁾ = ReLU(Z⁽¹⁾)`, `Z⁽²⁾ = A⁽¹⁾ W⁽²⁾ + b⁽²⁾`,
   `A⁽²⁾ = ReLU(Z⁽²⁾)`, `ŷ = A⁽²⁾ W⁽³⁾ + b⁽³⁾`. Code lưu
   `activations` (gồm cả `X` đầu vào) và `pre_activations` (các `Z`)
   vào `cache` trả về cùng dự đoán.
3. **Tham chiếu source:** `src/house_price/models/mlp.py`,
   `MLPRegressorModel._forward`.
4. **Hiểu lầm thường gặp:** "Forward chỉ cần tính dự đoán, không cần
   lưu gì." Sai: backprop cần `A` của tầng trước và `Z` để tính đạo
   hàm ReLU; không lưu cache thì phải tính lại toàn bộ forward.

### C7. Backpropagation tính gradient thế nào?

1. **Đáp ngắn:** Dùng quy tắc chuỗi truyền đạo hàm từ output về các
   tầng trước: bắt đầu từ đạo hàm MSE theo dự đoán `(2/B)(ŷ - y)`,
   rồi nhân ngược qua từng tầng, nhân thêm đạo hàm ReLU tại mỗi tầng ẩn.
2. **Giải thích sâu:** `δ⁽³⁾ = (2/B)(ŷ - y)`,
   `δ⁽²⁾ = (δ⁽³⁾ (W⁽³⁾)ᵀ) ⊙ ReLU'(Z⁽²⁾)`,
   `δ⁽¹⁾ = (δ⁽²⁾ (W⁽²⁾)ᵀ) ⊙ ReLU'(Z⁽¹⁾)`; rồi
   `dW⁽ˡ⁾ = (A⁽ˡ⁻¹⁾)ᵀ δ⁽ˡ⁾` và `db⁽ˡ⁾` là tổng `δ⁽ˡ⁾` theo batch.
   Code lặp từ tầng cuối về tầng đầu, tính gradient cho tầng hiện tại
   trước rồi mới truyền `delta` về tầng trước.
3. **Tham chiếu source:** `src/house_price/models/mlp.py`,
   `MLPRegressorModel._backward`.
4. **Hiểu lầm thường gặp:** "Backprop là một thuật toán tối ưu riêng."
   Sai: backprop chỉ là cách tính gradient hiệu quả bằng quy tắc
   chuỗi; việc cập nhật tham số do optimizer (ở đây là Adam) đảm nhiệm.

### C8. Adam khác SGD thường ở điểm nào? Vì sao dùng Adam?

1. **Đáp ngắn:** Adam giữ trung bình động của gradient (`m`, bậc nhất)
   và của bình phương gradient (`v`, bậc hai), hiệu chỉnh độ lệch
   ban đầu rồi cập nhật mỗi tham số với bước tự điều chỉnh theo độ
   lớn gradient của chính nó.
2. **Giải thích sâu:** Công thức: `mₜ = β₁ mₜ₋₁ + (1-β₁) gₜ`,
   `vₜ = β₂ vₜ₋₁ + (1-β₂) gₜ²`, `m̂ₜ = mₜ/(1-β₁ᵗ)`,
   `v̂ₜ = vₜ/(1-β₂ᵗ)`, `θₜ = θₜ₋₁ - η m̂ₜ/(sqrt(v̂ₜ)+ε)`. Nhờ `v`,
   tham số có gradient lớn được bước nhỏ lại, tham số có gradient nhỏ
   được bước lớn lên; nhờ `m`, hướng cập nhật mượt hơn SGD thường.
   Code dùng `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`,
   `learning_rate=0.001`, bộ đếm `t` tăng theo mỗi mini-batch.
3. **Tham chiếu source:** `src/house_price/models/mlp.py`,
   `MLPRegressorModel._adam_update`.
4. **Hiểu lầm thường gặp:** "Adam có bias-correction nên không cần
   learning rate nữa." Sai: bias-correction chỉ sửa độ lệch của `m`,
   `v` lúc đầu (khi chúng khởi tạo bằng 0); learning rate `η` vẫn là
   siêu tham số quyết định bước cập nhật và vẫn phải chọn.

### C9. He initialization là gì? Vì sao dùng với ReLU?

1. **Đáp ngắn:** Khởi tạo trọng số bằng phân phối chuẩn `N(0,
   sqrt(2/fan_in))`, bias bằng 0. Hệ số `sqrt(2/fan_in)` giữ phương
   sai tín hiệu ổn định qua các tầng ReLU.
2. **Giải thích sâu:** ReLU cắt một nửa tín hiệu âm nên phương sai bị
   giảm một nửa qua mỗi tầng; nhân thêm hệ số 2 (so với Xavier dùng
   cho Sigmoid/tanh) bù đúng phần đó. Nếu khởi tạo quá nhỏ, tín hiệu
   tắt dần; quá lớn, tín hiệu bùng nổ và gradient mất ổn định. Code
   dùng `rng.normal(loc=0.0, scale=sqrt(2.0/fan_in))` với `fan_in` là
   số đầu vào của tầng.
3. **Tham chiếu source:** `src/house_price/models/mlp.py`,
   `MLPRegressorModel.build_model`.
4. **Hiểu lầm thường gặp:** "Khởi tạo trọng số bằng 0 là an toàn nhất."
   Sai: khởi tạo mọi trọng số bằng 0 làm mọi neuron trong một tầng
   nhận cùng gradient và học giống hệt nhau (đối xứng không phá vỡ
   được); phải khởi tạo ngẫu nhiên.

### C10. Epoch, batch size và learning rate ảnh hưởng thế nào?

1. **Đáp ngắn:** Epoch là một lượt qua toàn bộ train; batch size là số
   mẫu mỗi lần cập nhật; learning rate là độ lớn bước cập nhật. Dự
   án dùng epochs 100, batch 32, lr 0.001 theo đề bài.
2. **Giải thích sâu:** Batch nhỏ (32) cho gradient nhiễu hơn nhưng
   cập nhật nhiều lần mỗi epoch (15.129/32 ≈ 473 lần) và thường tổng
   quát tốt hơn batch lớn; batch quá nhỏ thì huấn luyện chậm và nhiễu
   quá mức. Learning rate quá lớn làm loss dao động hoặc phân kỳ; quá
   nhỏ thì 100 epoch không đủ hội tụ (bằng chứng: MLP 19 cột chưa
   scale sau 200 epoch vẫn train MSE 50,8 tỷ). Mỗi epoch code xáo
   trộn thứ tự mẫu rồi chia mini-batch.
3. **Tham chiếu source:** `src/house_price/models/mlp.py`,
   `MLPRegressorModel.fit` (vòng lặp epoch, `rng.permutation`, chia
   batch) và `__init__` (mặc định epochs, batch_size, learning_rate).
4. **Hiểu lầm thường gặp:** "Tăng epoch lúc nào cũng tốt hơn." Sai:
   quá nhiều epoch trên dữ liệu hữu hạn dẫn đến overfit; phải theo
   dõi val_loss và dừng khi nó không cải thiện (early stopping, xem
   câu C12).

### C11. Gradient check là gì? Kết quả trong dự án ra sao?

1. **Đáp ngắn:** Đối chiếu gradient do backprop tính với gradient số
   (finite difference: thay đổi nhỏ từng tham số rồi đo loss đổi bao
   nhiêu). Nếu hai bên khớp thì backprop đúng.
2. **Giải thích sâu:** Gradient số tính bằng `(L(θ+ε) - L(θ-ε)) / 2ε`
   với `ε` rất nhỏ; so sánh tương đối với gradient của `_backward`.
   Test trong repo cho sai số tương đối tối đa khoảng 1,7e-08, nhỏ
   hơn ngưỡng 1e-5, chứng tỏ `_backward` cài đặt đúng quy tắc chuỗi
   (kể cả hệ số `(2/B)` của đạo hàm MSE và đạo hàm ReLU). Đây là
   số của bản NumPy (float64); bản PyTorch viết tay
   (`MLPTorchManualModel`, tensor float32) có gradient check riêng
   với sai số tối đa 1,739e-03, xem câu G12.
3. **Tham chiếu source:** `tests/test_mlp_gradient.py`;
   `src/house_price/models/mlp.py`, `MLPRegressorModel._backward`.
4. **Hiểu lầm thường gặp:** "Gradient check để chạy mỗi lần huấn
   luyện." Sai: gradient check rất chậm (mỗi tham số cần 2 lần
   forward) nên chỉ dùng một lần để kiểm chứng code, không dùng trong
   huấn luyện thật.

### C12. Early stopping, Dropout và SIREN: cái nào đã có trong repo?

1. **Đáp ngắn:** Cả ba đều CHƯA implement trong repo; chúng chỉ là lý
   thuyết và hướng cải tiến, không được trình bày như tính năng đã có.
2. **Giải thích sâu:** Early stopping là dừng huấn luyện khi val_loss
   không cải thiện sau một số epoch; code hiện tại luôn chạy đủ 100
   epoch và chỉ ghi `val_loss` để theo dõi. Dropout là tắt ngẫu nhiên
   một số neuron khi huấn luyện để chống overfit; code không có
   Dropout layer nào. SIREN là mạng dùng hàm kích hoạt sin
   (`sin(ωx)`) cho tín hiệu liên tục; dự án dùng ReLU, không dùng
   SIREN. Hướng cải tiến đã ghi: bổ sung Early Stopping dựa trên
   `val_loss` và thử biến đổi `log1p(price)`.
3. **Tham chiếu source:** `src/house_price/models/mlp.py`,
   `MLPRegressorModel.fit` (không có early stopping, không có
   Dropout); `notebooks/Report.md` mục 5 (hướng cải tiến).
4. **Hiểu lầm thường gặp:** "Kể tên kỹ thuật trong báo cáo là coi như
   đã dùng." Sai: phải phân biệt rõ đã implement với chỉ đề xuất;
   giảng viên hỏi code của Dropout ở đâu thì không có code để chỉ.

---

## D. Stage 03: Model Training & Evaluation

### D1. MSE, RMSE, MAE khác nhau thế nào? Vì sao cần cả ba?

1. **Đáp ngắn:** MSE là trung bình bình phương sai số (đơn vị USD²);
   RMSE là căn bậc hai của MSE (đơn vị USD); MAE là trung bình trị
   tuyệt đối sai số (đơn vị USD). Cần cả ba vì MSE/RMSE phạt nặng lỗi
   lớn, MAE cho biết lỗi điển hình.
2. **Giải thích sâu:** Cùng một mô hình, RMSE luôn lớn hơn hoặc bằng
   MAE; khoảng cách giữa chúng cho biết có nhiều lỗi lớn hay không.
   PyTorch MLP trên test: RMSE 154.778 USD, MAE 89.839 USD, chênh
   lệch lớn phản ánh phân bố giá lệch phải và các căn đắt có sai số
   lớn (bản NumPy legacy: RMSE 153.881 USD, MAE 90.392 USD).
   Đọc RMSE một mình sẽ đánh giá quá bi quan về căn nhà điển hình;
   đọc MAE một mình sẽ bỏ qua các lỗi lớn.
3. **Tham chiếu source:** `src/house_price/evaluation.py`,
   `regression_metrics` (dùng `mean_squared_error`,
   `mean_absolute_error` của sklearn, RMSE là căn bậc hai MSE).
4. **Hiểu lầm thường gặp:** "RMSE nhỏ hơn MAE là tốt." Sai: RMSE
   không bao giờ nhỏ hơn MAE (do bất đẳng thức Jensen); nếu thấy
   RMSE < MAE thì chắc chắn tính sai.

### D2. R2 là gì? R2 = 0,8334 có nghĩa là gì?

1. **Đáp ngắn:** R2 = 1 - (tổng bình phương sai số)/(tổng bình phương
   độ lệch quanh giá trung bình). R2 = 0,8334 nghĩa là mô hình giải
   thích được 83,34% phương sai của giá quanh giá trị trung bình.
2. **Giải thích sâu:** Mẫu số là sai số của baseline "luôn đoán giá
   trung bình"; tử số là sai số của mô hình. R2 càng gần 1 càng tốt,
   R2 = 0 nghĩa là ngang baseline trung bình, R2 âm nghĩa là tệ hơn
   cả đoán trung bình. R2 = 0,8334 KHÔNG có nghĩa là dự đoán đúng
   83,34% số nhà, cũng không có nghĩa là sai số trung bình 16,66%.
   (Số này là của PyTorch MLP chính thức trên test; bản NumPy legacy
   là 0,8354.)
3. **Tham chiếu source:** `src/house_price/evaluation.py`,
   `regression_metrics` (dùng `r2_score` của sklearn).
4. **Hiểu lầm thường gặp:** "R2 là phần trăm dự đoán đúng." Sai:
   R2 là tỷ lệ phương sai được giải thích, không phải tỷ lệ mẫu dự
   đoán đúng; một mô hình có thể có R2 cao mà vẫn sai lớn ở nhà đắt.

### D3. MAPE dùng khi nào? Vì sao có thể trả None?

1. **Đáp ngắn:** MAPE là sai số tuyệt đối trung bình theo phần trăm so
   với giá thật. Chỉ có ý nghĩa khi giá thật khác 0 và không quá gần
   0; code trả `None` khi có giá trị thật bằng 0 để tránh chia cho 0.
2. **Giải thích sâu:** MAPE của PyTorch MLP trên test là 17,80%,
   nghĩa là sai số trung bình khoảng 18% giá thật (bản NumPy legacy
   là 18,06%). MAPE dễ hiểu với người không chuyên (nói phần trăm
   thay vì USD) nhưng nhạy với nhà rẻ: sai 20.000 USD trên nhà
   100.000 là 20%, trên nhà 1 triệu chỉ 2%.
   Vì giá nhà nhỏ nhất 75.000 USD (không có giá 0) nên MAPE tính
   được trong dự án này.
3. **Tham chiếu source:** `src/house_price/evaluation.py`,
   `regression_metrics` (nhánh `if np.any(y_true == 0): mape = None`).
4. **Hiểu lầm thường gặp:** "MAPE lúc nào cũng tính được và so sánh
   được giữa các bài toán." Sai: MAPE không xác định khi giá thật
   bằng 0 và bị méo khi giá thật rất nhỏ; không dùng MAPE để so sánh
   hai bài toán có thang giá khác nhau.

### D4. Đọc biểu đồ Actual vs Predicted và đường loss thế nào?

1. **Đáp ngắn:** Actual vs Predicted: điểm càng gần đường chéo
   y = ŷ càng tốt; điểm lệch xa là nhà dự đoán sai nhiều. Đường loss
   train giảm dần là bình thường; val loss đi ngang hoặc tăng trong
   khi train loss vẫn giảm là dấu hiệu overfit.
2. **Giải thích sâu:** Trên dữ liệu này, điểm ở vùng giá cao thường
   lệch nhiều hơn vì ít mẫu nhà đắt và phương sai lớn (đuôi phải dài).
   Bản NumPy MLP: train MSE từ 421 tỷ (epoch đầu) xuống 23,1 tỷ
   (epoch cuối), test MSE 23,6 tỷ, gần với train MSE cuối nên mô hình
   không overfit nặng. Nếu test MSE lớn hơn hẳn train MSE cuối thì
   mới nghi overfit.
3. **Tham chiếu source:** `src/house_price/visualization.py`,
   `plot_actual_vs_predicted`, `plot_training_history`;
   `src/house_price/train_evaluate.py`, `train_and_evaluate` (vẽ cho
   từng mô hình).
4. **Hiểu lầm thường gặp:** "Điểm nằm trên đường chéo là dự đoán hoàn
   hảo cho mọi nhà." Sai: đường chéo chỉ là tham chiếu; cần xem cả
   phân tán quanh đường chéo theo vùng giá, vì sai số ở nhà đắt lớn
   hơn nhiều so với nhà rẻ.

### D5. Vì sao không được chọn siêu tham số dựa trên điểm test?

1. **Đáp ngắn:** Vì test là phép đo cuối cùng; dùng nó để chọn thì
   phép đo không còn độc lập và điểm báo cáo sẽ lạc quan giả.
2. **Giải thích sâu:** Mỗi lần thử một cấu hình và xem điểm test là
   một lần "học" từ test. Chọn cấu hình tốt nhất trên test rồi báo
   chính điểm đó thì con số đã bị tối ưu theo test, không phản ánh
   khả năng trên dữ liệu hoàn toàn mới. Quy trình đúng: chọn trên
   validation (3.243 dòng), khóa cấu hình, đo một lần trên test
   (3.241 dòng).
3. **Tham chiếu source:** `src/house_price/train_evaluate.py`,
   `train_and_evaluate` (MLP fit với `X_val`/`y_val`, chỉ predict
   test sau khi fit xong); `docs/king-county-dataset.md` mục 12.
4. **Hiểu lầm thường gặp:** "Chỉ xem điểm test để tham khảo, không
   tính là chọn." Sai: mọi quyết định chịu ảnh hưởng của test, dù là
   "tham khảo", đều là test-peeking ở mức độ nào đó; xem câu A6.

---

## E. Stage 04: Model Benchmarking & Analysis

### E1. Linear Regression trong repo học hệ số bằng cách nào?

1. **Đáp ngắn:** Batch gradient descent được cài đặt bằng thư viện NumPy: khởi tạo beta bằng
   0, lặp tính gradient `(2/n) X_aug^T (X_aug beta - y)` với
   `X_aug = [1 | X]` rồi cập nhật `beta <- beta - lr * gradient`,
   dừng sớm khi thay đổi tương đối của loss nhỏ hơn 1e-6.
2. **Giải thích sâu:** Mặc định `lr=0.1`, `max_epochs=1000`,
   `tol=1e-6`. Dùng thay đổi tương đối (không phải tuyệt đối) vì MSE
   cỡ 1e10 nên ngưỡng tuyệt đối không có ý nghĩa chung. Nghiệm dạng
   đóng `(X^T X)^(-1) X^T y` chỉ dùng trong test để đối chiếu, không
   dùng để fit. Nếu loss không hữu hạn (phân kỳ do `X` chưa chuẩn
   hóa hoặc lr quá lớn) thì `fit` báo lỗi rõ ràng.
3. **Tham chiếu source:** `src/house_price/models/linear_regression.py`,
   `LinearRegressorModel.fit`, `export_params`.
4. **Hiểu lầm thường gặp:** "Linear Regression trong repo dùng
   `sklearn.linear_model.LinearRegression`." Sai: mô hình được cài đặt hoàn toàn bằng thư viện NumPy theo ADR-007; sklearn chỉ dùng cho tiền xử lý và metrics.

### E2. Decision Tree chọn đặc trưng và ngưỡng chia thế nào?

1. **Đáp ngắn:** Tại mỗi nút, thử từng đặc trưng ứng viên và từng
   ngưỡng (trung điểm hai giá trị riêng biệt liên tiếp), chọn cách
   chia làm MSE có trọng số của hai nút con nhỏ nhất; mẫu có
   `x <= threshold` đi trái.
2. **Giải thích sâu:** Tìm kiếm hiệu quả bằng sort + cumulative sum
   của `y` và `y²` để tính SSE mọi vị trí chia trong O(n) sau bước
   sort, thay vì tìm exhaustive O(n²). Dừng chia khi đạt `max_depth`,
   nút còn ít mẫu hơn `min_samples_split`, mọi cách chia vi phạm
   `min_samples_leaf`, hoặc mức giảm MSE không vượt
   `min_impurity_decrease`. Lá dự đoán bằng trung bình `y` trong lá.
3. **Tham chiếu source:** `src/house_price/models/decision_tree.py`,
   `DecisionTreeRegressorModel.fit`, `_best_split`,
   `_candidate_features`, `predict`.
4. **Hiểu lầm thường gặp:** "Cây dùng information gain như cây phân
   loại." Sai: cây hồi quy dùng phương sai/MSE làm tiêu chí chia,
   không dùng entropy hay Gini; lá dự đoán bằng trung bình chứ không
   phải lớp đa số.

### E3. Vì sao cây không giới hạn độ sâu lại tệ hơn cây depth 8?

1. **Đáp ngắn:** Vì overfit: cây không giới hạn học thuộc cả nhiễu
   trong train nên dự đoán test tệ hơn (R2 0,6880 so với 0,7314) mà
   còn chậm hơn nhiều (32,29 giây so với 1,12 giây). Cặp số test
   này chỉ minh họa overfit; việc chọn `max_depth=8` dựa trên số
   validation (R2 0,7198 so với 0,6926, xem
   `scripts/compare_validation.py`).
2. **Giải thích sâu:** Cây càng sâu thì mỗi lá càng ít mẫu, giá trị lá
   (trung bình `y`) càng nhiễu và càng nhạy với từng dòng train cụ
   thể. Giới hạn `max_depth=8` ép cây dừng sớm, mỗi lá có đủ mẫu để
   trung bình ổn định. Đây là ví dụ trực quan nhất của
   bias-variance tradeoff trong dự án.
3. **Tham chiếu source:** `src/house_price/models/decision_tree.py`,
   `DecisionTreeRegressorModel.fit` (điều kiện dừng `max_depth`);
   `docs/progress_results.md` (kết quả thăm dò).
4. **Hiểu lầm thường gặp:** "Cây sâu hơn thì học được nhiều hơn nên
   lúc nào cũng tốt hơn." Sai: học "nhiều hơn" bao gồm cả nhiễu; độ
   sâu phải chọn trên validation, không để mặc định không giới hạn
   khi báo cáo.

### E4. Random Forest giảm overfitting bằng cách nào?

1. **Đáp ngắn:** Bagging: mỗi cây học trên một mẫu bootstrap khác nhau
   (lấy có hoàn lại `n` dòng từ train) rồi lấy trung bình dự đoán;
   trung bình nhiều cây có phương sai thấp hơn cây đơn lẻ.
2. **Giải thích sâu:** Phương sai của trung bình `T` cây xấp xỉ
   `rho * sigma^2 + (1 - rho) * sigma^2 / T`, với `rho` là tương quan
   trung bình giữa các cây. Bootstrap làm các cây khác nhau nên giảm
   `rho`; lấy mẫu đặc trưng theo từng lần chia (khi `max_features`
   đặt) giảm `rho` thêm nữa. Khi `T` đủ lớn, số hạng thứ hai tiến về
   0 nên thêm cây chỉ cải thiện đến giới hạn rồi bão hòa.
3. **Tham chiếu source:** `src/house_price/models/random_forest.py`,
   `RandomForestRegressorModel.fit` (SeedSequence spawn, bootstrap
   qua `rng.integers`, mỗi cây một `tree_seed`), `predict` (trung
   bình các cây).
4. **Hiểu lầm thường gặp:** "Random Forest là trung bình của nhiều cây
   giống nhau nên không khác gì một cây." Sai: chính sự khác nhau
   giữa các cây (bootstrap + lấy mẫu đặc trưng) mới tạo ra hiệu quả
   giảm phương sai; cây giống hệt nhau thì trung bình bằng một cây.

### E5. Bootstrap và out-of-bag là gì?

1. **Đáp ngắn:** Bootstrap là lấy mẫu có hoàn lại `n` dòng từ tập
   train có `n` dòng để huấn luyện mỗi cây. Mỗi cây không thấy khoảng
   36,8% số dòng (xác suất `(1 - 1/n)^n` tiến về `1/e`); các dòng đó
   gọi là out-of-bag của cây.
2. **Giải thích sâu:** Vì lấy có hoàn lại nên một dòng có thể xuất
   hiện nhiều lần trong mẫu của một cây và hoàn toàn vắng mặt trong
   mẫu của cây khác. Code dùng `SeedSequence(random_state).spawn(n)`
   để mỗi cây có dãy con riêng, vừa tái lập được vừa khác nhau giữa
   các cây, không phụ thuộc RNG toàn cục. Code hiện tại không tính
   điểm out-of-bag; con số 36,8% chỉ để hiểu cơ chế.
3. **Tham chiếu source:** `src/house_price/models/random_forest.py`,
   `RandomForestRegressorModel.fit`.
4. **Hiểu lầm thường gặp:** "Out-of-bag là tập test riêng của rừng."
   Sai: OOB là các dòng không lọt vào bootstrap của một cây cụ thể;
   mỗi cây có tập OOB khác nhau, và repo không dùng OOB để đánh giá
   chính thức (đánh giá chính thức vẫn dùng tập test 3.241 dòng).

### E6. Feature importance của cây/rừng đọc thế nào?

1. **Đáp ngắn:** Độ quan trọng của một đặc trưng là tổng mức giảm MSE
   mà các lần chia trên đặc trưng đó đem lại, nhân với số mẫu trong
   nút, rồi chuẩn hóa về tổng bằng 1.
2. **Giải thích sâu:** Code cộng `n_node * improvement` vào
   `importances[feature]` mỗi khi chia, rồi chia cho tổng. Đặc trưng
   được chia nhiều và giảm MSE nhiều thì quan trọng. Lưu ý: importance
   đo đóng góp vào việc giảm MSE trên train, không phải quan hệ nhân
   quả với giá; đặc trưng tương quan cao với nhau sẽ "chia" importance
   cho nhau. Rừng hiện tại không tổng hợp sẵn importance; muốn vẽ thì
   lấy trung bình `feature_importances_` của từng cây trong `trees_`.
3. **Tham chiếu source:** `src/house_price/models/decision_tree.py`,
   `DecisionTreeRegressorModel.fit` (tính `importances`);
   `src/house_price/models/random_forest.py`,
   `RandomForestRegressorModel.trees_`.
4. **Hiểu lầm thường gặp:** "Đặc trưng quan trọng nhất là nguyên nhân
   làm giá tăng." Sai: importance chỉ đo mức độ mô hình dùng đặc
   trưng đó để giảm sai số; tương quan không phải nhân quả, và đặc
   trưng quan trọng có thể chỉ là biến thay thế cho yếu tố thật (ví
   dụ `sqft_living` thay cho "nhà lớn ở khu tốt").

### E7. Khi nào cây/rừng thắng MLP và khi nào thua?

1. **Đáp ngắn:** Trên dữ liệu bảng có thang đo lệch nhau, cây/rừng
   thắng vì không nhạy với thang đo; trên dữ liệu đã chuẩn hóa đầy đủ,
   MLP thắng nhờ học quan hệ phi tuyến mượt mà.
2. **Giải thích sâu:** Thí nghiệm 19 cột thô: Random Forest R2 0,8802,
   MLP chỉ 0,6024 vì gradient MLP hội tụ kém khi `sqft_lot` lên tới
   1,6 triệu trong khi cờ nhị phân chỉ 0/1. Trên pipeline 88 cột đã
   chuẩn hóa: PyTorch MLP R2 0,8334 vượt Linear 0,8119; bảng benchmark chính
   cho thấy rừng đứng đầu trên cùng pipeline này (R2 0,8638). Cây còn thắng
   khi dữ liệu có nhiều tương tác dạng ngưỡng và không cần ngoại suy;
   MLP thắng khi quan hệ mượt và có đủ dữ liệu.
3. **Tham chiếu source:** `docs/progress_results.md` (bảng benchmark
   chính và mục Supplementary); `src/house_price/train_evaluate.py`,
   `train_and_evaluate`.
4. **Hiểu lầm thường gặp:** "Random Forest R2 0,8802 chứng tỏ rừng tốt
   hơn MLP R2 0,8354 (bản NumPy, thí nghiệm 19 cột thô)." Sai: hai số đo trên hai pipeline khác nhau
   (19 cột thô so với 88 cột chuẩn hóa); so sánh công bằng phải cùng
   split, cùng preprocessing, cùng tập test, tức bảng benchmark chính.

### E8. So sánh công bằng 4 mô hình cần những điều kiện nào?

1. **Đáp ngắn:** Cùng cách chia dữ liệu, cùng tiền xử lý, cùng tập
   test, cùng bộ chỉ số, cùng một lần chạy pipeline; không chỉnh siêu
   tham số trên test.
2. **Giải thích sâu:** Chỉ một điều khác nhau cũng làm so sánh vô
   nghĩa: khác split thì test khác nhau; khác preprocessing thì đầu
   vào khác nhau (88 cột so với 19 cột thô đã cho ra R2 MLP bản NumPy 0,8354
   so với 0,6024); khác metric thì không so được. `train_evaluate.py`
   đảm bảo điều này bằng cách chạy cả 4 mô hình trong một lần chạy
   trên cùng `prepare_data` và cùng `regression_metrics`, rồi sắp
   bảng theo MSE tăng dần.
3. **Tham chiếu source:** `src/house_price/train_evaluate.py`,
   `train_and_evaluate`, `build_models`;
   `src/house_price/model_comparison.py`, `compare_models`.
4. **Hiểu lầm thường gặp:** "Lấy số tốt nhất của mỗi mô hình từ các
   báo cáo khác nhau rồi xếp hạng." Sai: số từ các pipeline khác nhau
   không so sánh được; phải chạy lại cùng một pipeline như bảng
   benchmark chính của Stage 04.

---

## F. Outline thuyết trình

Dàn ý trình bày (không làm slide thật; mỗi phần trình bày miệng
khoảng 2-4 phút, tổng khoảng 20-25 phút):

1. **Introduction:** Bài toán dự đoán giá nhà King County (21.613 giao
   dịch, 19 cột đầu vào, mục tiêu `price`); vì sao là hồi quy; tổng
   quan 4 Stage và 4 mô hình chính thức; kết quả chính (PyTorch MLP
   R2 0,8334 trên test, RF tốt nhất 0,8638).
2. **Algorithm:** MLP: Dense(32, ReLU) -> Dense(16, ReLU) ->
   Dense(1, Linear), forward/backprop, MSE, Adam, He init; 3 mô hình
   đối chiếu: Linear (batch gradient descent), Decision Tree (CART,
   chia theo MSE có trọng số), Random Forest (bagging + trung bình).
   Nhấn mạnh hiệu chỉnh: Sigmoid trong (0, 1) không bao giờ bằng 0/1;
   Softmax cộng lại bằng 1.
3. **Implementation:** MLP chính thức dùng PyTorch theo ADR-008
   (`MLPTorchRegressorModel`, 6 bước trong slide của cô giáo); 3 mô
   hình còn lại được cài đặt bằng thư viện NumPy theo ADR-007; bản
   NumPy MLP giữ làm đối chiếu legacy, bản PyTorch viết tay chỉ phục
   vụ học backprop (ADR-009). Mẫu chung `__init__` -> `build_model`
   -> `fit` -> `predict`; gradient check bản NumPy (sai số 1,7e-08)
   và bản viết tay (1,739e-03, float32); pipeline chống leakage
   (chia theo nhóm id, fit preprocessor chỉ trên train).
4. **Demo:** Chạy `python -m house_price.train_evaluate`, xem bảng
   metric in ra và file `outputs/metrics.csv`; dự đoán giá một căn
   nhà mới qua pipeline đã fit.
5. **Visualization:** Biểu đồ Actual vs Predicted của từng mô hình;
   đường loss train/val của MLP; biểu đồ cột so sánh RMSE/MAE/R2;
   feature importance của cây.
6. **Evaluation:** Bộ chỉ số MSE/RMSE/MAE/R2/MAPE và cách đọc (RMSE
   song song MAE vì giá lệch phải); quy tắc chỉ chọn trên validation,
   đo một lần trên test; giới hạn test-peeking của thí nghiệm bổ sung.
7. **Results:** Bảng benchmark chính Stage 04 (4 mô hình chính thức,
   pipeline 88 cột, test 3.241 dòng, ngày 19/11/2026): RF R2 0,8638,
   PyTorch MLP 0,8334, Linear 0,8119, Decision Tree 0,7314; bản NumPy
   MLP ngoài bảng (R2 validation 0,8310; số test 0,8354 là lịch sử
   17/11); thí nghiệm bổ sung 19 cột thô cho thấy vai trò của chuẩn
   hóa (MLP 0,6024 trên 19 cột thô so với 0,8354 trên pipeline 88 cột;
   hai số thuộc hai pipeline khác nhau, chỉ mang tính minh họa
   ablation); cây depth 8 thắng cây không giới hạn (0,7314 so với
   0,6880; cặp số test minh họa overfit, căn cứ chọn là số validation
   R2 0,7198 so với 0,6926).
8. **Conclusion:** Kết luận chính (chuẩn hóa quyết định hiệu năng MLP;
   PyTorch MLP vượt baseline tuyến tính; rừng là đối chiếu mạnh nhất
   trên dữ liệu bảng); hạn chế (chưa early stopping/Dropout, chưa thử
   log1p(price), rừng được cài đặt bằng thư viện NumPy nên chạy chậm);
   hướng cải tiến (Early Stopping theo val_loss, log1p(price), đồng
   nhất notebook sang 88 đặc trưng).

---

## G. PyTorch và 3 bản MLP (mục 10-11 spec Lab 4)

Mục này trả lời các khái niệm PyTorch trong spec lab 4. Bản chính
thức là PyTorch (`MLPTorchRegressorModel`); bản NumPy và bản PyTorch
viết tay chỉ dùng để đối chiếu và học cơ chế (xem câu G12).

### G1. Tensor khác NumPy array ở điểm nào? dtype và device là gì?

1. **Đáp ngắn:** Tensor là mảng của PyTorch, giống NumPy array nhưng
   ghi được đồ thị tính toán để autograd tự tính đạo hàm, và chạy
   được trên GPU. `dtype` là kiểu số của phần tử (ở đây float32);
   `device` là nơi tensor sống (CPU hay CUDA).
2. **Giải thích sâu:** NumPy array không có khái niệm gradient:
   muốn đạo hàm phải tự viết chain rule như bản NumPy của dự án.
   Tensor có thêm thuộc tính `requires_grad` và `grad_fn`: khi
   `requires_grad=True`, mọi phép tính trên tensor được ghi lại
   thành đồ thị để `backward()` suy ra đạo hàm. Dự án chuyển mảng
   NumPy float64 sang tensor float32 bằng `torch.from_numpy(X.astype
   (np.float32))` vì torch mặc định tính float32; float32 nhanh và
   nhẹ hơn float64 nhưng làm tròn nhiều hơn (xem gradient check
   1,739e-03 của bản viết tay so với 1,7e-08 của bản NumPy).
   Device: code trong repo không chỉ định device nên tensor ở CPU
   mặc định; muốn chạy GPU phải `.to("cuda")` cả model lẫn dữ liệu,
   còn không thì mọi thứ vẫn ở CPU.
3. **Tham chiếu source:** `src/house_price/models/mlp_torch.py`,
   `MLPTorchRegressorModel.fit` (chuyển `X`/`y` sang float32);
   `src/house_price/models/mlp_torch_manual.py`, `build_model`
   (`torch.randn(..., dtype=torch.float32)`, `requires_grad_(False)`).
4. **Hiểu lầm thường gặp:** "Tensor chỉ là NumPy array đổi tên."
   Sai: tensor mang thêm đồ thị tính toán và device; chính hai thứ
   đó mới làm autograd và GPU khả thi.

### G2. TensorDataset và DataLoader làm gì? batch_size và shuffle?

1. **Đáp ngắn:** `TensorDataset` gói cặp tensor (X, y) thành một
   dataset có thể lập chỉ mục theo mẫu; `DataLoader` chia dataset
   thành mini-batch và đưa từng batch cho vòng lặp huấn luyện.
   `batch_size=32` là số mẫu mỗi lần cập nhật; `shuffle=True` xáo
   trộn thứ tự mẫu mỗi epoch.
2. **Giải thích sâu:** Không dùng DataLoader thì phải tự xáo trộn
   bằng `rng.permutation` và cắt batch tay như bản NumPy. DataLoader
   nhận `generator` có seed cố định (`torch.Generator().manual_seed
   (random_state)`) nên thứ tự xáo trộn lặp lại được với cùng seed,
   và `shuffle=True` mới xáo mỗi epoch; để `shuffle=False` thì các
   batch giữ nguyên thứ tự gốc, mô hình có thể học theo thứ tự dữ
   liệu (xấu). Batch 32 trên 15.129 mẫu train cho khoảng 473 lần
   cập nhật mỗi epoch.
3. **Tham chiếu source:** `src/house_price/models/mlp_torch.py`,
   `MLPTorchRegressorModel.fit` (bước 1: `TensorDataset`,
   `DataLoader`); bản manual dùng đúng hai class này để nạp batch.
4. **Hiểu lầm thường gặp:** "DataLoader là phần huấn luyện nên phải
   viết tay mới tính là manual." Sai: DataLoader chỉ nạp dữ liệu,
   không tính gradient; bản manual vẫn dùng DataLoader mà vẫn gọi
   là viết tay vì phần học (forward/backward/Adam) viết tay.

### G3. nn.Module, __init__, forward, nn.Linear, nn.ReLU là gì?

1. **Đáp ngắn:** `nn.Module` là lớp cơ sở của mọi mạng PyTorch:
   `__init__` khai báo các tầng con, `forward` định nghĩa đường đi
   của dữ liệu qua mạng. `nn.Linear` là tầng Dense (nhân ma trận +
   bias, có tham số học được); `nn.ReLU` là hàm kích hoạt không có
   tham số.
2. **Giải thích sâu:** `_MLPNetwork` kế thừa `nn.Module`, trong
   `__init__` tạo các `nn.Linear` và `nn.ReLU` rồi gói bằng
   `nn.Sequential`; `forward` chỉ gọi `self.net(x)`. Khi gọi
   `network(X_batch)`, Python thực ra gọi `__call__`, nó chạy
   `forward` và ghi đồ thị autograd. `nn.Linear(fan_in, fan_out)`
   giữ `weight` shape `(fan_out, fan_in)` và `bias` shape
   `(fan_out,)`; ngược với bản NumPy giữ `W` shape
   `(fan_in, fan_out)` nên phép nhân là `x @ W.T` bên trong Linear.
   `nn.ReLU()` không có trọng số nên không xuất hiện trong
   `state_dict`.
3. **Tham chiếu source:** `src/house_price/models/mlp_torch.py`,
   `_MLPNetwork.__init__` và `forward`.
4. **Hiểu lầm thường gặp:** "Phải tự gọi `forward` trong training
   loop." Sai: gọi `network(X_batch)` là đủ; gọi trực tiếp
   `forward` sẽ bỏ qua các hook của `nn.Module` và không ghi đồ
   thị đúng cách.

### G4. Vì sao tầng output dùng Linear cho hồi quy?

1. **Đáp ngắn:** Hồi quy cần đầu ra là số liên tục không giới hạn
   (giá nhà 75.000-7.700.000 USD); Linear `f(z) = z` trả đúng giá
   trị đó, còn ReLU cắt giá trị âm và Sigmoid/Softmax gò đầu ra
   vào một khoảng.
2. **Giải thích sâu:** Tầng cuối `nn.Linear(16, 1)` không kèm ReLU
   nên dự đoán có thể là bất kỳ số thực nào, khớp với `price` là
   số liên tục. Nếu dùng ReLU ở tầng cuối thì mọi dự đoán đều lớn
   hơn hoặc bằng 0 và gradient bằng 0 với mọi dự đoán âm, làm mất
   khả năng sửa lỗi khi mô hình đoán thấp. Sigmoid chỉ hợp với
   phân loại nhị phân (xác suất trong (0, 1)), Softmax chỉ hợp với
   phân loại đa lớp (tổng bằng 1); cả hai đều sai bản chất bài
   hồi quy.
3. **Tham chiếu source:** `src/house_price/models/mlp_torch.py`,
   `_MLPNetwork.__init__` (chỉ thêm `nn.ReLU()` khi `fan_out != 1`);
   `src/house_price/models/mlp.py`, `_forward` (nhánh tầng cuối
   dùng Linear).
4. **Hiểu lầm thường gặp:** "Thêm ReLU ở tầng cuối cho dự đoán
   không âm thì tốt hơn." Sai: giá nhà luôn dương là tính chất của
   dữ liệu, không phải ràng buộc của hàm mất mát; ép ReLU ở output
   làm méo gradient và không giúp gì khi mô hình đã học được giá
   dương từ dữ liệu.

### G5. Forward pass chạy thế nào trong bản PyTorch?

1. **Đáp ngắn:** Đưa batch tensor qua từng tầng: Linear nhân ma
   trận cộng bias, ReLU cắt âm ở hai tầng ẩn, Linear cuối trả một
   số cho mỗi mẫu; shape đi từ `(32, 88)` thành `(32, 1)`.
2. **Giải thích sâu:** Với batch 32 mẫu: `(32, 88)` -> Linear ->
   `(32, 32)` -> ReLU -> Linear -> `(32, 16)` -> ReLU -> Linear ->
   `(32, 1)`. Mỗi lần forward, autograd ghi lại phép tính của từng
   tầng vào đồ thị để `backward()` dùng sau. Bản PyTorch không lưu
   cache `Z`/`A` như bản NumPy vì đồ thị autograd đã giữ các giá
   trị trung gian; bản viết tay phải tự lưu cache vì không có đồ thị.
3. **Tham chiếu source:** `src/house_price/models/mlp_torch.py`,
   `_MLPNetwork.forward`; `src/house_price/models/mlp_torch_manual.py`,
   `_forward` (trả `(dự đoán, (activations, pre_activations))`).
4. **Hiểu lầm thường gặp:** "Forward của PyTorch và NumPy khác
   toán." Sai: cùng một phép toán, cùng kiến trúc; chỉ khác ai giữ
   các giá trị trung gian (autograd giữ hộ, NumPy tự giữ trong cache).

### G6. MSE là gì và vì sao dùng cho hồi quy?

1. **Đáp ngắn:** MSE là trung bình bình phương sai số
   `L = (1/B) Σ (y - ŷ)²`; dùng cho hồi quy vì nó đo khoảng cách
   giữa dự đoán và giá thật, khả vi khắp nơi nên tính được gradient.
2. **Giải thích sâu:** `nn.MSELoss()` mặc định tính trung bình trên
   cả batch và mọi đầu ra, trả một số vô hướng để `backward()` tính
   được. Đạo hàm theo dự đoán là `(2/B)(ŷ - y)`, chính là điểm bắt
   đầu của backprop ở cả ba bản MLP. MSE phạt lỗi lớn mạnh hơn lỗi
   nhỏ (bình phương), phù hợp khi nhà đắt đoán sai nhiều là nghiêm
   trọng. Nhược điểm: đơn vị USD² nên phải lấy căn (RMSE) mới đọc
   được bằng USD; và MSE nhạy với outlier (nhà 7,7 triệu USD).
3. **Tham chiếu source:** `src/house_price/models/mlp_torch.py`,
   `fit` (`criterion = nn.MSELoss()`); bản manual dùng hàm
   `mse_loss` viết tay (`torch.mean(torch.square(y_pred - y_true))`).
4. **Hiểu lầm thường gặp:** "MSE và RMSE khác nhau về bản chất nên
   chọn một trong hai làm loss." Sai: RMSE là căn bậc hai của MSE,
   tối thiểu hóa MSE cũng tối thiểu hóa RMSE; chỉ khác đơn vị đọc.

### G7. Optimizer là gì? Gradient descent khác Adam ở điểm nào?

1. **Đáp ngắn:** Optimizer là đối tượng cập nhật tham số từ
   gradient: gradient descent (GD) trừ thẳng `lr * gradient`;
   Adam chia gradient đã làm mượt cho độ lớn gradient của từng
   tham số nên bước cập nhật tự điều chỉnh theo từng tham số.
2. **Giải thích sâu:** GD thường: `θ <- θ - η g`. Adam giữ hai
   trung bình động `m` (gradient) và `v` (bình phương gradient),
   hiệu chỉnh độ lệch ban đầu rồi cập nhật
   `θ <- θ - η m̂/(sqrt(v̂)+ε)`. Nhờ `v`, tham số có gradient lớn
   được bước nhỏ lại và ngược lại; nhờ `m`, hướng cập nhật mượt
   hơn. Dự án dùng `torch.optim.Adam(network.parameters(),
   lr=0.001)` ở bản chính thức; hai bản viết tay dùng cùng công
   thức với `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8` trong
   `_adam_update`. Adam không tính gradient, nó chỉ tiêu thụ
   gradient mà backprop/autograd đã tính.
3. **Tham chiếu source:** `src/house_price/models/mlp_torch.py`,
   `fit` (bước 3: `torch.optim.Adam`); `src/house_price/models/mlp.py`
   và `mlp_torch_manual.py`, `_adam_update`.
4. **Hiểu lầm thường gặp:** "Adam là một dạng backprop." Sai:
   backprop tính gradient, Adam dùng gradient để cập nhật tham số;
   hai việc khác nhau, có thể thay Adam bằng SGD mà không đổi cách
   tính gradient.

### G8. Vì sao phải zero_grad? Gradient tích lũy là gì?

1. **Đáp ngắn:** PyTorch cộng dồn gradient vào `.grad` của tham số
   sau mỗi `backward()` chứ không ghi đè; không gọi
   `optimizer.zero_grad()` thì gradient của batch sau cộng chồng
   lên gradient batch trước, bước cập nhật sai.
2. **Giải thích sâu:** `.grad` bắt đầu bằng `None`, sau lần
   `backward()` đầu tiên thành tensor gradient; lần `backward()`
   sau cộng thêm vào tensor đó. Hành vi cộng dồn này có ích khi
   muốn cộng gradient nhiều batch nhỏ thành một batch lớn
   (gradient accumulation), nhưng trong vòng lặp thường thì mỗi
   batch phải bắt đầu từ gradient bằng 0. Vì vậy thứ tự đúng là
   `zero_grad()` -> forward -> loss -> `backward()` -> `step()`,
   và `zero_grad()` phải ở đầu vòng lặp batch, không phải cuối
   epoch. Bản NumPy và bản viết tay không gặp vấn đề này vì
   `_backward` trả gradient mới mỗi lần gọi, không cộng dồn vào
   biến trạng thái.
3. **Tham chiếu source:** `src/house_price/models/mlp_torch.py`,
   `fit` (vòng lặp batch: `optimizer.zero_grad()` là dòng đầu
   tiên).
4. **Hiểu lầm thường gặp:** "`zero_grad` xóa trọng số đã học." Sai:
   nó chỉ đặt gradient về 0; trọng số chỉ đổi ở `optimizer.step()`.

### G9. loss.backward(), autograd và chain rule; optimizer.step() là gì?

1. **Đáp ngắn:** `loss.backward()` chạy autograd: nó duyệt đồ thị
   tính toán đã ghi lúc forward và áp chain rule để tính gradient
   của loss theo từng tham số. `optimizer.step()` sau đó mới cập
   nhật tham số bằng gradient đó; `step()` không phải backprop.
2. **Giải thích sâu:** Autograd không phải một công thức đạo hàm
   viết sẵn cho MLP: nó là cơ chế tự động vi phân - mỗi phép tính
   tensor ghi lại hàm ngược của nó, và `backward()` nhân các đạo
   hàm riêng theo chain rule từ loss về các tầng, đúng công thức
   `δ⁽³⁾ = (2/B)(ŷ - y)` rồi `δ⁽²⁾`, `δ⁽¹⁾` như câu C7. Bản viết
   tay làm đúng việc này bằng tay trong `_backward` (tự nhân
   `delta @ W.T` và nhân đạo hàm ReLU), nên gọi nó là "backprop
   viết tay"; bản chính thức gọi `loss.backward()` thì autograd làm
   thay. `optimizer.step()` chỉ đọc `.grad` và áp công thức Adam;
   nếu quên gọi `backward()` thì `.grad` là `None` và `step()` không
   cập nhật gì; nếu gọi `step()` hai lần liên tiếp mà không
   `backward()` lại thì lần hai dùng lại gradient cũ.
3. **Tham chiếu source:** `src/house_price/models/mlp_torch.py`,
   `fit` (bước 4: `zero_grad` -> forward -> loss -> `backward` ->
   `step`); `src/house_price/models/mlp_torch_manual.py`,
   `_backward` và `_adam_update` (làm tay hai việc autograd và
   optimizer làm hộ).
4. **Hiểu lầm thường gặp:** "Gọi `optimizer.step()` là đã backprop."
   Sai: `step()` không tính gradient; không có `backward()` trước
   đó thì không có gì để cập nhật. Và "autograd là backprop viết
   tay" cũng sai: autograd tự xây đồ thị và tự suy đạo hàm, còn
   viết tay là tự viết từng đạo hàm theo chain rule.

### G10. train() vs eval(); no_grad() và inference_mode() khác nhau thế nào?

1. **Đáp ngắn:** `train()` đặt mạng ở chế độ huấn luyện,
   `eval()` đặt ở chế độ suy luận; khác biệt chỉ có ý nghĩa khi
   mạng có tầng hoạt động khác nhau giữa hai chế độ (Dropout,
   BatchNorm). `no_grad()` tắt việc ghi đồ thị autograd trong khối
   lệnh; `inference_mode()` cũng tắt ghi đồ thị, nhanh hơn và an
   toàn hơn cho suy luận thuần túy.
2. **Giải thích sâu:** Mạng trong dự án chỉ có Linear và ReLU nên
   `train()`/`eval()` không đổi kết quả số; code vẫn gọi đúng vì
   đó là thói quen bắt buộc khi sau này thêm Dropout/BatchNorm:
   `fit` gọi `self.network.train()` đầu mỗi epoch, `predict` và
   `_mse` gọi `eval()`. `no_grad()` dùng khi tính toán không cần
   gradient (đánh giá, dự đoán, cập nhật tay trong `_adam_update`):
   nó tắt ghi đồ thị nên tiết kiệm bộ nhớ và nhanh hơn, nhưng
   tensor tạo trong `no_grad()` vẫn dùng được bình thường sau đó.
   `inference_mode()` mạnh hơn: ngoài tắt ghi đồ thị, nó cấm sửa
   tensor đã dùng trong suy luận, nên nhanh hơn nữa nhưng không
   dùng được khi còn cần cập nhật tham số; dự án dùng `no_grad()`
   (trong `_mse`, `predict`, `_adam_update`) và chưa dùng
   `inference_mode()` ở đâu.
3. **Tham chiếu source:** `src/house_price/models/mlp_torch.py`,
   `fit` (`train()`), `_mse` và `predict` (`@torch.no_grad()` +
   `eval()`); `src/house_price/models/mlp_torch_manual.py`,
   `_adam_update` và `_mse` (`@torch.no_grad()`).
4. **Hiểu lầm thường gặp:** "Quên `eval()` thì dự đoán sai."
   Sai với mạng này (không có Dropout/BatchNorm nên không đổi);
   nhưng quên `no_grad()` khi predict thì autograd vẫn ghi đồ thị,
   tốn bộ nhớ vô ích và có thể giữ tham chiếu làm rò rỉ bộ nhớ.

### G11. state_dict, torch.save, torch.load, load_state_dict, map_location; CPU vs CUDA; precision và reproducibility

1. **Đáp ngắn:** `state_dict` là dict tên tầng -> tensor tham số
   đã học; `torch.save` ghi dict đó (kèm cấu hình, history) ra
   file `.pt`; `torch.load` đọc lại; `load_state_dict` đổ tham số
   vào một mạng mới cùng kiến trúc; `map_location="cpu"` ép tensor
   về CPU khi nạp, kể cả file lưu trên GPU.
2. **Giải thích sâu:** `state_dict` chỉ chứa tham số học được
   (weight/bias của các `nn.Linear`); nó không chứa kiến trúc, nên
   `load` phải `build_model` lại mạng rồi mới
   `load_state_dict(checkpoint["state_dict"])`, và phải kiểm tra
   shape khớp (test `test_torch_mlp_load_rejects_wrong_architecture`
   bắt lỗi khi kiến trúc khác). `torch.load` mặc định nạp tensor
   về đúng device đã lưu; file lưu trên máy có GPU mà nạp trên máy
   CPU sẽ lỗi nếu thiếu `map_location="cpu"`, nên code luôn truyền
   `map_location="cpu"`. Về precision: dự án tính float32 (tensor
   torch mặc định), khác bản NumPy float64, nên số có thể lệch nhỏ
   (gradient check 1,739e-03 so với 1,7e-08). Về reproducibility:
   `build_model` seed cả `random`, NumPy và torch
   (`torch.manual_seed`), DataLoader dùng generator có seed, nên
   cùng seed 42 cho cùng kết quả; nhưng chạy trên GPU khác hoặc
   đổi thứ tự phép tính float có thể làm số lệch ở chữ số cuối.
   Lưu ý an toàn: `load` dùng `weights_only=True` nên chỉ nạp
   tensor và kiểu dữ liệu nguyên thủy, không thực thi code lạ trong
   file checkpoint.
3. **Tham chiếu source:** `src/house_price/models/mlp_torch.py`,
   `save`/`load`; `src/house_price/models/mlp_torch_manual.py`,
   `save`/`load` (lưu trực tiếp `weights`/`biases`, không dùng
   `state_dict` vì không có `nn.Module`).
4. **Hiểu lầm thường gặp:** "File `.pt` chứa cả mô hình lẫn kiến
   trúc nên nạp là chạy." Sai: file chỉ chứa tham số (và metadata
   do code tự ghi); không có code class thì không tái tạo được
   mạng. Và "lưu model là lưu cả preprocessor" cũng sai: checkpoint
   không chứa scaler/imputer/encoder hay thứ tự cột.

### G12. Cái nào mới là "manual" thật, PyTorch cung cấp cái gì? 3 bản MLP khác nhau thế nào? Vì sao tách validation/test? Benchmark lịch sử có ý nghĩa gì?

1. **Đáp ngắn:** "Manual" thật là tự viết công thức đạo hàm
   (chain rule) và bước cập nhật Adam; PyTorch cung cấp tensor,
   autograd, optimizer có sẵn, DataLoader và `state_dict`.
   Ba bản MLP cùng kiến trúc, khác nhau ở ai tính gradient và ai
   cập nhật tham số. Validation để chọn, test để đo cuối; benchmark
   lịch sử chỉ để đối chiếu, không thay bảng chính thức.
2. **Giải thích sâu:** Bảng phân biệt đầy đủ xem
   `docs/algorithms/mlp.md` mục 6. Tóm tắt: bản PyTorch chính thức
   dùng autograd + `torch.optim.Adam` (PyTorch làm cả hai việc);
   bản NumPy viết tay cả backprop lẫn Adam bằng mảng NumPy; bản
   PyTorch viết tay dùng tensor torch nhưng viết tay cả backprop
   lẫn Adam, `requires_grad=False`, và test quét source cấm
   `backward()`/`autograd.grad`/`torch.optim` trong training path.
   DataLoader không tính là "không manual" vì nó chỉ nạp dữ liệu.
   Tách validation/test: mọi quyết định chọn mô hình và siêu tham
   số chỉ dựa trên validation (3.243 dòng); test (3.241 dòng) chỉ
   đo một lần cuối, vì dùng test để chọn thì điểm báo cáo lạc quan
   giả (xem câu A5, A6). Benchmark lịch sử 17/11 (4 mô hình NumPy:
   RF 0,8638, NumPy MLP 0,8354, Linear 0,8119, DT 0,7314) giữ lại
   để chứng minh pipeline không đổi khi thêm PyTorch: 3 mô hình
   NumPy cũ tái lập đúng số (lệch 0) trong bảng chính thức
   19/11/2026, còn số test của bản NumPy MLP chỉ còn là lịch sử vì
   bản đó đã ra khỏi bảng chính thức.
3. **Tham chiếu source:** `docs/adr/008-mlp-pytorch.md`,
   `docs/adr/009-mlp-pytorch-manual.md`,
   `docs/progress_results.md` (bảng chính thức 19/11 và bảng lịch
   sử 17/11), `src/house_price/train_evaluate.py`, `build_models`.
4. **Hiểu lầm thường gặp:** "Dùng `nn.Linear` và DataLoader thì
   không còn gì là manual." Sai: manual nằm ở gradient và optimizer,
   không nằm ở tầng hay bộ nạp dữ liệu. Và "số lịch sử bằng số mới
   nên bảng cũ vẫn là chính thức" cũng sai: bảng chính thức là
   bảng 19/11 với 4 mô hình; bảng 17/11 chỉ còn giá trị đối chiếu.

### G13. Vì sao bản manual đặt requires_grad=False?

1. **Đáp ngắn:** Để chứng minh training path không đi qua autograd:
   tham số không ghi đồ thị tính toán nên không có gradient tự động
   nào được tạo ra; mọi gradient đều do `_backward` tính tay.
2. **Giải thích sâu:** Khi `requires_grad=False`, mọi phép tính trên
   tensor tham số không tạo `grad_fn`, nên loss tính từ chúng cũng
   không có đồ thị; gọi `loss.backward()` trên loss đó sẽ lỗi vì
   không có gì để vi phân. Nhờ đó test
   `test_manual_training_path_uses_no_autograd` có thể quét source
   và khóa việc gọi `backward()`, `autograd.grad` hay `torch.optim`
   trong training path: nếu code vô tình gọi autograd thì test fail
   ngay. Riêng `@torch.no_grad()` trong `_adam_update` chỉ tạm tắt
   việc ghi đồ thị trong khối cập nhật, còn `requires_grad=False`
   trên tham số mới là thứ bảo đảm ngay từ forward đã không có đồ
   thị nào tồn tại. Ngoài ra, không ghi đồ thị còn tiết kiệm bộ nhớ
   trong lúc huấn luyện. Lưu ý: `requires_grad=False` không làm mất
   khả năng tính toán, vì forward/backward tay chỉ cần phép tính số
   học trên tensor, không cần đồ thị.
3. **Tham chiếu source:** `src/house_price/models/mlp_torch_manual.py`,
   `build_model` (`requires_grad_(False)`), `_adam_update`
   (`@torch.no_grad()`); `tests/test_mlp_torch_manual.py`
   (`test_manual_training_path_uses_no_autograd`,
   `test_manual_parameters_shape_dtype_device_and_no_grad`).
4. **Hiểu lầm thường gặp:** "`requires_grad=False` chỉ là tối ưu
   tốc độ." Sai: ở đây nó là một phần của thiết kế chứng minh, tắt
   autograd ngay từ tham số chứ không chỉ tiết kiệm bộ nhớ. Và
   "đặt `requires_grad=False` thì tensor không học được" cũng sai:
   tensor vẫn cập nhật được bằng phép gán tay trong `_adam_update`.

### G14. Vì sao validation không được cập nhật weights?

1. **Đáp ngắn:** Vì code chỉ dùng validation để đo, không để học:
   validation chỉ đi qua `_mse` (forward thuần túy, không backward,
   không bước cập nhật) nên weights không đổi sau khi đo validation.
2. **Giải thích sâu:** Trong `fit` của cả hai bản PyTorch, sau mỗi
   epoch code gọi `_mse(X_val, y_val)`; `_mse` được trang trí
   `@torch.no_grad()` (bản chính thức gọi thêm `eval()`, bản manual
   không có `nn.Module` nên không có `eval()`), chỉ forward rồi trả
   về một số float; không có `loss.backward()` hay
   `optimizer.step()`/`_adam_update` nào nhận dữ liệu validation.
   Điều này đúng cả khi `fit` không nhận validation (lúc đó
   `history_` chỉ có `loss`). Về phương pháp: validation dùng để
   theo dõi và chọn cấu hình; nếu cập nhật weights trên validation
   thì validation trở thành một phần của train, điểm validation sẽ
   lạc quan giả và mất vai trò chọn mô hình; test là tập duy nhất
   chỉ đo một lần cuối (xem câu A5, A6 và G12). Lưu ý: Linear,
   Decision Tree và Random Forest trong `train_and_evaluate` chỉ
   nhận `X_train`/`y_train` nên validation càng không lấn vào quá
   trình fit của ba mô hình này.
3. **Tham chiếu source:** `src/house_price/models/mlp_torch.py`,
   `fit` (bước 5) và `_mse`; `src/house_price/models/mlp_torch_manual.py`,
   `fit` và `_mse`; `src/house_price/train_evaluate.py`,
   `train_and_evaluate` (docstring: Linear/DT/RF chỉ nhận
   `X_train`/`y_train`).
4. **Hiểu lầm thường gặp:** "Tính `val_loss` mỗi epoch là đã học
   trên validation." Sai: đo loss khác cập nhật weights; chỉ
   backward cộng bước cập nhật mới đổi weights. Và "cho validation
   vào fit thì mô hình tốt hơn" cũng sai về phương pháp: điểm đo sẽ
   không còn trung thực.

### G15. Vì sao gradient check float32 sai số lớn hơn float64 nhiều?

1. **Đáp ngắn:** Vì float32 làm tròn nhiều hơn float64, và finite
   difference trừ hai số gần bằng nhau nên càng khuếch đại sai số
   làm tròn đó.
2. **Giải thích sâu:** Epsilon máy (khoảng cách tương đối nhỏ nhất
   phân biệt được với 1) của float32 là khoảng 1,2e-07, của float64
   là khoảng 2,2e-16, tức float32 làm tròn thô hơn khoảng 5e8 lần.
   Gradient số tính bằng finite difference trung tâm
   `(f(x+h) - f(x-h)) / (2h)` với `h = eps = 1e-3`: tử số là hiệu
   hai giá trị loss gần bằng nhau, nên sai số tuyệt đối do làm tròn
   của mỗi lần tính loss bị chia cho `2h = 2e-3`, tức khuếch đại
   khoảng 500 lần; đồng thời `h` nhỏ cũng làm sai số cắt cụt cỡ
   `O(h^2)` nhỏ đi, nên với float32 thì sai số làm tròn chiếm ưu
   thế. Vì vậy gradient check của bản manual (tensor float32) đo
   được sai số tương đối tối đa 1,739e-03 với eps 1e-3, và test đặt
   ngưỡng 1e-2: đủ rộng để chứa sai số làm tròn này nhưng vẫn đủ
   chặt để bắt lỗi đạo hàm thật (thiếu hệ số `2/n` của MSE sẽ lệch
   cỡ `O(1)`). Bản NumPy dùng float64 nên cùng phép kiểm chứng chỉ
   lệch khoảng 1,7e-08 với ngưỡng 1e-5. Kiểm chứng float64 trong `tests/test_mlp_torch_manual.py` (mạng 3 -> 4 -> 4 -> 1,
   41 tham số, eps 1e-3, central difference) đo được sai số tương đối tối đa
   1,475e-11 và sai số tuyệt đối tối đa 8,517e-13 (ngưỡng 1e-7), nhỏ hơn kết quả
   float32 (1,739e-03) khoảng 8 bậc.
3. **Tham chiếu source:** `tests/test_mlp_torch_manual.py`
   (`GRADIENT_TOLERANCE = 1e-2`, `FINITE_DIFF_EPS = 1e-3`, đo được
   1,739e-03); `tests/test_mlp_gradient.py` (bản NumPy float64,
   khoảng 1,7e-08, ngưỡng 1e-5);
   `src/house_price/models/mlp_torch_manual.py`, `_backward` và
   `mse_loss`.
4. **Hiểu lầm thường gặp:** "Sai số 1,739e-03 nghĩa là chain rule
   viết sai." Sai: sai số này là sai số làm tròn của phép kiểm
   chứng (finite difference trên float32), không phải lỗi đạo hàm;
   ngưỡng 1e-2 được chọn đúng vì lý do đó. Và "giảm eps của finite
   difference sẽ luôn chính xác hơn" cũng sai: eps càng nhỏ thì sai
   số cắt cụt càng nhỏ nhưng sai số làm tròn do phép trừ càng lớn.

---

## H. Bảng mapping slide-to-source (6 bước của cô giáo)

6 bước trong slide "Training a DNN with PyTorch" map vào code như sau.
Cột notebook dùng số cell trong
`notebooks/practice_04_mlp_regression.ipynb`.

| Bước trong slide | Source chính xác (bản PyTorch chính thức) | Notebook | Bản viết tay (đối chiếu) |
|---|---|---|---|
| 1. Chuẩn bị dữ liệu: Dataset, DataLoader | `mlp_torch.py`, `MLPTorchRegressorModel.fit`: `TensorDataset` (float32), `DataLoader` (batch 32, shuffle, generator seed 42) | Cell 33 (gọi `fit`; dữ liệu 88 cột chuẩn bị ở bước 5) | `mlp_torch_manual.py`, `fit`: cùng `TensorDataset` + `DataLoader` |
| 2. Định nghĩa mô hình: nn.Module | `mlp_torch.py`, `_MLPNetwork.__init__`/`forward` (dòng 31-51): `nn.Sequential` Linear -> ReLU -> Linear -> ReLU -> Linear; `build_model` seed và tạo mạng | Cell 31 (`MLPTorchRegressorModel(...).build_model(input_size=88)`) | `mlp_torch_manual.py`, `build_model` (dòng 97-137): list tensor weights/biases, He init, `requires_grad=False`; không có `nn.Module` |
| 3. Hàm loss MSE và optimizer Adam | `mlp_torch.py`, `fit`: `criterion = nn.MSELoss()`, `optimizer = torch.optim.Adam(network.parameters(), lr=0.001)` | Cell 33 (nằm trong `fit`, notebook không viết lại loss/optimizer) | `mlp_torch_manual.py`: `mse_loss` (dòng 39) và `_adam_update` (dòng 208) viết tay; không dùng `nn.MSELoss`/`torch.optim` |
| 4. Training loop | `mlp_torch.py`, `fit` (dòng 170-183): `train()` rồi mỗi batch `zero_grad` -> forward -> loss -> `loss.backward()` -> `optimizer.step()` | Cell 33 (`mlp_torch_model.fit(...)`) | `mlp_torch_manual.py`, `fit` (dòng 310-320): `_forward` -> `_backward` -> `_adam_update` |
| 5. Validation | `mlp_torch.py`, `fit`: sau mỗi epoch `_mse` trên train và validation, ghi `history_["loss"]`/`["val_loss"]`; `_mse` dùng `@torch.no_grad()` + `eval()` | Cell 35 (vẽ đường loss train/val) | `mlp_torch_manual.py`, `fit`: `_mse` viết tay, cùng `history_` |
| 6. Save/Load | `mlp_torch.py`, `save`/`load` (dòng 210-246): `torch.save` checkpoint (state_dict + cấu hình + history) ra `outputs/mlp_torch_parameters.pt`; `load` dùng `torch.load(..., map_location="cpu", weights_only=True)` + `load_state_dict` | Cell 33 (`fit` tự lưu; notebook không demo `load` riêng) | `mlp_torch_manual.py`, `save`/`load` (dòng 362-462): lưu trực tiếp `weights`/`biases`, không dùng `state_dict` |

Lưu ý: bước 4 của bản viết tay là nơi duy nhất khác hẳn bản chính
thức (tự tính gradient thay cho autograd); các bước còn lại giữ
nguyên vai trò.

---

## I. Quick viva guide: trả lời 1 câu trong 30 giây

Đọc câu hỏi, trả lời đúng một ý chính rồi dừng; giảng viên hỏi sâu
thì mở mục G tương ứng.

| Câu hỏi thường gặp | Trả lời 30 giây |
|---|---|
| Tensor khác NumPy array thế nào? | Tensor giống array nhưng ghi đồ thị tính toán để autograd tính đạo hàm và chạy được trên GPU; dự án dùng float32, device CPU mặc định. (G1) |
| DataLoader làm gì? | Gói (X, y) bằng TensorDataset rồi chia batch 32, shuffle mỗi epoch bằng generator có seed nên lặp lại được. (G2) |
| nn.Module là gì? | Lớp cơ sở của mạng: `__init__` khai báo tầng, `forward` định nghĩa đường dữ liệu; gọi `network(x)` là chạy `forward` kèm ghi đồ thị. (G3) |
| Vì sao output dùng Linear? | Hồi quy cần số liên tục không giới hạn; ReLU cắt âm, Sigmoid/Softmax gò vào khoảng, đều sai bản chất bài toán. (G4) |
| Forward pass chạy thế nào? | Batch `(32, 88)` qua Linear/ReLU thành `(32, 32)`, `(32, 16)`, rồi Linear cuối ra `(32, 1)`. (G5) |
| Vì sao dùng MSE? | Trung bình bình phương sai số, khả vi khắp nơi, đạo hàm `(2/B)(ŷ - y)` là điểm bắt đầu của backprop. (G6) |
| Adam khác GD thế nào? | GD trừ thẳng `lr * gradient`; Adam làm mượt gradient (m, v) rồi cập nhật từng tham số theo độ lớn gradient của nó. (G7) |
| Vì sao phải zero_grad? | PyTorch cộng dồn gradient vào `.grad`; không xóa thì gradient batch sau cộng chồng lên batch trước. (G8) |
| backward() và step() khác nhau thế nào? | `backward()` tính gradient bằng autograd + chain rule; `step()` cập nhật tham số bằng gradient đó; `step()` không phải backprop. (G9) |
| train() vs eval(), no_grad()? | `train()`/`eval()` đổi chế độ (Dropout/BatchNorm); mạng này không có nên không đổi số, vẫn gọi cho đúng. `no_grad()` tắt ghi đồ thị khi đánh giá/dự đoán. (G10) |
| state_dict là gì? | Dict tên tầng -> tensor tham số; `torch.save` ghi ra `.pt`, `load` phải build lại mạng rồi `load_state_dict`; luôn `map_location="cpu"`. (G11) |
| Cái nào mới là manual thật? | Tự viết đạo hàm chain rule và Adam mới là manual; PyTorch cung cấp tensor, autograd, optimizer, DataLoader. Bản viết tay viết cả hai, bản chính thức để PyTorch làm. (G12) |
| Vì sao bản manual đặt requires_grad=False? | Để training không qua autograd: tham số không ghi đồ thị nên backward() không có gì để vi phân; mọi gradient đều tính tay. (G13) |
| Vì sao validation không cập nhật weights? | Validation chỉ đi qua _mse (forward + no_grad), không backward, không bước cập nhật; nó chỉ để theo dõi và chọn cấu hình. (G14) |
| Vì sao gradient check float32 lệch hơn float64? | Epsilon máy float32 lớn hơn float64 khoảng 5e8 lần, finite difference trừ hai số gần bằng nhau nên khuếch đại sai số làm tròn; đo được 1,739e-03 với ngưỡng 1e-2. (G15) |
| 3 bản MLP khác nhau thế nào? | Cùng kiến trúc 32-16-1; chính thức dùng autograd + Adam của torch, NumPy legacy viết tay bằng NumPy, bản viết tay viết tay bằng tensor torch; chỉ bản chính thức nằm trong benchmark. (G12) |
| Vì sao tách validation và test? | Validation để chọn mô hình và siêu tham số; test chỉ đo một lần cuối, dùng test để chọn thì điểm lạc quan giả. (A5) |
| Kết quả chính là gì? | Test 19/11/2026, 3.241 dòng: RF R2 0,8638 tốt nhất, PyTorch MLP 0,8334, Linear 0,8119, DT 0,7314; bản NumPy MLP ngoài bảng (val 0,8310). (G12) |
