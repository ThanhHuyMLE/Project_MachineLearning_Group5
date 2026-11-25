# Đối chiếu với project CLASSIFYING-SPAM-EMAILS

Tài liệu này đối chiếu project Hồi quy giá nhà với cấu trúc trong
`CLASSIFYING-SPAM-EMAILS`. Mục tiêu là nhận diện khoảng trống chức năng,
không sao chép máy móc các file của bài toán Phân loại.

## Phần đã có tương đương

| Project spam | Project giá nhà | Trạng thái |
|---|---|---|
| `src/preprocess.py` | `house_price/preprocessing.py` | Đầy đủ; có chia theo `id`, validation và chống Data Leakage |
| `src/model.py` | `house_price/models/` | Đầy đủ hơn; tách class cho từng mô hình |
| `src/eda.py` | `house_price/eda.py` | Đã có |
| `src/evaluate.py` | `house_price/evaluation.py` | Đã có metric đúng cho Regression |
| `src/visualize.py` | `house_price/visualization.py` | Đã có |
| `src/train.py`, `pipeline.py` | `house_price/train_evaluate.py` | Đã gộp thành một pipeline có CLI |

Không thêm lại các wrapper `train.py`, `evaluate.py`, `visualize.py` vì sẽ tạo
hai đường chạy cho cùng một logic và dễ làm kết quả không nhất quán.

## Khoảng trống đã bổ sung

### Artifact suy luận

Trước đây pipeline chỉ lưu checkpoint MLP. Median, scaler và one-hot encoder
chưa được lưu, nên không thể tái tạo đúng 88 đặc trưng cho dữ liệu mới.

`house_price/artifacts.py` lưu:

- `mlp_torch_parameters.pt`: checkpoint MLP;
- `preprocessor.joblib`: preprocessor đã fit trên train;
- `inference_manifest.json`: phiên bản định dạng và hợp đồng cột.

Ngoài artifact suy luận, cả bốn mô hình chính thức đều xuất trạng thái
đã học sang JSON. Decision Tree lưu các node dạng mảng gọn; Random Forest
nén từng cây bằng zlib + base64 bên trong JSON để tránh file quá lớn.

### Dự đoán dữ liệu mới

`house_price/predict.py` nhận CSV thô, tạo Feature Engineering, dùng lại
preprocessor đã lưu rồi dự đoán bằng MLP. File đầu ra giữ nguyên cột đầu vào và
thêm `predicted_price`.

### Báo cáo độc lập

`house_price/report.py` đọc `metrics.csv` và sinh `report.md`, `report.html`
mà không huấn luyện lại mô hình.

### Hợp đồng dữ liệu suy luận

`prepare_feature_frame()` tạo đúng 19 cột đầu vào từ dữ liệu chưa có `price`.
`clean_dataset()` tiếp tục dành cho dữ liệu huấn luyện và bắt buộc có mục tiêu.

## Phần chưa đưa vào

- `tracking.py`: pipeline hiện đã in thời gian từng model và ghi toàn bộ
  metrics/predictions. Run tracking nhiều phiên chưa cần thiết cho phạm vi lab.
- `config.yaml`: cấu hình benchmark đang tập trung trong `BENCHMARK_CONFIG` và
  đã có test. Chuyển sang YAML lúc này tạo thêm dependency và hai nguồn cấu hình.
- `app.py`: giao diện Streamlit không thuộc yêu cầu bài thực hành và cần thêm
  dependency. CLI dự đoán đủ để kiểm chứng luồng suy luận.

## Lệnh mới

Sau khi chạy huấn luyện:

```bash
PYTHONPATH=src python -m house_price.predict \
  --input data/new_houses.csv \
  --artifact-dir outputs \
  --output outputs/predictions_new.csv
```

Sinh báo cáo từ benchmark test:

```bash
PYTHONPATH=src python -m house_price.report --metrics outputs/metrics.csv
```
