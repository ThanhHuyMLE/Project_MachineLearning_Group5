# ADR-004: Lựa chọn triển khai MLP ban đầu

- Trạng thái: Đã thay thế bởi [ADR-008](008-mlp-pytorch.md)
- Ngày thay thế: 19/11/2026

## Bối cảnh

ADR này từng ghi nhận các lựa chọn triển khai MLP trong giai đoạn
khởi tạo dự án. Quyết định đó không còn áp dụng cho source hiện tại.

## Quyết định hiện hành

MLP chính thức dùng PyTorch qua `MLPTorchRegressorModel` trong
`src/house_price/models/mlp_torch.py`. Mô hình tuân theo workflow
Dataset/DataLoader, `torch.nn`, MSELoss, autograd và Adam. Chi tiết và
lý do lựa chọn được ghi tại ADR-008.

Bản `MLPTorchManualModel` trong
`src/house_price/models/mlp_torch_manual.py` chỉ phục vụ học thuật và
vấn đáp; không nằm trong benchmark chính thức.
