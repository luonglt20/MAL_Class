# Đọc đóng góp Attention trên UI

Sau khi phân tích PE hoặc CAPE JSON bằng checkpoint ba nhánh tương thích,
UI hiển thị **Đóng góp Attention vào điểm family ứng viên**. Ứng viên là family
có logit lớn nhất trước quyết định abstain. Nó có thể khác kết luận cuối cùng
`UNKNOWN`; bảng giải thích không biến ứng viên thành nhãn được xác nhận.

- Đóng góp dương tăng logit của ứng viên đó, đóng góp âm giảm logit.
- Bảng nhóm gồm hai đường đọc: modality và evidence. Cộng tất cả hàng với bias
  trong phần mở rộng sẽ tái dựng logit. Đây không phải tổng xác suất.
- Phần mở rộng hiển thị tối đa mười đóng góp dương và mười âm, kèm chỉ số vị trí
  và mô tả Signature, Static, event hoặc graph. Các hàng rút gọn không đủ để tự
  cộng lại toàn bộ logit; dùng tổng theo nhóm để kiểm tra.
- PE thiếu luồng động vẫn có bảng Signature/Static; không tự tạo behavior động.
- Khi kiểm tra tái dựng sai quá `1e-4`, UI bỏ bảng đóng góp và thông báo chưa đạt.
  Kết quả phân loại vẫn dùng cùng logits và chính sách quyết định.

Các giá trị đã được self/cross-attention kết hợp ngữ cảnh. Đóng góp ở vị trí
Signature có thể chứa thông tin từ Static hoặc Dynamic. Không diễn giải dấu
dương thành “token này độc hại”, không dùng trọng số attention hay đóng góp
này thay thế evidence chain, và không dùng chúng làm ngưỡng nhận nhầm mới.

Pipeline tái sử dụng đúng một lần forward cho cả dự đoán và đóng góp. Temporary
hooks được dọn khi thành công hoặc lỗi; các readout dùng chung model được tuần
tự hóa để tránh lẫn dữ liệu giữa phiên UI. Checkpoint cũ hai nhánh và đầu vào
bị từ chối bởi schema/vocabulary guard không có bảng mới. JSON output chứa
`attention_trace.family_readout` với các trường số thuần, không có tensor.

Kiểm chứng ngày 2026-10-29: hai checkpoint WinMET trên cùng một report thật,
chạy MPS ở eval mode, tái dựng sai lần lượt `2.38e-7` và `5.96e-8`; cả hai giữ
`UNKNOWN`, byte checkpoint không đổi. Playwright đã kiểm tra CAPE fixture và
PE launcher của setuptools trên trình duyệt: bảng hiển thị và route đúng,
không chạy file mẫu. Regression test kiểm tra một forward, tổng đóng góp,
JSON serialization, guard từ chối trước suy luận, dọn hooks và shared-model
concurrency. Đây là kiểm chứng tích hợp, không phải đánh giá accuracy.

[Bản ghi kiểm chứng và SHA-256](experiments/attention-readout-ui-20261029.json).
[Giới hạn faithfulness đã đo](attention-faithfulness.md).
