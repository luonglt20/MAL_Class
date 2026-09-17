# MAL_Class

Đồ án độc lập về phân loại đa nhãn họ mã độc và suy luận hành vi bằng Hybrid Attention đa phương thức. MAL_dectect chỉ là nguồn tham khảo ý tưởng; các kết quả của dự án đó không phải kết quả của MAL_Class.

## Phạm vi

Đầu vào bao gồm PE headers, sections, entropy, imports/exports, chữ ký số, strings và signature nguyên tử; sự kiện sandbox có tham số, thời gian, trạng thái; đồ thị quan hệ process/thread/file/registry/network. Không chạy mẫu mã độc. SHA-256 chỉ dùng định danh và khử trùng lặp.

Static self-attention, temporal attention với bias quan hệ/thời gian theo cặp và graph attention được kết hợp bằng cross-attention hai chiều ở mức event. Nhánh graph dùng Fourier time encoding và causal multi-hop path attention. Mỗi family và ATT&CK technique có learned query riêng để chú ý tới bằng chứng liên quan; family dùng sigmoid/BCE đa nhãn. Evidence engine gồm 15 chuỗi nhiều bước và kiểm tra thứ tự, success, process/target trước khi xác nhận behavior.

Mã hiện đang phát triển. Chưa có thực nghiệm WinMET chứng minh các ngưỡng nghiệm thu; mọi số liệu của MAL_dectect đều không được dùng làm kết quả của dự án này.

## Sử dụng

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
malware-hybrid inspect-schema tests/fixtures/cape_process_injection.json
malware-hybrid predict tests/fixtures/cape_process_injection.json
malware-hybrid train /path/to/cape-reports --output hybrid_attention.pt --window-size 256 --window-overlap 128 --conformal-alpha 0.10
malware-hybrid predict report.json --checkpoint hybrid_attention.pt
malware-hybrid sync-frameworks --output artifacts/frameworks
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Lệnh train chia dữ liệu thành train/validation/calibration/test riêng; calibration tạo tập dự đoán Mondrian split-conformal nhị phân cho từng family. TLSH/ssdeep được đối chiếu với prototype chỉ sinh từ train split, không đưa raw fuzzy hash vào token. Lệnh `sync-frameworks` tải đúng ATT&CK STIX 19.2 và NIST SP 800-53 Rev.5.1.1 OSCAL đã khóa SHA-256, sau đó kiểm tra ID/tên trong crosswalk. Không cung cấp checkpoint: chỉ chạy evidence engine, family luôn là `UNKNOWN`, và confidence behavior là mức đầy đủ của rule chứ không phải xác suất mô hình.

Bộ protocol có đúng 7 baseline: indicator rule, bag-of-token, BiLSTM, temporal transformer, graph attention, sequence+graph và full hybrid attention. Attention được kiểm định bằng flow ablation, deletion AOPC, stability và Spearman agreement với Layer Integrated Gradients ở mức event. Các lớp baseline đã chạy được, nhưng chưa được phép ghi điểm benchmark cho tới khi chạy cùng split dữ liệu thật.

## Tài liệu và nguồn tham khảo

- [Trạng thái triển khai](docs/implementation-status.md).
- [README gốc được bảo tồn](references/legacy/README.original.md).
- [MAL_dectect](https://github.com/luonglt20/MAL_dectect): tham khảo ý tưởng.

Các file cũ trong Data/, Documentation/, Reports/ và các script BiLSTM/XAI cũ trong Source_Code/ là hiện vật tham khảo; không nằm trong pipeline malware-hybrid và không được dùng làm kết quả nghiệm thu của đồ án mới.
