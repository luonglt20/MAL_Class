# MAL_Class

Đồ án độc lập về phân loại đa nhãn họ mã độc và suy luận hành vi bằng Hybrid Attention đa phương thức. MAL_dectect chỉ là nguồn tham khảo ý tưởng; các kết quả của dự án đó không phải kết quả của MAL_Class.

## Phạm vi

Đầu vào bao gồm PE headers, sections, entropy, imports/exports, chữ ký số, strings và signature nguyên tử; sự kiện sandbox có tham số, thời gian, trạng thái; đồ thị quan hệ process/thread/file/registry/network. Không chạy mẫu mã độc. SHA-256 chỉ dùng định danh và khử trùng lặp.

Static self-attention, temporal attention với bias quan hệ/thời gian theo cặp và graph attention được kết hợp bằng cross-attention hai chiều ở mức event. Nhánh graph dùng Fourier time encoding và causal multi-hop path attention. Mỗi family và ATT&CK technique có learned query riêng để chú ý tới bằng chứng liên quan; family dùng sigmoid/BCE đa nhãn. Evidence engine gồm 15 chuỗi nhiều bước và kiểm tra thứ tự, success, process/target trước khi xác nhận behavior.

Mã đang phát triển. Đã có pilot family-attention trên reservoir WinMET 500 mẫu, nhưng held-out macro-F1 còn thấp và behavior head chưa được huấn luyện vì thiếu nhãn âm/dương độc lập. Chưa có ngưỡng nghiệm thu full-hybrid nào được chứng minh; mọi số liệu của MAL_dectect đều không được dùng làm kết quả của dự án này.

## Sử dụng

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
malware-hybrid inspect-schema tests/fixtures/cape_process_injection.json
malware-hybrid predict tests/fixtures/cape_process_injection.json
malware-hybrid train /path/to/cape-reports --output hybrid_attention.pt --max-reports 600 --seed 42 --window-size 256 --window-overlap 128 --conformal-alpha 0.10
malware-hybrid predict report.json --checkpoint hybrid_attention.pt
malware-hybrid download-ember --output Data/raw
malware-hybrid train-ember Data/raw/Win32_train.zip --output models/ember2024_hybrid.pt --rows-per-week 128 --max-samples 7000
malware-hybrid train-malbehavd Data/raw/malbehavd/MalBehavD-V1-dataset.csv --output models/malbehavd_api_attention.pt --seed 42
malware-hybrid download-ember --split test --output Data/raw
malware-hybrid test-ember Data/raw/Win32_test.zip --checkpoint models/ember2024_hybrid.pt
PYTHONPATH=src streamlit run Source_Code/hybrid_attention_dashboard.py
malware-hybrid sync-frameworks --output artifacts/frameworks
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Lệnh train chia dữ liệu thành train/validation/calibration/test riêng; calibration tạo tập dự đoán Mondrian split-conformal nhị phân cho từng family và chọn ngưỡng cảnh báo theo FPR trên benign calibration. Một mẫu không vượt ngưỡng cảnh báo mặc định trả `UNKNOWN`; chỉ trả `BENIGN` nếu có ngưỡng benign riêng đã được hiệu chuẩn. Imphash chỉ khớp prototype train theo digest chính xác; TLSH/ssdeep dùng fuzzy similarity. Không đưa raw digest vào token mô hình. Lệnh `sync-frameworks` tải đúng ATT&CK STIX 19.2 và NIST SP 800-53 Rev.5.1.1 OSCAL đã khóa SHA-256, sau đó kiểm tra ID/tên trong crosswalk. Không cung cấp checkpoint: chỉ chạy evidence engine, family luôn là `UNKNOWN`, và confidence behavior là mức đầy đủ của rule chứ không phải xác suất mô hình.

Dashboard nhận file PE tối đa 50 MiB và phân tích tĩnh trong bộ đọc PE giới hạn kích thước; file không được thực thi. Đầu vào CAPE/WinMET JSON mới cung cấp luồng động. Có thể huấn luyện từ thư mục report tại máy hoặc EMBER2024 Win32 ZIP đã khóa checksum. CAPE/WinMET có sample cap seeded-reservoir (UI mặc định 600); EMBER2024 lấy mẫu reservoir có seed trên shard tuần, mặc định tối đa 7.000 mẫu. Vì vậy các kết quả subset không phải benchmark chính thức. Các TTP yếu trong EMBER không được dùng làm ground-truth âm; behavior head chỉ bật khi split train có cả bằng chứng dương và âm mạnh. Ngưỡng family được hiệu chuẩn riêng trên calibration split, giới hạn empirical FPR, rồi đánh giá trên test temporal riêng. Cài package và extras bằng `pip install '.[dev,dashboard]'`.

MalBehavD-V1 chỉ dùng trong thí nghiệm chuỗi API động riêng, không ghép với PE/CAPE khác nguồn. Split group theo SHA-256 và chuỗi API chuẩn hóa để tránh rò rỉ chuỗi lặp; các nhóm có cùng chuỗi nhưng nhãn xung đột bị cách ly. Nhãn giữ dạng `0/1` vì README tác giả không chỉ rõ mapping số sang benign/malware. Kết quả nhánh này không đại diện cho phân loại family hay ATT&CK.

Bộ protocol có đúng 7 baseline: indicator rule, bag-of-token, BiLSTM, temporal transformer, graph attention, sequence+graph và full hybrid attention. Attention được kiểm định bằng flow ablation, deletion AOPC, stability và Spearman agreement với Layer Integrated Gradients ở mức event. Các lớp baseline đã chạy được, nhưng chưa được phép ghi điểm benchmark cho tới khi chạy cùng split dữ liệu thật.

## Tài liệu và nguồn tham khảo

- [Trạng thái triển khai](docs/implementation-status.md).
- [Tải và chuẩn hóa WinMET có giới hạn tài nguyên](docs/winmet-workflow.md).
- [MalBehavD-V1 phụ trợ: API sequence, giới hạn và điều kiện dùng](docs/malbehavd-workflow.md).
- [README gốc được bảo tồn](references/legacy/README.original.md).
- [MAL_dectect](https://github.com/luonglt20/MAL_dectect): tham khảo ý tưởng.
- [EMBER2024 của nhóm tác giả](https://github.com/FutureComputing4AI/EMBER2024) và [bản lưu dữ liệu được khóa checksum](https://huggingface.co/datasets/joyce8/EMBER2024): nguồn feature tĩnh/nhãn; không chứa binary để thực thi.

Các file cũ trong Data/, Documentation/, Reports/ và các script BiLSTM/XAI cũ trong Source_Code/ là hiện vật tham khảo; không nằm trong pipeline malware-hybrid và không được dùng làm kết quả nghiệm thu của đồ án mới.
