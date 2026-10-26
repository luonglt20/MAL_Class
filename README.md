# MAL_Class

Đồ án độc lập về phân loại đa nhãn họ mã độc và suy luận hành vi bằng Hybrid Attention đa phương thức. MAL_dectect chỉ là nguồn tham khảo ý tưởng; các kết quả của dự án đó không phải kết quả của MAL_Class.

## Phạm vi

Đầu vào bao gồm PE headers, sections, entropy, imports/exports, chữ ký số, strings và signature nguyên tử; sự kiện sandbox có tham số, thời gian, trạng thái; đồ thị quan hệ process/thread/file/registry/network. Không chạy mẫu mã độc. SHA-256 chỉ dùng định danh và khử trùng lặp.

Trọng tâm mô hình là ba nhánh đầu vào độc lập: Signature (PEiD/YARA an toàn/imphash/fuzzy prototype/chữ ký số), Static PE (header/section/import/resource/string/entropy) và Dynamic/Behavior (chuỗi event theo thời gian cùng execution graph). Mỗi nhánh có self-attention riêng; sau đó token signature và static PE cross-attend trực tiếp tới event/graph token, còn event/graph token cross-attend ngược lại tới từng bằng chứng cấu trúc. Vì vậy mô hình có thể học liên hệ như import/DLL đáng ngờ ↔ hành động tải/ghi/chạy file, thay vì chỉ cộng ba điểm tổng quát. Temporal attention giữ thứ tự, process/thread, success và thời gian; graph attention giữ quan hệ thực thể và causal path. Learned query riêng cho family/behavior tiếp tục attention tới ba nhánh và evidence token để phân loại đa nhãn.

Mã đang phát triển. Các pilot WinMET 1.000/1.600 trước đây và EMBER2024 static-only được huấn luyện trước khi tách nhánh Signature; xem chúng là kết quả lịch sử, không gán cho kiến trúc hiện tại. Kiến trúc ba nhánh hiện tại đã chạy hai seed trên cùng split WinMET 1.600 mẫu: attention mAP 0.285/0.369 so với bag 0.264/0.247; khi tắt cross-attention, mAP giảm 0.024/0.059 và calibrated macro-F1 giảm 0.051/0.089. Đây mới là tín hiệu ban đầu trên một split theo cụm, AVClass weak labels; chỉ 105 mẫu có family để đánh giá, static evidence trung vị 7 token/report, behavior head tắt do thiếu nhãn độc lập, và withheld split chưa được đánh giá. Model chỉ đưa family score có ý nghĩa khi chọn checkpoint tương ứng; không checkpoint thì family `UNKNOWN`. Verdict malware tiếp tục `UNKNOWN` trừ khi đạt release gate độc lập. Không có kết quả nào hiện là benchmark nghiệm thu hay bảo đảm phát hiện. Chi tiết và giới hạn ở [trạng thái triển khai](docs/implementation-status.md) và [workflow WinMET](docs/winmet-workflow.md).

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

Lệnh train chia dữ liệu thành train/validation/calibration/test riêng; family calibration dùng simultaneous split-conformal trên vector nhãn đã biết để tránh gọi hàng trăm marginal 90% dự đoán là “confirmed” đồng thời. Checkpoint cũ chỉ có marginal conformal sẽ hạ kết quả thành `suspected`, không `confirmed`. Malware verdict `MALICIOUS`/`BENIGN` cũng trả `UNKNOWN` trừ khi checkpoint vượt release gate độc lập; train/calibration đơn thuần không đủ điều kiện. Imphash chỉ khớp prototype train theo digest chính xác; TLSH/ssdeep dùng fuzzy similarity. Không đưa raw digest vào token mô hình. Lệnh `sync-frameworks` tải đúng ATT&CK STIX 19.2 và NIST SP 800-53 Rev.5.1.1 OSCAL đã khóa SHA-256, sau đó kiểm tra ID/tên trong crosswalk. Không cung cấp checkpoint: chỉ chạy evidence engine, family luôn là `UNKNOWN`, và confidence behavior là mức đầy đủ của rule chứ không phải xác suất mô hình.

Dashboard nhận một hoặc nhiều PE (tối đa 50 MiB/file) và phân tích tĩnh trong bộ đọc PE giới hạn kích thước; file không được thực thi. Chế độ batch nhận upload hoặc đường dẫn thư mục local (tối đa 500 file/256 MiB, không theo symlink), xuất từng kết quả và lỗi sang CSV. Đầu vào CAPE/WinMET JSON mới cung cấp luồng động. Có thể huấn luyện từ thư mục report tại máy hoặc EMBER2024 Win32 ZIP đã khóa checksum. CAPE/WinMET có sample cap seeded-reservoir (UI mặc định 600); EMBER2024 lấy mẫu reservoir có seed trên shard tuần, mặc định tối đa 7.000 mẫu. Vì vậy các kết quả subset không phải benchmark chính thức. Các TTP yếu trong EMBER không được dùng làm ground-truth âm; behavior head chỉ bật khi split train có cả bằng chứng dương và âm mạnh. Ngưỡng family được hiệu chuẩn riêng trên calibration split, giới hạn empirical FPR, rồi đánh giá trên test temporal riêng. Cài package và extras bằng `pip install '.[dev,dashboard]'`.

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
