# MAL_Class

Đồ án độc lập về phân loại đa nhãn họ mã độc và suy luận hành vi bằng Hybrid Attention đa phương thức. MAL_dectect chỉ là nguồn tham khảo ý tưởng; các kết quả của dự án đó không phải kết quả của MAL_Class.

## Phạm vi

Đầu vào bao gồm PE headers, sections, entropy, imports/exports, chữ ký số, strings và signature nguyên tử; sự kiện sandbox có tham số, thời gian, trạng thái; đồ thị quan hệ process/thread/file/registry/network. Không chạy mẫu mã độc. SHA-256 chỉ dùng định danh và khử trùng lặp.

Trọng tâm mô hình là ba nhánh đầu vào độc lập: Signature (PEiD/YARA an toàn/imphash/fuzzy prototype/chữ ký số), Static PE (header/section/import/resource/string/entropy) và Dynamic/Behavior (chuỗi event theo thời gian cùng execution graph). Mỗi nhánh có self-attention riêng. Signature và Static PE cross-attend trực tiếp tới token của nhau, rồi cả hai nhánh cross-attend tới event/graph token; event/graph token cũng chú ý ngược lại tới từng bằng chứng cấu trúc. Vì vậy mô hình có thể học liên hệ giữa nhiều dấu hiệu, như import/DLL đáng ngờ cùng file được tải, ghi rồi thực thi, thay vì chỉ cộng ba điểm tổng quát. Temporal attention giữ thứ tự, process/thread, success và thời gian; graph attention giữ quan hệ thực thể và causal path. Learned query riêng cho family/behavior tiếp tục chú ý tới ba nhánh và evidence token để phân loại đa nhãn.

Mã đang phát triển. Parser hiện đọc đúng `target.file.pe` và `static.pe`, lọc toàn bộ YARA rule family/verdict và giữ các nhóm feature khi imports vượt budget. Pilot PE-v2 có hai seed trên cùng reservoir/split WinMET 1.000 mẫu: validation mAP trung bình 0.408 so với bag 0.293, calibrated macro-F1 0.230 so với 0.146. Top-1/top-3 chưa thắng nhất quán; chỉ 63 validation rows có family đã học để tính ranking, và behavior head chưa train. Các số liệu trước khi sửa schema/YARA là lịch sử, không đại diện cho pipeline hiện tại. Verdict malware giữ `UNKNOWN` cho đến khi đạt release gate độc lập. Chưa có benchmark nghiệm thu. Chi tiết ở [trạng thái triển khai](docs/implementation-status.md) và [workflow WinMET](docs/winmet-workflow.md).

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
PYTHONPATH=src python3 -m pytest -q
```

Lệnh train chia dữ liệu thành train/validation/calibration/test riêng; family calibration dùng simultaneous split-conformal trên vector nhãn đã biết để tránh gọi hàng trăm marginal 90% dự đoán là “confirmed” đồng thời. Checkpoint cũ chỉ có marginal conformal sẽ hạ kết quả thành `suspected`, không `confirmed`. Malware verdict `MALICIOUS`/`BENIGN` cũng trả `UNKNOWN` trừ khi checkpoint vượt release gate độc lập; train/calibration đơn thuần không đủ điều kiện. Imphash chỉ khớp prototype train theo digest chính xác; TLSH/ssdeep dùng fuzzy similarity. Không đưa raw digest vào token mô hình. Lệnh `sync-frameworks` tải đúng ATT&CK STIX 19.2 và NIST SP 800-53 Rev.5.1.1 OSCAL đã khóa SHA-256, sau đó kiểm tra ID/tên trong crosswalk. Không cung cấp checkpoint: chỉ chạy evidence engine, family luôn là `UNKNOWN`, và confidence behavior là mức đầy đủ của rule chứ không phải xác suất mô hình.

Dashboard nhận một hoặc nhiều PE (tối đa 50 MiB/file) và phân tích tĩnh trong bộ đọc PE giới hạn kích thước; file không được thực thi. Chế độ batch nhận upload hoặc đường dẫn thư mục local (tối đa 500 file/256 MiB, không theo symlink), xuất từng kết quả và lỗi sang CSV. Đầu vào CAPE/WinMET JSON mới cung cấp luồng động. Có thể huấn luyện từ thư mục report tại máy hoặc EMBER2024 Win32 ZIP đã khóa checksum. CAPE/WinMET có sample cap seeded-reservoir (UI mặc định 600); EMBER2024 lấy mẫu reservoir có seed trên shard tuần, mặc định tối đa 7.000 mẫu. Vì vậy các kết quả subset không phải benchmark chính thức. Các TTP yếu trong EMBER không được dùng làm ground-truth âm; behavior head chỉ bật khi split train có cả bằng chứng dương và âm mạnh. Ngưỡng family được hiệu chuẩn riêng trên calibration split, giới hạn empirical FPR, rồi đánh giá trên test temporal riêng. Cài package và extras bằng `pip install '.[dev,dashboard]'`.

Dashboard mặc định **Tự chọn model theo đầu vào**: PE hoặc report chỉ có static dùng model Signature–Static; report có event thực thi dùng model hybrid. [Registry model local](configs/inference_models.json) khóa tên file, kích thước và SHA-256; mỗi kết quả/CSV ghi rõ model đã dùng. Không tự tải weights hoặc chạy EXE. Model chưa cài thì trả `UNKNOWN`; checksum sai báo lỗi. Lựa chọn checkpoint thủ công vẫn ghi đè chế độ tự chọn. Xem [hướng dẫn routing](docs/model-routing.md).

Checkpoint cần phù hợp nhóm feature đầu vào. Guard vocabulary trả `UNKNOWN` và bỏ neural inference nếu dưới 50% token được checkpoint nhận diện; dashboard/CSV ghi rõ lý do. Đây là kiểm tra tương thích ban đầu, chưa phải bộ phát hiện OOD đã kiểm định. CLI `predict` không truyền checkpoint vẫn là evidence-only; cơ chế tự chọn được dùng trong dashboard.

Behavior chỉ được `confirmed` khi có điểm hợp lệ từ behavior head đã train và chuỗi evidence đầy đủ. Nếu chỉ có evidence engine, kết luận tối đa là `suspected`; `evidence_status` ghi riêng chuỗi có đủ bước hay không. Mức đầy đủ chuỗi không phải xác suất của mô hình.

MalBehavD-V1 chỉ dùng trong thí nghiệm chuỗi API động riêng, không ghép với PE/CAPE khác nguồn. Split group theo SHA-256 và chuỗi API chuẩn hóa để tránh rò rỉ chuỗi lặp; các nhóm có cùng chuỗi nhưng nhãn xung đột bị cách ly. Nhãn giữ dạng `0/1` vì README tác giả không chỉ rõ mapping số sang benign/malware. Kết quả nhánh này không đại diện cho phân loại family hay ATT&CK.

Bộ protocol có đúng 7 baseline: indicator rule, bag-of-token, BiLSTM, temporal transformer, graph attention, sequence+graph và full hybrid attention. Có công cụ kiểm định flow ablation, deletion AOPC, stability và Spearman agreement với Layer Integrated Gradients ở mức event; chưa hoàn thành tất cả phép đo trên dữ liệu thật. Runner `scripts/compare_winmet_protocol_baselines.py` tái dựng và kiểm tra fingerprint các split của experiment đã lưu, dùng chung vocabulary train, hàm loss và budget epoch, chỉ chọn model bằng validation và fit threshold bằng calibration. Các so sánh bag lịch sử không dùng auxiliary ranking loss như attention, nên không được quy toàn bộ chênh lệch đó cho kiến trúc.

```bash
PYTHONPATH=src python scripts/compare_winmet_protocol_baselines.py \
  artifacts/winmet/volume2_pev2_hybrid3_attention_1000_seed44.json \
  --report artifacts/winmet/volume2_pev2_protocol_baselines_1000_seed44.json \
  --models temporal_transformer graph_attention sequence_graph bag_of_tokens --device mps
```

Lệnh trên cần experiment, JSONL và checkpoint local đã tạo trước đó; không tự tải chúng, không ghi đè report và không tính metric trên test. `indicator_rule` không có family classifier nên không có điểm family để so sánh.

Parser trace hiện dùng `cape-process-relations-v3`: nhận diện pseudo-handle của chính process nhưng không coi thao tác self-memory là remote injection. Checkpoint mới lưu phiên bản chuẩn hóa đã train; registry chọn model v3 đã train lại và từ chối trace khác phiên bản. Pilot v3 trên cùng split đạt validation mAP/macro-F1 0.3569/0.2282, so với bag cùng hàm loss 0.2537/0.1554; chưa tốt hơn v2 và chưa đạt nghiệm thu. Các metric PE-v2 ở trên không được gán cho bản v3. Runner WinMET mới dùng cùng ranking loss cho attention và bag mặc định; muốn tái lập baseline lịch sử phải chỉ rõ `--bag-ranking-weight 0`.

Attention theo cửa sổ mới dùng `per_sample_v1`, tránh thay đổi kết quả chỉ vì padding của batch. Checkpoint cũ giữ `legacy_batch_tail`, không đổi nghĩa ngầm; để tái lập training cũ bằng runner WinMET, chỉ rõ `--temporal-window-policy legacy_batch_tail`. Model sửa cửa sổ chưa được thay vào registry trước khi train/test. Các ablation mới `remove_resource_relations`, `remove_process_context`, `mask_graph_edges_keep_nodes` giữ nguyên token/node/nhãn sau encoding; phép phá target/xóa cạnh ở mức report cũ có nhiễu do đổi graph và pruning.

Biến thể [relation-value attention](docs/relation-value-attention.md) bổ sung quan hệ vào phần value thay vì chỉ bias trước softmax; bật bằng `--temporal-relation-values` trong runner WinMET. Mặc định tắt, checkpoint cũ không tự bật tham số chưa train, và model UI không tự đổi. Unit test/gradient không được dùng thay bằng chứng tăng độ chính xác; cần train/test cùng split với model `per_sample_v1` không bật biến thể này.

## Tài liệu và nguồn tham khảo

- [Trạng thái triển khai](docs/implementation-status.md).
- [Tải và chuẩn hóa WinMET có giới hạn tài nguyên](docs/winmet-workflow.md).
- [MalBehavD-V1 phụ trợ: API sequence, giới hạn và điều kiện dùng](docs/malbehavd-workflow.md).
- [README gốc được bảo tồn](references/legacy/README.original.md).
- [MAL_dectect](https://github.com/luonglt20/MAL_dectect): tham khảo ý tưởng.
- [EMBER2024 của nhóm tác giả](https://github.com/FutureComputing4AI/EMBER2024) và [bản lưu dữ liệu được khóa checksum](https://huggingface.co/datasets/joyce8/EMBER2024): nguồn feature tĩnh/nhãn; không chứa binary để thực thi.

Các file cũ trong Data/, Documentation/, Reports/ và các script BiLSTM/XAI cũ trong Source_Code/ là hiện vật tham khảo; không nằm trong pipeline malware-hybrid và không được dùng làm kết quả nghiệm thu của đồ án mới.
