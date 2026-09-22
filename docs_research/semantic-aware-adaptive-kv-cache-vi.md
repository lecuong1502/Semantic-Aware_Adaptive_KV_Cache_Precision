# Semantic-Aware Adaptive KV Cache Precision cho Single-User LLM Inference dưới Áp lực VRAM

*(Dự án độc lập — xây từ đầu, không phụ thuộc vào project cá nhân trước đó. Engine tạm gọi là `MicroInfer`, có thể đổi tên tùy ý.)*

## 1. Outline bài báo

### Tiêu đề
**Semantic-Aware Adaptive KV Cache Precision for Robust Single-User LLM Inference under Unpredictable VRAM Contention on Consumer GPUs**

### Abstract (khung nháp)
- Vấn đề: Inference LLM cục bộ, single-user trên laptop/consumer GPU gặp phải tình trạng **VRAM bị chiếm dụng không thể đoán trước** bởi các ứng dụng chạy song song (trình duyệt, app chat, phần mềm render nền) — khác hẳn datacenter serving nơi việc cấp phát GPU được kiểm soát và có thể dự đoán theo workload.
- Khoảng trống: Các công trình adaptive KV cache trước đây (FineServe, MorphServe, MIRAGE, eLLM, KV-RM) đều giả định môi trường multi-tenant, có scheduler nhìn thấy toàn bộ workload, và tối ưu đồng đều trên toàn bộ cache.
- Đóng góp: (1) Khảo sát thực nghiệm pattern VRAM contention trên laptop khi dùng đa nhiệm thông thường; (2) một cơ chế phát hiện áp lực VRAM tại runtime, nhẹ, không cần scheduler hợp tác; (3) chính sách phân bổ precision KV cache thích ứng theo ngữ nghĩa (FP16 → INT8 → INT4), dựa trên importance score suy ra từ attention thay vì giảm đồng đều; (4) đánh giá NLP theo từng loại tác vụ, cho thấy chỗ nào bị ảnh hưởng nặng nhất (QA văn bản dài, hội thoại nhiều lượt, tóm tắt) và mức độ semantic-aware allocation giảm thiểu tác động này so với uniform quantization và so với hiện trạng OOM/crash.

### 1. Giới thiệu
- Nêu bối cảnh xu hướng inference cục bộ (Ollama, LM Studio, llama.cpp) — ngày càng phổ biến trên laptop có VRAM hạn chế, dùng chung với nhiều ứng dụng khác.
- Đối chiếu rõ ràng với các công trình datacenter serving (trích dẫn FineServe/MorphServe/MIRAGE/eLLM/KV-RM là công trình gần nhất; nêu chính xác giả định nào của họ không còn đúng trong bối cảnh local: GPU riêng, workload có thể quan sát được, lập lịch nhiều request).
- Đặt 3 câu hỏi nghiên cứu (RQ):
  - **RQ1**: VRAM khả dụng trên laptop biến động thế nào trong điều kiện đa nhiệm thực tế, và nó tương tác ra sao với 1 phiên sinh văn bản đang chạy dở?
  - **RQ2**: Một cơ chế runtime, không cần scheduler hợp tác, có thể phát hiện và phản ứng với áp lực VRAM đủ nhanh để tránh crash toàn bộ phiên không?
  - **RQ3**: Việc phân bổ precision theo importance ngữ nghĩa/attention (so với đồng đều) có giữ được chất lượng NLP ở cấp độ tác vụ tốt hơn không, với cùng 1 mức nén trung bình?
- Tóm tắt đóng góp và preview kết quả.

### 2. Related Work
- **KV cache compression/quantization**: KVQuant, KIVI, KVTuner, Cocktail, KVmix, FlashInfer FP8-KV, LMDeploy TurboMind — định vị là phân bổ precision *tĩnh/offline*; công trình của bạn là *runtime-reactive*.
- **Adaptive/runtime-aware serving**: FastCache, FineServe, MorphServe, MIRAGE, eLLM, KV-RM, Recency/Frequency Adaptive KV Caching — đều thuộc bối cảnh datacenter/multi-tenant; nêu rõ trong bảng giả định của từng công trình (GPU dùng riêng/chia sẻ, scheduler có nhìn thấy workload không, multi-request hay single-session) để làm rõ khoảng trống.
- **Attention-guided importance/eviction**: Ada-KV, H2O-style attention-based eviction, AttentionRAG — tái sử dụng ý tưởng importance suy ra từ attention, nhưng áp dụng cho *phân bổ precision dưới áp lực bộ nhớ từ bên ngoài*, chứ không phải *eviction* hay *ngân sách nén offline*.
- **Hệ thống inference consumer/edge**: llama.cpp, PowerInfer, ATSInfer, APEX — các hệ thống kỹ thuật xử lý OOM bằng cách từ chối/crash hoặc offload tĩnh; không có công trình nào xử lý việc thích ứng precision **giữa phiên** do áp lực VRAM **từ bên ngoài** (không phải do chính model) gây ra.

### 3. Khảo sát vấn đề (RQ1)
- Setup thực nghiệm: laptop RTX 4050/4060, model + bộ prompt cố định, các nguồn tải nền (Chrome nhiều tab, Discord chia sẻ màn hình, OBS ghi hình, demo game).
- Đo đạc: VRAM khả dụng theo thời gian (qua NVML polling), phương sai, tần suất/độ lớn của spike, tương quan với các hành vi người dùng phổ biến.
- Sản phẩm: một bộ "VRAM contention trace" nhỏ + thống kê mô tả — đây là đóng góp thực nghiệm có thể trích dẫn được, ngay cả trước khi đến phần đóng góp hệ thống.

### 4. Thiết kế Hệ thống (RQ2) — xem Phần 2 của tài liệu này để có kiến trúc đầy đủ
- VRAM Pressure Monitor
- Precision Controller (tầng chính sách)
- Paged, Mixed-Precision KV Cache Manager
- Tích hợp vào decode loop (MicroInfer, xây từ đầu — xem Phần 5)

### 5. Chính sách Precision theo Ngữ nghĩa (RQ3)
- Tính importance score theo block, suy ra từ attention.
- Ánh xạ importance → precision tier (FP16 / INT8 / INT4).
- Chu kỳ re-scoring và xử lý staleness khi chuyển chủ đề trong hội thoại nhiều lượt.
- Định nghĩa chính sách + pseudocode (xem mục 3.3 bên dưới).

### 6. Đánh giá NLP
- Tác vụ: coherence hội thoại nhiều lượt, QA văn bản dài (phân tầng theo vị trí câu trả lời), độ trung thực của tóm tắt.
- Bộ dữ liệu: MT-Bench hoặc bộ tự xây nhiều lượt; NarrativeQA hoặc bộ QA văn bản dài tiếng Việt/Anh tự xây; CNN/DailyMail hoặc VietNews.
- Các điều kiện so sánh: (a) không có contention (upper bound), (b) contention → OOM/crash (hiện trạng), (c) contention → uniform quantization, (d) contention → semantic-aware quantization (đề xuất).
- Metric: anchor-fact recall/coherence score (hội thoại), EM/F1 phân theo vị trí câu trả lời (QA), ROUGE-L/BERTScore (tóm tắt); cộng thêm metric hệ thống (latency P50/P99, throughput, tỷ lệ OOM).

### 7. Kết quả (khung, chưa có số liệu)
- Kết quả RQ1: biểu đồ khảo sát contention.
- Kết quả RQ2: độ trễ phát hiện, tỷ lệ false positive/negative của việc phát hiện áp lực, tỷ lệ tránh được OOM.
- Kết quả RQ3: metric theo từng tác vụ dưới mỗi điều kiện; phân tích tương quan giữa attention magnitude và mức suy giảm chất lượng.

### 8. Thảo luận & Hạn chế
- Phạm vi: 1 GPU, 1 model; khả năng tổng quát hóa qua các họ model/biến thể attention khác (MHA/GQA/MLA); overhead của việc tính importance score; engine (MicroInfer) là prototype nghiên cứu, không phải production-grade — nêu rõ đây là giới hạn phạm vi, không phải điểm yếu cần che giấu.

### 9. Kết luận & Hướng phát triển
- Mở rộng sang model hybrid attention/SSM; mở rộng chính sách sang cả weights (không chỉ KV cache); giám sát liên tục ở cấp OS như 1 service.

### Venue mục tiêu
- Chính: MLSys, EuroSys, các workshop track (ML for Systems @ NeurIPS/ICML), hoặc ACL/EMNLP *Industry/System-Demonstration track* (vì có phần đánh giá NLP).
- Dự phòng: arXiv preprint + GitHub repo trình bày kỹ càng — đủ để làm artifact mạnh cho CV kể cả khi chưa được accept.

---

## 2. Kiến trúc Hệ thống

### 2.1 Sơ đồ thành phần tổng quan (dạng văn bản)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Host Application                          │
│  (chat UI / CLI gọi MicroInfer để sinh văn bản)                   │
└───────────────────────────┬───────────────────────────────────────┘
                             │ generate(prompt, session_state)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Inference Orchestrator                       │
│  - Sở hữu decode loop                                             │
│  - Gọi VRAM Monitor trước/định kỳ trong lúc decode                │
│  - Gọi Precision Controller khi có pressure event                │
│  - Gọi Attention Scorer để làm mới importance map                 │
└───────┬───────────────────────┬───────────────────────┬──────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌───────────────────┐   ┌───────────────────────┐
│ VRAM Pressure  │     │ Attention Importance│  │ Precision Controller  │
│ Monitor        │     │ Scorer              │  │ (Policy Engine)        │
│ - NVML polling │     │ - đọc attention weight│  │ - map importance→tier │
│ - phát hiện    │     │   mỗi decode step    │  │ - hysteresis/cooldown │
│   ngưỡng/xu hướng│   │ - score theo block   │  │ - tạo re-quant plan    │
└───────┬────────┘     │ - decay/staleness    │  └──────────┬─────────────┘
        │              └──────────┬───────────┘             │
        │                         │                          │
        └───────────┬─────────────┴──────────────┬───────────┘
                     ▼                            ▼
        ┌─────────────────────────────────────────────────┐
        │        Paged Mixed-Precision KV Cache Manager     │
        │  - block table (page → precision tier)             │
        │  - kernel (de)quantize FP16 / INT8 / INT4           │
        │  - re-quantize tại chỗ cho block đã tồn tại         │
        └───────────────────────┬───────────────────────────┘
                                 ▼
                  ┌───────────────────────────┐
                  │   MicroInfer CUDA Kernels   │
                  │ (attention, GEMM, softmax)  │
                  │      — xây từ đầu           │
                  └───────────────────────────┘
```

### 2.2 Luồng dữ liệu mỗi decode step
1. Orchestrator yêu cầu token tiếp theo; trước khi dispatch attention kernel, nó kiểm tra 1 flag nhẹ do VRAM Monitor thread đặt (không blocking call trên hot path — monitor chạy trên thread/timer riêng, ví dụ mỗi 50–100ms).
2. Nếu pressure flag được set, Precision Controller được gọi **giữa** các decode step (không bao giờ giữa lúc kernel đang chạy) để tránh trạng thái cache không nhất quán.
3. Controller tham chiếu importance map hiện tại (làm mới mỗi `R` step, xem 3.3) và tạo ra **re-quantization plan**: danh sách `(block_id, tier_hiện_tại, tier_mục_tiêu)`.
4. KV Cache Manager áp dụng plan: với các block bị hạ cấp, quantize tại chỗ bằng kernel quantization của chính engine (xây ở Milestone 2, xem mục 4.1); với các block đủ điều kiện nâng cấp (khi áp lực giảm), khôi phục từ buffer lưu trữ nhỏ hoặc tính lại nếu không giữ buffer (trade-off có thể điều chỉnh, xem 3.4).
5. Decode tiếp tục với block table đã cập nhật.

### 2.3 Ranh giới Module (cho việc triển khai)
| Module | Trách nhiệm | Interface chính |
|---|---|---|
| `vram_monitor` | Poll NVML, duy trì thống kê free-memory dạng rolling, phát pressure event | `PressureEvent poll()`, `subscribe(callback)` |
| `attention_scorer` | Tích lũy attention mass theo block, xuất importance score | `update(attn_weights, block_ids)`, `get_scores() -> Dict[block_id, float]` |
| `precision_controller` | Map score + pressure level thành re-quant plan | `plan(scores, pressure_level, current_tiers) -> List[BlockAction]` |
| `kv_cache_manager` | Sở hữu paged block, thực thi (de)quantize, cập nhật block table | `requantize(block_id, target_tier)`, `alloc_block()`, `evict_block()` |
| `core_kernels` | Attention, GEMM, softmax, quantize-dequantize INT8/INT4 — **xây từ đầu cho dự án này** | primitive tối giản, test kỹ; không phụ thuộc inference engine bên ngoài |
| `eval_harness` | Chạy tác vụ NLP dưới mỗi điều kiện trong 4 điều kiện, thu metric | `run_condition(task, condition) -> Metrics` |
| `contention_simulator` | Bật/tắt tải giả lập chiếm GPU memory cho thí nghiệm có kiểm soát | `start_load(pattern)`, `stop_load()` |

---

## 3. Thiết kế Chi tiết

### 3.1 VRAM Pressure Monitor
- **Nguồn tín hiệu**: `nvidia-ml-py` (binding NVML) polling `nvmlDeviceGetMemoryInfo` trên background thread, độc lập với CUDA stream dùng cho inference.
- **Các mức pressure** (3 tier, đơn giản, dễ giải thích):
  - `GREEN`: VRAM free > `T_high` (ví dụ > 20% tổng)
  - `YELLOW`: `T_low` < VRAM free ≤ `T_high` (ví dụ 10–20%)
  - `RED`: VRAM free ≤ `T_low` (ví dụ < 10%) — nguy cơ OOM cận kề
- **Debounce/hysteresis**: yêu cầu mức tín hiệu duy trì `K` lần poll liên tiếp (ví dụ 3 lần × 50ms = 150ms) trước khi bắn transition event, để tránh thrashing khi có spike thoáng qua.
- **Đầu ra**: 1 event queue mà Orchestrator xử lý giữa các decode step; không bao giờ block decode loop.

### 3.2 Attention Importance Scorer
- **Cái gì cần tích lũy**: mỗi decode step, attention weight từ query token đến toàn bộ vị trí KV đã cache (lấy từ đầu ra softmax trung gian của attention kernel — vì kernel được xây từ đầu, expose cái này như 1 output debug/telemetry tùy chọn thay vì thêm overhead mặc định vào fast path).
- **Tổng hợp**: duy trì EWMA (exponentially-weighted moving average) theo block:
  `score[block] = α * mean_attn_to(block) + (1 - α) * score[block]_trước`
  với `α` tinh chỉnh (ví dụ 0.2) để cân bằng giữa độ nhạy và ổn định khi hội thoại chuyển chủ đề.
- **Layer weighting**: có thể gán trọng số cao hơn cho layer sâu (thực nghiệm cho thấy attention ở layer sâu tương quan nhiều hơn với mức độ quan trọng ngữ nghĩa); expose thành vector `w_layer` tunable, mặc định đồng đều cho bản đầu tiên, sau đó ablate.
- **Xử lý staleness**: khi phát hiện heuristic chuyển chủ đề (ví dụ lượt mới của user có độ overlap từ vựng/embedding thấp so với cửa sổ gần đây), buộc re-score toàn bộ thay vì chỉ dựa vào decay của EWMA.

### 3.3 Precision Controller (Policy Engine)
- **Đầu vào**: pressure level (`GREEN`/`YELLOW`/`RED`), importance score theo block, tier hiện tại theo block, tỷ lệ nén mục tiêu tương ứng với pressure level.
- **Chính sách (bản đầu, đơn giản, có thể ablate)**:
  1. Xếp hạng block theo importance score tăng dần (ít quan trọng nhất trước).
  2. Ở `YELLOW`: hạ cấp `X%` block thấp nhất xuống 1 tier (FP16→INT8), bỏ qua `N` token gần nhất (recency floor — ngữ cảnh gần gần như luôn quan trọng) và bất kỳ block nào được đánh dấu chứa anchor fact (nếu bật heuristic nhẹ này).
  3. Ở `RED`: mạnh tay hơn — hạ cấp `Y%` block thấp nhất (`Y > X`), có thể hạ 2 tier (FP16→INT4), hoặc đẩy sang CPU như phương án cuối trước khi từ chối/cắt bớt.
  4. Ở `GREEN` sau khi đã từng hạ cấp: chủ động nâng cấp lại các block về FP16 theo thứ tự importance, trong giới hạn ngân sách cho phép, để phục hồi chất lượng khi hết contention.
- **Pseudocode**:
```
function plan(scores, pressure, tiers, recency_floor):
    candidates = các block không nằm trong recency_floor, sắp xếp theo score tăng dần
    if pressure == RED:
        target_fraction = Y
        max_downgrade_steps = 2
    elif pressure == YELLOW:
        target_fraction = X
        max_downgrade_steps = 1
    else:
        return upgrade_plan(candidates_desc_by_score, tiers)  # phục hồi chủ động

    n = ceil(len(candidates) * target_fraction)
    actions = []
    for block in candidates[:n]:
        new_tier = downgrade(tiers[block], steps=max_downgrade_steps)
        actions.append((block, tiers[block], new_tier))
    return actions
```
- **Hysteresis cho chính policy**: không re-plan thường xuyên hơn mỗi `M` decode step (ví dụ 8–16) để giới hạn overhead re-quantization.

### 3.4 Paged Mixed-Precision KV Cache Manager
- Sở hữu paged KV cache xây từ đầu (xem Milestone 1, mục 4.1): mỗi page có field `precision_tier` trong entry block table, cùng với chỉ mục logical→physical thông thường.
- **Đường hạ cấp**: áp dụng kernel quantize INT8/INT4 của chính engine tại chỗ, giải phóng byte không dùng đến của precision cao hơn trả về allocator.
- **Đường nâng cấp**: 2 phương án cần cài đặt và so sánh:
  - *(a) Recompute-on-upgrade*: chạy lại forward pass của prefix liên quan cho block đó — tốn kém nhưng luôn đúng.
  - *(b) Retain-buffer*: giữ 1 bản shadow FP16 nhỏ (ở CPU hoặc dạng nén) cho các block vừa bị hạ cấp gần đây (LRU buffer có giới hạn) để việc nâng cấp chỉ là copy-back nhanh thay vì tính lại — có chi phí bộ nhớ riêng, đáng để ablate.
- **Tương tác với allocator**: đổi precision tier làm thay đổi kích thước byte của page; manager cần hỗ trợ resize/di chuyển page trong pool (hoặc over-provision slot theo từng tier để tránh fragmentation — đơn giản hơn cho bản triển khai đầu tiên).

### 3.5 Contention Simulator (cho thí nghiệm có kiểm soát)
- Một tiện ích độc lập, cấp phát/giải phóng GPU memory theo pattern có thể cấu hình (hàm bậc thang, sawtooth, spike ngẫu nhiên) để tái tạo lại trace thực nghiệm của RQ1 một cách xác định cho thí nghiệm RQ2/RQ3 — tách biệt "cơ chế có hoạt động không" khỏi "chờ 1 tab Chrome thật spike memory".

---

## 4. Hướng dẫn Triển khai "Vibe Coding" (làm việc với AI coding assistant)

Phần này viết sao cho bạn có thể giao từng phần, theo thứ tự, cho 1 AI coding assistant (ví dụ Claude Code) như 1 task có phạm vi rõ ràng — đủ nhỏ để kiểm chứng, đủ lớn để có ích. Vì engine được xây từ đầu, thứ tự build bắt đầu từ 1 lớp thấp hơn so với việc extend 1 engine có sẵn.

### 4.1 Thứ tự build đề xuất

**Milestone 0 — Đường inference tối thiểu chạy được (chưa có adaptivity)**
1. Forward pass cơ bản của 1 decoder-only transformer bằng CUDA/C++ (hoặc Python + custom kernel qua `pybind11`) cho 1 model mở nhỏ (0.5–1.5B tham số) — attention + GEMM FP16 thuần, chưa paging, dùng 1 buffer KV cache cố định kích thước. Mục tiêu: đúng, chưa cần tối ưu, verify với reference implementation của HuggingFace trên cùng prompt/seed.
2. Thêm KV cache **dạng paged** (precision cố định, chỉ FP16) — block table logical, physical page pool, cấp phát/giải phóng page. Verify: output giống hệt bản không paging.
3. Thêm kernel quantize/dequantize INT8 và INT4 cho block KV cache, dùng *tĩnh* trước (quantize 1 lần lúc cấp phát, chưa chuyển đổi runtime). Verify: mức suy giảm perplexity so với FP16 trên 1 tập held-out, khớp với kỳ vọng đã công bố của scheme quantization đã chọn (ví dụ gần tương đương mức suy giảm mà KIVI/KVQuant từng báo cáo).

**Milestone 1 — Contention & monitoring (chưa adaptivity, chỉ quan sát)**
4. **Contention Simulator** — độc lập, dễ test, mở đường cho thí nghiệm có kiểm soát sớm.
5. **VRAM Pressure Monitor** — bọc NVML polling, unit test với simulator.

**Milestone 2 — Adaptivity**
6. **Attention Importance Scorer** — móc vào attention kernel từ Milestone 0 để lấy score theo block; verify trước với 1 pattern attention giả lập đã biết trước (đưa vào 1 ma trận attention dựng sẵn, kiểm tra công thức tổng hợp) trước khi móc vào kernel thật.
7. **Precision Controller** — logic policy thuần, không CUDA; unit-test đầy đủ với score/pressure level giả lập.
8. **Re-quantization runtime trong KV Cache Manager** — mở rộng phần quantization tĩnh ở Milestone 0 bước 3 để hỗ trợ đổi tier **tại chỗ, giữa phiên**. Đây là phần rủi ro và mới nhất — làm đường hạ cấp trước (tái dùng kernel quantize tĩnh), đường nâng cấp sau.
9. **Kết nối Orchestrator** — nối decode loop với các phần trên mà không thêm synchronization stall; profile bằng `nsys`/`ncu` trước/sau để xác nhận monitor không thêm overhead đáng kể mỗi step.

**Milestone 3 — Đánh giá**
10. **Eval harness** — dataset tác vụ + metric + bộ chạy thí nghiệm 4 điều kiện.

Thứ tự này quan trọng: chỉ riêng Milestone 0 (1 đường inference paged+quantized xây từ đầu, đúng) đã là 1 artifact kỹ thuật đáng kể, có thể trình bày độc lập — coi đây là 1 checkpoint có thể dừng lại và vẫn có thứ để show, trước khi cam kết làm toàn bộ hệ thống adaptive.

### 4.2 Cách prompt cho từng module (ví dụ với Precision Controller)
Khi giao 1 module cho AI coding assistant, cung cấp:
- **Interface** (function signature từ bảng ở mục 2.3).
- **Pseudocode** (3.3) như 1 spec, không phải code để copy nguyên — yêu cầu nó implement, rồi viết unit test bao phủ: hành vi hysteresis, loại trừ recency floor, thứ tự nâng cấp ở trạng thái GREEN, và edge case (danh sách block rỗng, tất cả block đã ở tier thấp nhất).
- Yêu cầu rõ ràng **giữ module này không phụ thuộc CUDA/NVML** để nó luôn unit-test được độc lập — đây là thực hành tốt nên nói trước (tách biệt logic policy khỏi I/O phần cứng).

### 4.3 Kỷ luật kiểm chứng (quan trọng cho 1 research artifact, nhất là khi xây từ đầu)
- Với mỗi module, yêu cầu assistant sinh **cả implementation lẫn 1 test đúng đắn nhỏ** trước khi tích hợp — điều này quan trọng hơn so với phát triển app thông thường, vì bug đúng đắn âm thầm trong đường re-quantization có thể tạo ra văn bản "trông hợp lý nhưng sai" (khó phát hiện bằng mắt thường).
- Với mọi kernel CUDA mới (đặc biệt Milestone 0, vì không có implementation tham chiếu sẵn có để dựa vào), luôn validate bằng số với 1 reference PyTorch/HuggingFace trên cùng seed/prompt cố định trước khi tin vào bất kỳ con số benchmark nào — 1 benchmark cho thấy "tăng tốc" từ 1 kernel bị lỗi là 1 kiểu lỗi thường gặp cần đề phòng rõ ràng, và dễ mắc phải hơn khi xây từ đầu so với khi extend code đã biết đúng.
- Duy trì 1 **benchmark log** liên tục (config, phần cứng, git commit hash, kết quả) ngay từ phiên bản chạy được đầu tiên — đây sẽ trở thành experiment log của paper, giúp tránh phải suy ra lại kết quả sau này.

### 4.4 Cấu trúc repo đề xuất
```
adaptive-kv-inference/
├── core_kernels/              # Milestone 0: attention, GEMM, paged cache, static quant — xây từ đầu
│   └── tests/                 # validate số học với reference PyTorch/HF
├── contention_simulator/
├── vram_monitor/
├── attention_scorer/
├── precision_controller/
│   └── tests/                 # unit test logic thuần, không cần GPU
├── kv_cache_manager/          # re-quantization runtime trên nền paged cache của core_kernels
├── orchestrator/               # tích hợp decode loop
├── eval/
│   ├── datasets/               # hội thoại / QA văn bản dài / tóm tắt
│   ├── metrics/
│   └── run_experiment.py       # tạo bảng so sánh 4 điều kiện
├── experiments/logs/          # benchmark log dạng append-only (xem mục 4.3)
└── paper/                     # outline, hình ảnh, bản nháp (nguồn của tài liệu này)
```

---

## 5. Kiến thức Cần có

### 5.1 CUDA/GPU Programming (nền tảng bắt buộc)

**Mô hình lập trình CUDA**
- Thread/Block/Grid hierarchy: warp (32 thread, SIMT) là khái niệm quan trọng nhất — mọi tối ưu hiệu năng xoay quanh nó
- Cách tính thread index (`blockIdx`, `threadIdx`, `blockDim`, `gridDim`) để map dữ liệu
- Warp divergence: khi thread trong cùng warp rẽ nhánh `if/else` khác nhau → chạy tuần tự, giảm hiệu năng — quan trọng khi viết causal masking
- Occupancy: tỷ lệ warp active/SM, bị giới hạn bởi register/shared memory/thread per block; occupancy cao không đồng nghĩa nhanh nhất

**Memory Hierarchy**
- Register (nhanh nhất, per-thread) → Shared memory (rất nhanh, per-block) → L2 Cache → Global memory/HBM (chậm nhất, bottleneck chính)
- Memory coalescing: thread trong 1 warp truy cập địa chỉ liền kề → gộp giao dịch bộ nhớ — áp dụng liên tục khi thiết kế layout page của paged KV cache
- Roofline model: kernel bị giới hạn bởi compute hay memory bandwidth? Arithmetic Intensity = FLOPs/Bytes; attention decode (batch=1) thường memory-bound, GEMM lớn thường compute-bound

**Kernel cần viết và tối ưu**
- Tiled GEMM: chia ma trận thành tile, load vào shared memory để giảm truy cập global memory
- Online softmax: tính incrementally, cập nhật max/sum khi duyệt qua từng block — core của Flash Attention, bắt buộc hiểu kỹ
- Reduction: dùng warp shuffle (`__shfl_down_sync`) để tổng hợp giá trị nhanh hơn shared memory + syncthreads

**Profiling Tools**
- Nsight Systems (nsys): xem timeline tổng thể, phát hiện gap/idle time, kernel launch overhead
- Nsight Compute (ncu): phân tích sâu 1 kernel — occupancy thực tế, memory throughput, warp stall reason (`sm__throughput`, `dram__throughput`, `achieved_occupancy`)

**Kết nối CUDA với Python**
- Viết kernel .cu → compile thành shared library → expose qua pybind11
- Quản lý memory giữa PyTorch tensor và CUDA kernel qua `torch::Tensor` C++ API hoặc `.data_ptr()`

### 5.2 Transformer Architecture & Inference Internals

**Self-Attention chi tiết**
- `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V` — chia `sqrt(d_k)` để tránh dot-product quá lớn làm bão hòa softmax
- Multi-head: chia `d_model` thành `h` head, mỗi head học pattern khác nhau — layout tensor (reshape/transpose) là nguồn lỗi phổ biến
- Causal masking: thường dùng offset/index thay vì tạo mask matrix tường minh để tiết kiệm bộ nhớ

**Biến thể attention (ảnh hưởng thiết kế KV cache)**
- MHA: mỗi head có K/V riêng — cache lớn nhất
- MQA: tất cả head chia sẻ 1 K/V — cache nhỏ nhất, giảm chất lượng
- GQA: trung gian, nhóm Q head dùng chung K/V head (LLaMA-2, Mistral) — **nên chọn model dùng GQA** để paper thực tế hơn

**Position Encoding**
- RoPE: chuẩn hiện tại của model mở — áp dụng lên Q/K trước khi tính attention; cần kiểm tra sai số khi cache bị quantize/dequantize ảnh hưởng đến vector đã áp RoPE

**KV Cache — Cơ chế và Công thức**
```
KV cache size = 2 (K và V) × num_layers × num_kv_heads × head_dim × seq_len × batch_size × bytes_per_element
```
- Ví dụ: model 7B, GQA 8 KV head, head_dim 128, 32 layer, FP16 → ~0.5MB/token → context 32K đã chiếm 16GB

**Paged Attention / PagedAttention**
- Ý tưởng từ vLLM (lấy cảm hứng virtual memory OS): chia cache thành page cố định (ví dụ 16 token/page), block table ánh xạ logical→physical block
- Đây chính là cơ chế sẽ được mở rộng thêm field `precision_tier`
- Trade-off page size: nhỏ → ít lãng phí, nhiều overhead quản lý; lớn → ngược lại

**Quantization cơ bản**
- Symmetric (`q = round(x/scale)`) vs Asymmetric (`q = round(x/scale) + zero_point`) — KV cache thường dùng asymmetric
- Per-tensor / per-channel / per-token — per-token thường dùng cho KV cache vì magnitude giữa token khác nhau nhiều
- Outlier: giá trị bất thường lớn kéo dãn scale, giảm độ chính xác phần còn lại — vấn đề nổi tiếng trong quantization LLM
- Cần đo perplexity degradation INT8/INT4 vs FP16 — con số bắt buộc phải có trong paper

**Attention Weight — Ý nghĩa làm Importance Score**
- Attention weight cao không hoàn toàn đồng nghĩa "quan trọng ngữ nghĩa" — cần tự kiểm chứng bằng thực nghiệm (có tranh luận "attention is not explanation")
- Attention sink: token đầu tiên (BOS) thường nhận attention cao bất kể ngữ nghĩa — cần loại trừ, nếu không importance map sẽ luôn ưu tiên giữ token đầu 1 cách giả tạo

### 5.3 Hệ điều hành & Hệ thống

**Quản lý bộ nhớ GPU**
- VRAM quản lý bởi driver, mỗi CUDA context giữ 1 phần VRAM riêng
- Consumer GPU **không có cơ chế cô lập/đảm bảo VRAM** giữa các process (khác MIG/MPS trong datacenter) — đây là điểm mấu chốt biện minh cho toàn bộ đề tài
- Unified Memory (`cudaMallocManaged`) cho phép oversubscribe nhưng thường chậm hơn do page fault qua PCIe — cần giải thích rõ vì sao không chọn hướng này

**NVML API**
- `nvmlDeviceGetMemoryInfo()`: total/free/used tại thời điểm gọi — API chính cho VRAM Monitor
- `nvmlDeviceGetUtilizationRates()`: % dùng GPU compute — phân biệt "bận vì compute" vs "bận vì memory"
- Python binding: `nvidia-ml-py` (import `pynvml`)
- Hạn chế: polling-based (không event-driven) → luôn có độ trễ; trade-off giữa interval ngắn (nhanh, tốn CPU) và dài (tiết kiệm, chậm) — cần ablate trong paper

**Concurrency**
- Python: `threading` (CUDA call thường release GIL) hoặc `asyncio`
- C++: `std::thread`, `std::atomic` để chia sẻ pressure state không cần lock nặng
- Race condition cần tránh: Controller đọc score trong khi Scorer đang cập nhật — dùng mutex hoặc double-buffering
- CUDA Streams: re-quantize có thể chạy trên stream riêng, overlap với compute decode step tiếp theo (tối ưu nâng cao, để ở Future Work)

**Control Systems cơ bản**
- Hysteresis: không phản ứng ngay khi vượt ngưỡng, cần tín hiệu ổn định 1 khoảng thời gian — tránh thrashing
- Feedback loop: đo (VRAM) → quyết định (policy) → hành động (requantize) → đo lại
- Rate limiting: giới hạn tần suất hành động tối đa (không re-plan quá 1 lần mỗi M step)

**Đo lường Benchmark hệ thống**
- Latency percentile (P50/P99) quan trọng hơn average vì phản ánh worst-case
- Throughput: đo riêng prefill (token/s xử lý prompt) và decode (token/s sinh mới) vì đặc tính khác nhau

### 5.4 NLP Evaluation Methodology

**Metric**
- Perplexity: `PPL = exp(-1/N * Σ log P(token_i|context))` — dùng làm sanity check trước khi vào task-specific eval
- ROUGE-L: overlap từ vựng bề mặt (LCS-based) — hạn chế: không hiểu ngữ nghĩa, paraphrase đúng vẫn bị điểm thấp
- BERTScore: similarity ngữ nghĩa qua embedding — khắc phục hạn chế ROUGE; nên report song song cả 2
- EM/F1: chuẩn từ SQuAD cho QA extractive

**Needle in a Haystack**
- Chèn 1 fact cụ thể vào 1 vị trí xác định trong văn bản dài, hỏi lại — đo theo vị trí và độ dài context
- Phù hợp trực tiếp với đề tài: kiểm soát needle nằm trong block nào của paged cache, đo accuracy khi block đó bị hạ precision
- Với tiếng Việt: tự ghép văn bản Wikipedia + chèn fact tự soạn (bộ có sẵn chủ yếu tiếng Anh)

**LLM-as-Judge**
- Dùng LLM mạnh hơn chấm điểm theo rubric (thang 1-5 cho coherence/relevance/factuality)
- Hạn chế cần nêu rõ: self-preference bias, position bias (cần đảo vị trí khi so sánh A/B), không thay thế hoàn toàn human eval — nên có thêm 1 lượng nhỏ human eval đối chiếu

**Multi-turn Dialogue Evaluation**
- Cấu trúc MT-Bench: hội thoại 2+ lượt, lượt sau phụ thuộc ngữ cảnh lượt trước
- Điều chỉnh riêng: thêm câu hỏi anchor-fact recall — lượt đầu cho thông tin cụ thể, lượt sau (xa đủ để nằm trong vùng có thể bị hạ precision) hỏi lại

**Thiết kế Thí nghiệm**
- Kiểm soát biến: khi so sánh uniform vs semantic-aware, phải đảm bảo cùng tổng tỷ lệ nén trung bình, chỉ khác cách chọn block để nén
- Statistical significance: chạy nhiều seed/mẫu để có confidence interval; paired comparison (paired t-test) khi có thể
- Ablation study: bỏ từng thành phần policy (recency floor, layer weighting, re-scoring định kỳ) để chứng minh đóng góp riêng lẻ — chuẩn bị sẵn từ đầu, không đợi reviewer yêu cầu

### 5.5 Kỹ năng Nghiên cứu

**Đọc paper hiệu quả**
- Thứ tự: Abstract → Introduction → Kết quả/Kết luận → Hình/Bảng → Method chi tiết
- Kỹ thuật "3-pass reading": pass 1 (5 phút, ý chính), pass 2 (1 giờ, hiểu method ở mức khái niệm), pass 3 (sâu, chỉ khi cần implement lại)

**Thiết kế thí nghiệm có kiểm soát biến**
- Nguyên tắc "1 biến thay đổi tại 1 thời điểm", áp dụng xuyên suốt cả phần Systems lẫn NLP eval
- Luôn có baseline rõ ràng trước khi thêm biến mới

**Viết Related Work**
- Không liệt kê máy móc — mỗi đoạn kết ở câu nêu rõ **gap**: "tuy nhiên, các công trình này giả định..., trong khi bối cảnh của chúng tôi..."
- Nhóm công trình theo chủ đề, không theo thời gian

**Thống kê cơ bản**
- Confidence interval, standard error; vì sao 1 lần chạy không đủ tin cậy (đặc biệt hệ thống có tính ngẫu nhiên như GPU scheduling)
- Paired vs unpaired test

**Viết Abstract/Introduction**
- Công thức: bối cảnh chung → vấn đề cụ thể chưa giải quyết → đóng góp → kết quả nổi bật (số liệu cụ thể) → ý nghĩa rộng hơn
- Tránh overclaim — nêu đúng phạm vi (single-user, single-GPU) ngay từ đầu

### 5.6 Công cụ hỗ trợ
- Git/GitHub để quản lý code + benchmark log
- PyTorch/HuggingFace Transformers — dùng làm reference validate kernel tự viết
- LaTeX (nếu venue yêu cầu format riêng) hoặc Markdown (arXiv/preprint)
