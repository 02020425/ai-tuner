# AI Tuner

一键修音 Web 服务 — 上传跑调的音频，自动修正音准和节奏。

## 处理管线

```
用户音频
  │
  ▼
[节奏修正] ← none / 节拍网格 / 参考音频 DTW 对齐
  │
  ▼
[音高修正] ← DSP 音阶 / DSP 参考 / AI HiFi-GAN
  │
  ▼
[后处理]   ← pitch 中值滤波 + 音量归一化
  │
  ▼
输出
```

## 三种修音模式

| 模式 | 原理 | 适合场景 |
|------|------|---------|
| **DSP 音阶** | 自动检测调性，将每个音拉到最近的调内音（pyrubberband） | 不知道原唱、没有参考音频 |
| **DSP 参考音频** | DTW 对齐原唱音高，拉到对应的目标音高（pyrubberband） | 有原唱/标准版本，修得更准 |
| **AI 神经网络** | HiFi-GAN vocoder 按目标音高重建波形，音高和共振峰一起生成 | 跑调严重也能保持音色自然 |

## 节奏修正

商业软件（Melodyne、Auto-Tune）的做法是级联：先修节奏，再修音高。本项目同样放在音高修正之前，不改变现有管线。

| 模式 | 原理 |
|------|------|
| **节拍网格** | 检测 BPM + onset，量化到最近 16 分音符网格 |
| **参考音频** | DTW 对齐用户音频到原唱的演唱时机 |

## 后处理润色

- **Pitch smoothing**：中值滤波去掉修正后的孤立跳变和颤音 overshoot，只对 voiced 帧生效
- **音量归一化**：RMS 响度归一化到 -18dBFS，带 soft limiter 防止削波

## 技术栈

- **音高检测**: pYIN (librosa)
- **调性识别**: Krumhansl-Schmuckler 算法
- **时间对齐**: DTW (Dynamic Time Warping) — 音高对齐 + 节奏对齐
- **DSP 音高搬移**: Rubber Band Library (pyrubberband)
- **AI 音频重建**: HiFi-GAN vocoder with pitch conditioning（PyTorch）
- **训练数据生成**: WORLD vocoder — 分离 F0/频谱包络/非周期性，独立偏移
- **后端**: FastAPI (Python)
- **前端**: 原生 HTML/CSS/JS，零依赖

## 项目结构

```
ai-tuner/
├── backend/
│   ├── app.py                 # FastAPI 服务入口（5 个 API）
│   ├── tuner.py               # 3 条修音管线 + 节奏预处理 + 后处理
│   ├── pitch_detector.py      # 音高检测 & 调性识别
│   ├── alignment.py           # DTW 音高对齐
│   ├── rhythm_corrector.py    # 节奏修正（节拍网格 + DTW 参考对齐）
│   ├── neural_vocoder.py      # HiFi-GAN 推理模型（加载 weights 用）
│   └── requirements.txt
├── scripts/
│   ├── generate_training_data.py  # WORLD vocoder 把干净人声搞跑调 → 构造配对训练数据
│   ├── hifi_gan.py                # HiFi-GAN 模型定义（Generator + Discriminator + Loss）
│   ├── train.py                   # 训练脚本（对抗训练微调）
│   └── requirements_train.txt
├── frontend/
│   └── index.html              # 三栏 UI（DSP 修音 / 参考音频 / DSP vs AI 对比）
├── data/clean/                 # ← 干净人声放这里
├── data/training/              # ← generate_training_data.py 生成到这里
├── models/                     # ← tuner.pth 训练好放这里
├── checkpoints/                # 训练断点保存
├── outputs/                    # 处理后音频下载
└── uploads/                    # 用户上传临时目录
```

## 快速开始

### 仅推理（使用已训练好的模型）

```bash
pip install -r backend/requirements.txt
cd backend && uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

打开 http://localhost:8000

### 完整流程（从训练到部署）

```bash
# 1. 安装训练依赖
pip install -r scripts/requirements_train.txt

# 2. 把干净人声 WAV 文件放到 data/clean/

# 3. 生成配对训练数据（WORLD vocoder 故意搞跑调）
python scripts/generate_training_data.py \
    --input_dir data/clean/ \
    --output_dir data/training/ \
    --pairs_per_file 50 \
    --formant_shift_ratio 0.4

# 4. 训练模型
python scripts/train.py \
    --data_dir data/training/ \
    --checkpoint_dir checkpoints/ \
    --batch_size 8 \
    --num_epochs 100

# 5. 模型自动导出到 models/tuner.pth

# 6. 启动服务
cd backend && uvicorn app:app --host 0.0.0.0 --port 8000
```

## API

### `POST /api/tune/scale`
DSP 音阶模式修音。
- `audio` (file): 待修正的音频
- `key` (string, optional): 调性，默认 "auto"
- `scale` (string, optional): 音阶类型，默认 "major"
- `strength` (float, optional): 修正力度 0.0-1.0
- `rhythm` (string, optional): 节奏修正，默认 "none"，可选 "grid" / "reference"
- `rhythm_ref` (file, optional): 节奏参考音频（rhythm=reference 时需要）
- `rhythm_bpm` (float, optional): 目标 BPM（rhythm=grid 时可选，留空自动检测）

### `POST /api/tune/reference`
DSP 参考音频模式修音。
- `audio` (file): 待修正的音频
- `reference` (file): 参考音频（原唱）
- `strength` (float, optional): 修正力度 0.0-1.0
- `rhythm` (string, optional): 节奏修正，默认 "none"，可选 "reference"（用参考音频同步节奏）

### `POST /api/tune/neural`
AI 神经网络修音（需 models/tuner.pth）。
- `audio` (file): 待修正的音频
- `key` (string, optional): 默认 "auto"
- `scale` (string, optional): 默认 "major"
- `strength` (float, optional): 修正力度 0.0-1.0
- `rhythm` (string, optional): 同 scale 端点

### `POST /api/tune/compare`
同时运行 DSP 和 AI，返回对比结果。
- 参数同 scale 端点

### `GET /api/download/{filename}`
下载处理后的音频。

### `GET /api/scales` · `GET /api/keys` · `GET /api/rhythms`
返回可用的音阶类型、调性、节奏修正模式列表。

## DSP vs 神经网络

| | DSP（pyrubberband） | 神经网络（HiFi-GAN） |
|---|---|---|
| **原理** | 检测音高 → 频域拉伸搬移 | 提取 mel 频谱 + 目标音高 → vocoder 重建波形 |
| **共振峰** | 搬移音高分开了共振峰，跑调远了音色变形 | 音高和共振峰一起生成，音色自然 |
| **跑调严重时** | 音色失真明显 | 能重新"唱"出来 |
| **推理速度** | 毫秒级，纯 CPU | 秒级，GPU 加速 |
| **依赖** | 数学库，几十 MB | 模型权重，~50 MB |
| **部署** | 零模型，直接跑 | 本地加载 .pth，不调任何外部 API |

### 为什么不能调大模型 API

修音需要的是专用音频神经网络（vocoder），不是 LLM。LLM 的输入输出是文字，无法理解音频波形、音高、共振峰。修音用的音频模型是本地部署的——没有现成的 API 可以调。

### 为什么神经网络能解决共振峰问题

DSP 修音把人声拆成"音高"和"音色"两个独立的东西操作，但人声里两者是耦合的。神经 vocoder 不拆，直接从 mel 频谱 + 目标音高重建整个波形，音高和共鸣特征一起生成，不存在"移了 A 丢了 B"。

### 工程化难点

- **配对训练数据缺失**：现实中不存在同一人"跑调版"和"正确版"的配对数据。取巧方案是用 DSP 把干净人声故意搞跑调，反向构造训练数据。
- **域差距**：DSP 合成的跑调数据与真实跑调有分布差异。改用 WORLD vocoder 分离 F0 和频谱包络，让共振峰也按比例偏移（`formant_shift_ratio=0.4`），缩小域差距。
- **实时性**：neural vocoder 比 Rubber Band 慢 1-2 个数量级，单次推理数秒，不适合实时修音流。
- **可控性**：神经网络是黑盒，可能把刻意的滑音当成跑调修掉。DSP 每一步可解释可调试。
- **并发**：多用户同时推理对 GPU 显存压力大。

## 训练方案

**基于 RVC 架构改造**（已落地为 scripts/train.py）：

RVC 有三个模块：内容编码器（"在唱什么"）+ 音高编码器（"唱了多高"）+ HiFi-GAN 解码器（合并重建波形）。修音只需改动一处——把音高编码器提取的跑调音高替换成修正后的目标音高，内容和音色不变。

**训练数据**：`generate_training_data.py` 用 WORLD vocoder 分离 F0/频谱包络/非周期性，独立偏移 F0 并联动共振峰来制造更真实的跑调数据。任何干净的人声都能用来生成无限量训练对。

**算力需求**：

| 环节 | 最低配置 | 推荐配置 |
|------|---------|---------|
| 微调 HiFi-GAN | RTX 3060 (12GB) | RTX 4090 (24GB) |
| 仅推理（部署） | CPU 也行 | GTX 1060+ |

**训练周期**：家用显卡 1-3 天。

## 设计讨论（2026-05-22）

### 节奏修正

商业软件（Melodyne、Auto-Tune）的做法是级联而非统一——先时间拉伸修节奏，再音高偏移修音。两者解耦。已落地为 `backend/rhythm_corrector.py`，支持节拍网格量化和参考音频 DTW 对齐两种模式。

### WORLD 替代 pyrubberband（训练数据生成）

- WORLD 把声音拆成 F0、频谱包络、非周期性三个独立组件
- 训练数据生成时独立偏移 F0，共振峰按 `formant_shift_ratio` 联动偏移
- 比 pyrubberband 频域搬移更接近真实跑调（真实跑调时共振峰也跟着偏移）
- pyrubberband 继续用于 DSP 推理管线，两者各司其职

### 真跑调数据怎么用？（不能用 AI 输出当 target）

纯蒸馏会逐代累积 vocoder artifacts（相位不自然、高频细节缺失），质量越来越差。三个替代方案：

| 方案 | 做法 | 特点 |
|------|------|------|
| **A - 判别器重定向** | 生成器用合成数据学修音，判别器用真实干净人声学音质 | 改动最小，两个信号各司其职 |
| **B - mel 一致性约束** | 约束"除了音高，其他别动"，self-supervision | 不需要任何 target 音频 |
| **C - 两阶段训练（推荐）** | 预训练学修音映射 → 微调时判别器学真实音质 + mel 一致性正则 | 优雅，不把 AI 输出当 ground truth |

## 模型状态

| 文件 | 训练数据 | 适用范围 | 状态 |
|------|---------|---------|------|
| `models/tuner_female_only.pth` | OpenCpop (1 女声) | 仅女声，男声会出电音 | 已废弃 |
| `models/tuner.pth` | 待训练 | 男女通用 | TODO |

> **已知问题 2026-05-25**：仅用 OpenCpop 一个女歌手训练的模型对男声无效。AI vocoder 换到音域/音色不匹配的输入时输出噪声。需要多歌手数据重新训练。

## TODO

- [x] ~~节奏修正（级联在音高修正之前）~~
- [x] ~~WORLD vocoder 替代 pyrubberband 生成训练数据~~
- [x] ~~Pitch smoothing + 音量归一化后处理~~
- [x] ~~全功能 Web UI（4 Tab：DSP 音阶 / 参考音频 / AI 修音 / DSP vs AI）~~
- [x] ~~MP3/WAV/FLAC 多格式支持~~
- [ ] **多歌手训练**：下载 M4Singer / OpenSinger，男女声混合 WORLD 生成配对数据，重训通用模型
- [ ] 两阶段训练（真跑调数据微调、mel 一致性约束）
- [ ] 人声/伴奏分离预处理（demucs），直接上传带伴奏的音频
- [ ] 请求队列管理，避免并发推理撑爆显存
- [ ] 流式处理（WebSocket），边录边修
- [ ] 前端波形对比（修前 vs 修后可视化）
- [ ] 移动端适配
