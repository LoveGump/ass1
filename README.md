# 作业一 文本分类

在同一份 NYT 新闻数据划分上比较三种词袋表示、三种 100 维词向量表示和 BERT 微调。原始要求位于 `作业一要求.docx`，原始数据位于 `HW-1/`。

## 提交内容

- `output/pdf/report.pdf`：中文实验报告（由真实运行结果生成）。
- `src/`：数据处理、传统模型、BERT 完整实现。
- `results/`：固定划分、数据统计、实验参数、Accuracy / Macro-F1、混淆矩阵和逐条预测。
- `docs/report.md`：可直接在 GitHub 阅读的报告。
- `pyproject.toml`、`uv.lock`：依赖及精确版本锁定。
- `tests/`：划分、表示、未知词和不平衡指标测试。

任务要求中 Task 1 总述为“三种”，但只列出 Binary Bag of Words 和 Word Frequency。本仓库完成两项明确要求，并将 **TF-IDF 作为补充第三项**；没有将其误写为原文明确列出的内容。

## 实测结果

全部使用同一份 1,152 条 NYT 测试集；BERT 已完整微调 3 轮，max_length=64。七组指标均通过逐条预测重算审计，5 项关键测试通过。

| 方法 | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Binary BoW | 0.9905 | 0.9754 |
| Word Frequency | 0.9887 | 0.9738 |
| TF-IDF | 0.9922 | 0.9812 |
| GloVe 100d | 0.9878 | 0.9715 |
| Word2Vec AG 100d | 0.9835 | 0.9581 |
| Word2Vec NYT 100d | 0.9896 | 0.9731 |
| BERT | 0.9809 | 0.9589 |

[PDF 实验报告](output/pdf/report.pdf) · [在线阅读报告](docs/report.md) · [完整数值](results/summary.csv)

## 环境安装

需要 Python 3.12、[uv](https://docs.astral.sh/uv/)、Git 和 curl。Windows / Linux 默认安装 CUDA 12.4 的 PyTorch 2.6.0，适用于支持该版本的 NVIDIA 驱动；没有 GPU 时代码自动使用 CPU，但 BERT 会慢得多。macOS 使用其平台对应的 PyTorch 包。

```powershell
git clone https://github.com/LoveGump/ass1.git
cd ass1
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
uv sync --frozen --python 3.12
```

Linux / macOS 可使用 `UV_CACHE_DIR=.uv-cache uv sync --frozen --python 3.12`。首次下载包含约 2.4 GiB 的 Windows PyTorch 安装包、约 440 MB 的 BERT 权重和约 862 MB 的 GloVe 压缩文件，请预留磁盘空间。训练输出、缓存和预训练权重保留在本地，不提交 Git。

若网络需要代理，可在当前终端设置 `HTTPS_PROXY` / `HTTP_PROXY` 为**自己实际使用的代理地址**。无法直连 Hugging Face 时，可设置 `HF_ENDPOINT=https://hf-mirror.com`；下载器校验固定版本的权重哈希。仓库不预设个人代理配置。

若官方依赖源较慢，可使用导出的同版本依赖清单（其中 CUDA 包仍需从 PyTorch 官方获取）：

```powershell
uv pip sync --python .venv/Scripts/python.exe --require-hashes requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match
```

此时可直接调用 .venv 中的 Python 执行下述脚本，避免重复同步；Linux 对应路径为 .venv/bin/python。

## 完整运行

在仓库根目录运行：

```powershell
uv run --frozen python scripts/run_all.py
```

该命令依次下载并校验权重、生成或验证固定划分、测试、训练七组模型、核验预测和指标、生成报告。重跑会重新训练并覆盖对应结果。请勿将不同数据文件或不同划分的结果混合；加载时会验证数据 SHA-256。

也可以分步运行：

```powershell
uv run --frozen python scripts/download_assets.py
uv run --frozen python -m src.common
uv run --frozen python -m pytest -q
uv run --frozen python -m src.classical --methods binary frequency tfidf
uv run --frozen python -m src.classical --methods w2v_ag w2v_nyt
uv run --frozen python -m src.classical --methods glove --glove artifacts/glove.6B.100d.txt
uv run --frozen python -m src.bert --model-dir artifacts/bert-base-uncased
uv run --frozen python scripts/audit_results.py
uv run --frozen python scripts/build_report.py
```

默认下载 Stanford 官方 glove.6B.zip，检查文件长度和解压 CRC，只提取 glove.6B.100d.txt，并记录 SHA-256。也支持通过下载器 --glove-source gensim 获取其无损格式转换版并检查官方公布的 MD5；使用时将 --glove 指向该 .gz 文件。两者均为 400,000 个词、100 维，来源为 Wikipedia 2014 + Gigaword 5（6B tokens）。BERT 固定版本为 86b5e0934494bd15c9632b12f734a8a67f723594，下载器验证权重 SHA-256。

## 使用已训练模型预测

训练后可直接对一条新闻预测类别（仅加载本仓库自己生成的模型文件）：

```powershell
uv run --frozen python -m src.predict --method tfidf --text "The team won the football championship after a dramatic final."
uv run --frozen python -m src.predict --method bert --text "The senator announced a new election campaign."
```

七种方法均支持 --method；BERT 的此单条预测入口使用 CPU，训练入口仍自动使用可用 GPU。

## 实验约定

1. 种子为 42。两次分层随机打乱划分得到训练 / 验证 / 测试集，比例约 80% / 10% / 10%；整数样本数导致微小舍入差。行号从 CSV 表头后的第一个样本起按 0 编号，保存在 `results/split.json`。
2. Binary、Frequency、TF-IDF 使用相同正则分词，保留英文词及词内撇号、统一小写。词表及 IDF 只在 NYT 训练集拟合，不过滤停用词、不截断词表。
3. GloVe 和 Word2Vec 使用重复词计权的有效词向量算术平均；跳过 OOV，全 OOV 文档为零向量。100 维特征的标准化仅在训练集拟合。
4. Word2Vec 用 Skip-gram、窗口 5、负采样 5、最低词频 5、10 轮、单 worker 和稳定词哈希。AG 实验用全部 AG 文本；NYT 实验只用 NYT **训练集**文本。
5. 六种传统方法均为 Logistic Regression；仅用验证集 Macro-F1 在 `C ∈ {0.1, 1, 10}` 中选择，平局取较小 C。不用测试集调参，不把验证集重新并入训练集。
6. BERT 全参数微调，**max_length=64，完整训练 3 epochs**。批量 16，梯度累积 2，有效批量约 32；AdamW，学习率 `2e-5`，权重衰减 0.01（排除 bias / LayerNorm），10% warmup，线性衰减，梯度裁剪 1.0。CUDA 使用 FP16。三轮完成后选验证集 Macro-F1 最高的 checkpoint，再评估测试集。
7. 原数据存在重复新闻。为遵循原始随机划分任务不擅自删除；报告明确列出跨集合重复数量，并补充排除与训练集完全相同文本后的测试指标。不据此把主表结果宣称为对完全未见新闻的无偏估计。
8. 保存版本、种子和逐条预测以便审计；不同硬件、驱动和库的浮点差异仍可能导致小幅变化。

## 文件结构

```text
HW-1/                       原始 CSV
src/common.py               固定划分、分词、指标、结果保存
src/classical.py            六组传统实验
src/bert.py                 BERT 三轮微调
scripts/download_assets.py  下载与校验
scripts/audit_results.py    从逐条预测重算全部指标
scripts/build_report.py     从结果生成 PDF、图表和 Markdown
scripts/run_all.py          完整流程
tests/test_pipeline.py      关键行为测试
results/                    可审计实验结果
docs/                       报告与图表
output/pdf/report.pdf       最终 PDF
artifacts/                  本地模型及下载文件（不入 Git）
```

报告生成使用 ReportLab；Windows 默认嵌入宋体，其他系统自动使用 ReportLab 内置中文 CID 字体，也可通过 `REPORT_FONT` 指定中文 TTF / TTC 字体路径。无需 LaTeX 或 Word。

## 参考

- [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)
- [Gensim 数据来源和校验值](https://github.com/RaRe-Technologies/gensim-data/blob/master/list.json)
- [BERT 模型卡](https://huggingface.co/google-bert/bert-base-uncased)
- [BERT 论文](https://aclanthology.org/N19-1423/)
