"""Build the Chinese PDF and Markdown report exclusively from audited run outputs."""
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["binary", "frequency", "tfidf", "glove", "w2v_ag", "w2v_nyt", "bert"]
NAMES = {"binary": "Binary BoW", "frequency": "Frequency", "tfidf": "TF-IDF",
         "glove": "GloVe", "w2v_ag": "Word2Vec AG", "w2v_nyt": "Word2Vec NYT", "bert": "BERT"}
DOCS = ROOT / "docs"
FIGS = DOCS / "figures"


def read(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def main():
    data = {m: read(m + ".json") for m in METHODS}
    stats, audit = read("data_stats.json"), read("audit.json")
    if audit["status"] != "passed":
        raise ValueError("Audit must pass before report generation.")
    FIGS.mkdir(parents=True, exist_ok=True)
    output = ROOT / "output" / "pdf"
    output.mkdir(parents=True, exist_ok=True)
    font_path = os.environ.get("REPORT_FONT", "C:/Windows/Fonts/simsun.ttc")
    if Path(font_path).exists():
        pdfmetrics.registerFont(TTFont("Chinese", font_path, subfontIndex=0))
    else:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    font = "Chinese" if Path(font_path).exists() else "STSong-Light"
    body = ParagraphStyle("body", fontName=font, fontSize=10.5, leading=17,
                          spaceAfter=8, wordWrap="CJK", textColor=colors.HexColor("#20252b"))
    heading = ParagraphStyle("heading", parent=body, fontSize=15, leading=22, spaceBefore=8, spaceAfter=12, textColor=colors.black)
    title = ParagraphStyle("title", parent=body, fontSize=24, leading=34, spaceAfter=18, textColor=colors.black)
    small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=12)
    cell = ParagraphStyle("cell", parent=body, fontSize=9, leading=13, spaceAfter=0)
    story, md = [], ["# NYT 新闻文本分类实验报告", ""]
    def p(text, style=body):
        story.append(Paragraph(escape(text).replace("\n", "<br/>"), style))
        md.extend([text, ""])
    def h(text):
        story.append(Paragraph(escape(text), heading))
        md.extend(["## " + text, ""])
    def table(rows, widths=None):
        converted = [[Paragraph(escape(str(x)), cell) for x in row] for row in rows]
        t = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef3")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#aab6c2")),
            ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#dce2e7")),
        ]))
        story.extend([t, Spacer(1, 10)])
        md.append("| " + " | ".join(str(x) for x in rows[0]) + " |")
        md.append("| " + " | ".join("---" for _ in rows[0]) + " |")
        md.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows[1:])
        md.append("")
    def fig(name, width, height):
        story.append(Image(str(FIGS / name), width=width, height=height))
        md.extend([f"![{name}](figures/{name})", ""])
    def page():
        story.append(PageBreak())

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig1, ax = plt.subplots(figsize=(8.6, 3.5))
    y = np.arange(len(METHODS))
    for offset, metric, color, label in [(-.18, "accuracy", "#377ba8", "Accuracy"), (.18, "macro_f1", "#c5773b", "Macro-F1")]:
        values = [data[m]["test"][metric] * 100 for m in METHODS]
        ax.barh(y + offset, values, .34, color=color, label=label)
        for yy, value in zip(y + offset, values):
            ax.text(value + .35, yy, f"{value:.2f}", va="center", fontsize=8)
    ax.set_yticks(y, [NAMES[m] for m in METHODS])
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("NYT test score (%)")
    ax.legend(loc="lower left", ncol=2)
    fig1.tight_layout()
    fig1.savefig(FIGS / "comparison.png", dpi=180)
    plt.close(fig1)
    history = data["bert"]["details"]["history"]
    fig2, axes = plt.subplots(1, 2, figsize=(8.6, 2.9))
    axes[0].plot([r["epoch"] for r in history], [r["train_loss"] for r in history], "o-", color="#377ba8")
    axes[0].set(xlabel="Epoch", ylabel="Training cross-entropy", xticks=[1, 2, 3])
    axes[1].plot([r["epoch"] for r in history], [r["validation_macro_f1"] for r in history], "o-", color="#c5773b")
    axes[1].set(xlabel="Epoch", ylabel="Validation Macro-F1", xticks=[1, 2, 3])
    fig2.tight_layout()
    fig2.savefig(FIGS / "bert_history.png", dpi=180)
    plt.close(fig2)
    fig3, ax = plt.subplots(figsize=(4.8, 3.5))
    matrix = np.array(data["bert"]["test"]["confusion_matrix"])
    ax.imshow(matrix, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max()/2 else "black")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=data["bert"]["labels"],
           yticklabels=data["bert"]["labels"], xlabel="Predicted label", ylabel="True label")
    fig3.tight_layout()
    fig3.savefig(FIGS / "bert_confusion.png", dpi=180)
    plt.close(fig3)

    best = max(METHODS, key=lambda m: data[m]["test"]["macro_f1"])
    best_score = data[best]["test"]
    p("NYT 新闻文本分类实验报告", title)
    p("作业一  |  词袋表示、词向量与 BERT 微调", heading)
    p(f"本实验使用同一 NYT 测试集比较七种方法。此次单次实验中，{NAMES[best]} 的测试 Macro-F1 最高，为 {best_score['macro_f1']:.4f}，Accuracy 为 {best_score['accuracy']:.4f}。全部结果来自实际训练；每条测试预测、划分行号和运行参数均随仓库保存。")
    h("一 数据与实验范围")
    p(f"NYT 共 {stats['rows']:,} 篇新闻，包含 business、politics、sports 三类；AG News 共 {stats['ag_rows']:,} 篇，仅用于训练其中一组 Word2Vec。固定随机种子 42，以分层随机方式打乱并按约 80% / 10% / 10% 划分。")
    rows = [["集合", "business", "politics", "sports", "总数"]]
    for s, label in [("train", "训练"), ("validation", "验证"), ("test", "测试")]:
        rows.append([label, *[stats["split_counts"][s][c] for c in ["business", "politics", "sports"]], stats["split_sizes"][s]])
    table(rows, [80, 100, 100, 100, 105])
    majority = stats["split_counts"]["test"]["sports"] / stats["split_sizes"]["test"]
    p(f"类别明显不平衡。仅预测 sports 的基线 Accuracy 为 {majority:.4f}，Macro-F1 为 {(2*majority/(1+majority))/3:.4f}。因此主要依据 Macro-F1 比较对少数类别的识别能力，同时报告 Accuracy。")
    p("要求说明：Task 1 总述写“三种”，但正文只列出 Binary 和 Word Frequency。本实验完成这两项，并补充 TF-IDF 作为第三种；此项属于对原文歧义的明确实现选择。")
    p(f"原 NYT 中有 {stats['duplicate_rows']} 条重复文本记录。保持原始样本开展规定的随机划分，并单独审计跨集合重复，避免隐瞒可能造成的乐观偏差。", small)

    page()
    h("二 表示方法与训练设置")
    p("词袋与词向量实验使用同一分词规则：转为小写，提取英文词并保留词内撇号，不移除停用词。BERT 则使用其预训练 WordPiece tokenizer。Accuracy 是正确预测比例；Macro-F1 是三个类别各自 F1 的算术平均，零分母按 0 处理。")
    table([["方法", "文档表示与参数"],
           ["Binary BoW", "词出现则为 1，否则为 0。完整训练集词表。"],
           ["Word Frequency", "记录每个词在文档中的出现次数，不进行长度归一化。"],
           ["TF-IDF", "tf × [log((1+N)/(1+df))+1]，随后做文档 L2 归一化。"],
           ["GloVe", "Stanford glove.6B.100d，400,000 个词，100 维预训练词向量。"],
           ["Word2Vec AG", "全部 AG 文本；Skip-gram，100 维，窗口 5，负采样 5。"],
           ["Word2Vec NYT", "仅 NYT 训练集文本；与 AG 模型使用相同参数。"]], [110, 375])
    p("两组 Word2Vec 最低词频为 5，训练 10 轮；使用单 worker、固定种子和稳定词哈希。词向量方法对文档内所有有效词向量取算术平均，重复出现的词重复计入；跳过 OOV，全 OOV 文档使用零向量。100 维特征标准化仅在训练集上拟合。")
    p("六组传统方法统一使用 Logistic Regression（L2 正则，lbfgs，最多 3000 次迭代）。在 C={0.1,1,10} 中只依据验证 Macro-F1 选择，平局取较小 C；所选模型仅训练于训练集，不再合并验证集。词表、IDF、标准化和 NYT Word2Vec 都不拟合验证或测试文本。")
    table([["方法", "词表或向量维数", "所选 C", "验证 Macro-F1"]] +
          [[NAMES[m], data[m]["details"].get("vocabulary_size", 100), data[m]["details"]["selected_C"],
            f"{data[m]['validation']['macro_f1']:.4f}"] for m in METHODS[:-1]], [130, 125, 85, 145])
    p("BERT 使用 google-bert/bert-base-uncased，全参数微调。max_length=64（含特殊 token），完整训练 3 轮；批量 16、累积 2、AdamW 学习率 2e-5、权重衰减 0.01、10% warmup 后线性衰减、梯度裁剪 1.0。bias 与 LayerNorm 权重不做衰减。CUDA 上使用 FP16。")

    page()
    h("三 统一测试集结果")
    table([["方法", "Accuracy", "Macro-F1", "运行秒数"]] +
          [[NAMES[m], f"{data[m]['test']['accuracy']:.4f}", f"{data[m]['test']['macro_f1']:.4f}",
            f"{data[m]['details']['seconds']:.1f}"] for m in METHODS], [145, 110, 110, 120])
    fig("comparison.png", 485, 197)
    p("运行时间包括各方法特征构建、训练、验证和预测等步骤，不含预训练资源下载。部分任务并行运行，且硬件与模型结构不同，因此时间仅描述本次运行，不作为严格速度基准。", small)
    traditional_best = max(METHODS[:-1], key=lambda m: data[m]["test"]["macro_f1"])
    delta = data["bert"]["test"]["macro_f1"] - data[traditional_best]["test"]["macro_f1"]
    p(f"传统方法中 {NAMES[traditional_best]} 的测试 Macro-F1 最高。BERT 与它的差值为 {delta:+.4f}。这是固定种子下的描述性比较，未进行多种子重复或显著性检验，不能据此断言某方法在所有数据上更好。")
    p("词袋模型保留直接区分类别的关键词；TF-IDF 降低常见词权重。平均词向量压缩到 100 维，能利用词之间的语义联系，但丢失词序，也可能稀释少数关键句。BERT 建模上下文，不过本作业只允许 64 个 token，长新闻的后续信息会被截断。")

    page()
    h("四 词向量覆盖与 BERT 训练")
    table([["词向量", "训练 token 覆盖率", "测试 token 覆盖率", "测试全 OOV 文档"]] +
          [[NAMES[m], f"{data[m]['details']['coverage']['train']['coverage']:.2%}",
            f"{data[m]['details']['coverage']['test']['coverage']:.2%}",
            data[m]["details"]["coverage"]["test"]["empty_documents"]] for m in ["glove", "w2v_ag", "w2v_nyt"]],
          [110, 135, 135, 105])
    ag_f1 = data["w2v_ag"]["test"]["macro_f1"]
    nyt_f1 = data["w2v_nyt"]["test"]["macro_f1"]
    p(f"AG Word2Vec 与 NYT Word2Vec 的测试 Macro-F1 分别为 {ag_f1:.4f} 和 {nyt_f1:.4f}。领域、语料长度与规模、词覆盖和低频词过滤同时变化，因此性能差异不能单独归因于“同领域”这一因素。覆盖率按词出现次数计算，而非不同词类型数量。")
    table([["Epoch", "训练平均损失", "验证 Macro-F1"]] +
          [[r["epoch"], f"{r['train_loss']:.4f}", f"{r['validation_macro_f1']:.4f}"] for r in history], [90, 190, 205])
    fig("bert_history.png", 485, 164)
    p(f"BERT 完成三轮训练后，依据验证 Macro-F1 选取第 {data['bert']['details']['best_epoch']} 轮 checkpoint。测试集不参与轮次选择。完整曲线数据保存在 results/bert_history.json。")
    p("由于模型只观察新闻开头，标题和导语可能携带较强类别信号，但有关截断影响的判断属于解释性假设；本实验未增加长序列对照，不能量化其因果影响。")

    page()
    h("五 分类误差与重复文本审计")
    fig("bert_confusion.png", 350, 255)
    report = data["bert"]["test"]["classification_report"]
    table([["类别", "Precision", "Recall", "F1", "样本数"]] +
          [[c, f"{report[c]['precision']:.4f}", f"{report[c]['recall']:.4f}", f"{report[c]['f1-score']:.4f}", int(report[c]["support"])]
           for c in data["bert"]["labels"]], [100, 95, 95, 95, 100])
    overlap = stats["identical_text_overlap"]
    p(f"完全相同文本的交集数量：训练/验证 {overlap['train/validation']}，训练/测试 {overlap['train/test']}，验证/测试 {overlap['validation/test']}（计不同文本数）。AG 与 NYT 的完全相同文本交集为 {stats['ag_nyt_exact_text_overlap']}。这些统计不涵盖改写或近重复新闻。")
    p(f"补充敏感性检查：排除与训练集文本完全相同的测试记录后，保留 {audit['test_rows_without_exact_train_duplicates']} / {audit['test_rows']} 条测试样本。以下仅重新计算已保存预测，不重新训练、不替代规定测试集主表。")
    table([["方法", "排重后 Accuracy", "排重后 Macro-F1"]] +
          [[NAMES[m], f"{audit['metrics_without_exact_train_duplicates'][m]['accuracy']:.4f}",
            f"{audit['metrics_without_exact_train_duplicates'][m]['macro_f1']:.4f}"] for m in METHODS], [175, 155, 155])

    page()
    h("六 复现与结论")
    p(f"本次训练环境：Python {data['bert']['environment']['python']}；GPU 为 {data['bert']['details']['gpu']}。所有结果使用同一保存划分，并通过逐条预测重算核验。")
    table([["依赖", "实际版本"]] + [[k, v] for k, v in data["bert"]["environment"]["packages"].items()], [245, 240])
    p("从仓库根目录依次运行：uv sync --frozen --python 3.12；uv run --frozen python scripts/run_all.py。分步命令、CPU 选项、网络设置和文件结构详见 README.md。PDF 和 Markdown 由同一结果文件生成。")
    p("可复核证据：results/split.json 保存划分与源文件 SHA-256；各方法 JSON 保存参数、验证和测试指标、混淆矩阵；*_predictions.csv 保存原始行号、真实标签和预测标签；audit.json 保存独立指标核验及重复文本敏感性结果。预训练资源来源和 SHA-256 记录在 assets.json。")
    p(f"本次比较中，{NAMES[best]} 达到最高测试 Macro-F1。结论仅适用于本数据、当前划分和给定超参数。未覆盖多随机种子、近重复去重和长文本 BERT 对照；这些是后续完善实验的方向，不作为本报告已经完成的实验。")
    h("参考资料")
    references = [
        "1. Stanford GloVe 项目与 glove.6B 预训练向量：https://nlp.stanford.edu/projects/glove/",
        "2. Mikolov 等，Efficient Estimation of Word Representations in Vector Space，2013：https://arxiv.org/abs/1301.3781",
        "3. Devlin 等，BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding，NAACL 2019：https://aclanthology.org/N19-1423/",
        "4. BERT 模型与固定权重版本：https://huggingface.co/google-bert/bert-base-uncased",
        "5. 数据与任务定义：仓库内 HW-1/nyt.csv、HW-1/ag.csv 及 作业一要求.docx。"
    ]
    for ref in references:
        p(ref, small)
    def footer(canvas, doc):
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#69727c"))
        canvas.drawString(48, 27, "NYT 新闻文本分类实验")
        canvas.drawRightString(A4[0] - 48, 27, f"{doc.page}")
    doc = SimpleDocTemplate(str(output / "report.pdf"), pagesize=A4, leftMargin=48, rightMargin=48,
                            topMargin=42, bottomMargin=44, title="NYT 新闻文本分类实验报告", author="")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (DOCS / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(output / "report.pdf")


if __name__ == "__main__":
    main()
