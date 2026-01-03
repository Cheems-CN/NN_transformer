"""
Generate simple block diagrams for the five model architectures described in the paper.

The resulting figure is saved under ``latex/main/figures/model_architectures.png`` so it
can be referenced directly in the LaTeX manuscript. Labels are intentionally kept in
Chinese to 保持与论文正文一致，如需切换语言可在模型定义处修改文本。
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

COLORS = ["#005f73", "#0a9396", "#94d2bd", "#ee9b00", "#ae2012", "#3d405b"]


def _draw_flow(ax, title: str, blocks: list[str]) -> None:
    """Render a left-to-right flow diagram for a single model."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.85, title, fontsize=12, fontweight="bold")

    x = 0.08
    box_w, box_h = 0.16, 0.28

    for idx, label in enumerate(blocks):
        color = COLORS[idx % len(COLORS)]
        box = FancyBboxPatch(
            (x, 0.35),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(box)
        ax.text(
            x + box_w / 2,
            0.49,
            label,
            ha="center",
            va="center",
            fontsize=9,
            wrap=True,
        )
        if idx < len(blocks) - 1:
            ax.annotate(
                "",
                xy=(x + box_w + 0.02, 0.49),
                xytext=(x + box_w + 0.07, 0.49),
                arrowprops=dict(arrowstyle="->", linewidth=1.1, color="#333333"),
            )
            x += box_w + 0.09


def build_figure(output_path: Path) -> None:
    models = [
        (
            "Pure CNN (ResNet18)",
            [
                "MRI 输入 224×224",
                "7×7 卷积 + BN + ReLU",
                "残差块堆叠 (64→512)",
                "全局平均池化",
                "4 类全连接分类头",
            ],
        ),
        (
            "Pure ViT",
            [
                "MRI 输入 224×224",
                "16×16 Patch 切分",
                "线性嵌入 + 位置编码",
                "8 层 Transformer Encoder",
                "[CLS] Token 分类头",
            ],
        ),
        (
            "Hybrid（串行融合）",
            [
                "CNN 感知前端",
                "残差块提取 14×14×512 特征",
                "展平 + 线性投影",
                "4 层 Transformer Encoder",
                "[CLS] Token 分类器",
            ],
        ),
        (
            "Parallel（并行融合）",
            [
                "共享 MRI 输入",
                "CNN 分支提取局部纹理",
                "Transformer 分支建模全局",
                "特征拼接/融合层",
                "联合分类头",
            ],
        ),
        (
            "Embedded（注意力嵌入）",
            [
                "CNN 主干",
                "卷积块中插入注意力瓶颈",
                "融合局部与上下文特征",
                "全局池化",
                "分类头输出 4 类",
            ],
        ),
    ]

    fig, axes = plt.subplots(
        nrows=len(models),
        ncols=1,
        figsize=(9, 11),
        constrained_layout=True,
    )

    for ax, (name, steps) in zip(axes, models):
        _draw_flow(ax, name, steps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    build_figure(root / "latex" / "main" / "figures" / "model_architectures.png")
