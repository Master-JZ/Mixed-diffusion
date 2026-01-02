import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ============================
# 1. 准备数据 (如果没有真实数据，这里模拟一个长尾分布)
# ============================
np.random.seed(42)
# 模拟 10,000 个物品，遵循 Zipf 定律 (参数 a=1.5 用于模拟典型的长尾)
num_items = 10000
ranks = np.arange(1, num_items + 1)
# 频率公式：Freq = C / rank^a
frequencies = 5000 / (ranks ** 0.8) 
frequencies = frequencies.astype(int) #以此作为你的Y轴数据

# 如果你有真实数据，请用真实数据的频率列表替换上面：
# frequencies = sorted_item_counts (从大到小排序的列表)
# num_items = len(frequencies)
# ranks = np.arange(1, num_items + 1)

# ============================
# 2. 绘图设置 (学术风格配色)
# ============================
# 设置字体 (根据论文要求，通常是 Times New Roman 或 Arial)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(figsize=(8, 5))

# 定义分割点 (例如前 20% 是热门物品)
split_idx = int(num_items * 0.2)
head_ranks = ranks[:split_idx]
head_freqs = frequencies[:split_idx]
tail_ranks = ranks[split_idx:]
tail_freqs = frequencies[split_idx:]

# ============================
# 3. 绘制曲线与填充区域
# ============================
# 绘制主曲线
ax.plot(ranks, frequencies, color='#2c3e50', linewidth=2, label='Interaction Frequency')

# 填充 Head 区域 (热门) - 使用浅米色/暖色
ax.fill_between(head_ranks, 0, head_freqs, color='#e67e22', alpha=0.15)

# 填充 Tail 区域 (长尾) - 使用浅灰蓝/冷色
ax.fill_between(tail_ranks, 0, tail_freqs, color='#3498db', alpha=0.15)

# ============================
# 4. 添加标注与装饰
# ============================
# 绘制分割线
ax.axvline(x=split_idx, color='gray', linestyle='--', linewidth=1, alpha=0.7)

# 添加文字标注 (Head Items)
ax.text(split_idx * 0.5, max(frequencies) * 0.3, 
        'Head Items\n(High Frequency)', 
        horizontalalignment='center', color='#d35400', fontweight='bold')

# 添加文字标注 (Tail Items)
ax.text(split_idx + (num_items - split_idx) * 0.3, max(frequencies) * 0.1, 
        'Long-tail Items\n(Sparse Interactions)', 
        horizontalalignment='center', color='#2980b9', fontweight='bold')

# 添加"动机"箭头 (指向尾部)
ax.annotate('Problem: \nMajority of items lack\nsufficient training data', 
            xy=(num_items * 0.6, frequencies[int(num_items * 0.6)]), 
            xytext=(num_items * 0.6, max(frequencies) * 0.5),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            horizontalalignment='center', fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

# ============================
# 5. 坐标轴与标签
# ============================
ax.set_xlabel('Item Rank (Sorted by Popularity)', fontsize=14, fontweight='bold')
ax.set_ylabel('Interaction Frequency', fontsize=14, fontweight='bold')
ax.set_xlim(0, num_items)
ax.set_ylim(0, max(frequencies) * 1.1)

# 去掉上方和右侧的边框 (更符合现代审美)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 显示网格 (可选)
ax.grid(axis='y', linestyle=':', alpha=0.3)

plt.tight_layout()

# ============================
# 6. 保存图片
# ============================
# 保存为 PDF (矢量图，放论文里最清晰)
plt.savefig('long_tail_motivation.pdf', dpi=300, bbox_inches='tight')
# 也可以保存为 PNG 预览
plt.savefig('long_tail_motivation.png', dpi=300, bbox_inches='tight')

plt.show()