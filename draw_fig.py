import matplotlib.pyplot as plt
import numpy as np

# 融合方式性能对比图       # Performance comparison chart of fusion methods
title = 'fusion'
acc1 = 0.9524  # Attention Fuse
acc2 = 1.0  # Mul
model_names = ['Attention Fuse', 'Mul']
accuracies = [acc1, acc2]

# # 单一模型和融合模型性能对比图   # Performance Comparison Diagram of Single Model and Fusion Model
# title = 'model'
# acc1 = 0.8333 # ChnGNN_Module
# acc2 = 0.8571  # Time_Module
# acc3 = 1.0  # Rgb_Module
# acc4 = 1.0 # All
# # 融合方式 & 准确率
# model_names = ['ChnGNN_Module', 'Time_Module', 'Rgb_Module', 'All']
# accuracies = [acc1, acc2, acc3, acc4]



# x-axis
x = np.arange(len(model_names))
width = 0.4

# Create a drawing
plt.figure(figsize=(8, 6))
bars = plt.bar(x, accuracies, width=width, color='skyblue')

# 添加每个柱子的数值标签    # Add numerical labels for each column
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')

plt.xticks(x, model_names)

plt.ylim(0.5, 1.00)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(f'./results/model_accuracy_bar_{title}.png')
plt.show()
