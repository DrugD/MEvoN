import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import font_manager
import matplotlib.colors as mcolors

plt.rcParams['font.serif'] = ['Times New Roman']

# 横坐标（path）
path_num = [1, 3, 5, 7, 9]

# GCN-branch的指标
gcn_r2 = [0.634004, 0.780594, 0.816022, 0.468328, 0.81849]
gcn_pearson = [0.891388, 0.892109, 0.911968, 0.896391, 0.906301]
gcn_mse = [0.569408, 0.076457, 0.064112, 0.185274, 0.063252]
gcn_mae = [0.711398, 0.210224, 0.189246, 0.371843, 0.187531]

# EVO-branch(GCN-based)的指标
evo_r2 = [0.81006, 0.791408, 0.828844, 0.805921, 0.814241]
evo_pearson = [0.901038, 0.894218, 0.912186, 0.901293, 0.905842]
evo_mse = [0.066189, 0.072689, 0.059643, 0.067632, 0.064732]
evo_mae = [0.181664, 0.188412, 0.16973, 0.182073, 0.178486]

# 自定义RGB颜色
colors_gcn = {'r2': (71, 82, 151), 'pearson': (56, 132, 85), 'mse': (240, 161, 66), 'mae': (212, 72, 81)}
colors_evo = {'r2': (141, 150, 201), 'pearson': (169, 219, 188), 'mse': (246, 197, 138), 'mae': (230, 150, 156)}

# 创建图
plt.figure(figsize=(10, 6))

# import pdb;pdb.set_trace()
# 绘制R²
plt.plot(path_num, gcn_r2, label='Mol(R²)', color=[x/255 for x in colors_gcn['r2']], marker='o', linewidth=3, markersize=10)
plt.plot(path_num, evo_r2, label='Evo(R²)', color=[x/255 for x in  colors_evo['r2']], marker='o', linewidth=3, markersize=10)

# 绘制Pearson
plt.plot(path_num, gcn_pearson, label='Mol(PCC)', color=[x/255 for x in  colors_gcn['pearson']], marker='o', linewidth=3, markersize=10)
plt.plot(path_num, evo_pearson, label='Evo(PCC)', color=[x/255 for x in  colors_evo['pearson']], marker='o', linewidth=3, markersize=10)

# 绘制MSE
plt.plot(path_num, gcn_mse, label='Mol(MSE)', color=[x/255 for x in  colors_gcn['mse']], marker='o', linewidth=3, markersize=10)
plt.plot(path_num, evo_mse, label='Evo(MSE)', color=[x/255 for x in  colors_evo['mse']], marker='o', linewidth=3, markersize=10)

# 绘制MAE
plt.plot(path_num, gcn_mae, label='Mol(MAE)', color=[x/255 for x in  colors_gcn['mae']], marker='o', linewidth=3, markersize=10)
plt.plot(path_num, evo_mae, label='Evo(MAE)', color=[x/255 for x in  colors_evo['mae']], marker='o', linewidth=3, markersize=10)


# # 在每个数据点上显示数值，保留三位有效数字，并添加随机偏移量来避免重叠
# def add_text_with_offset(x, y, text, fontsize=14):
#     offset_x = random.uniform(-0.5, 0.5)  # x方向随机偏移
#     offset_y = random.uniform(0.02, 0.02)  # y方向随机偏移
#     plt.text(x + offset_x, y + offset_y, f'{text:.3f}', fontsize=fontsize, ha='center', va='bottom')

offset_x = 0.0
offset_y = -0.1

# for i, (x, y) in enumerate(zip(path_num, gcn_r2)):
#     plt.text(x + offset_x, y + offset_y, f'{y:.3f}', fontsize=14, ha='center', va='bottom')
# for i, (x, y) in enumerate(zip(path_num, evo_r2)):
#     plt.text(x + offset_x, y + offset_y, f'{y:.3f}', fontsize=14, ha='center', va='bottom')

# for i, (x, y) in enumerate(zip(path_num, gcn_pearson)):
#     plt.text(x + offset_x, y + offset_y, f'{y:.3f}', fontsize=14, ha='center', va='bottom')
# for i, (x, y) in enumerate(zip(path_num, evo_pearson)):
#     plt.text(x + offset_x, y + offset_y, f'{y:.3f}', fontsize=14, ha='center', va='bottom')

# for i, (x, y) in enumerate(zip(path_num, gcn_mse)):
#     plt.text(x + offset_x, y + offset_y, f'{y:.3f}', fontsize=14, ha='center', va='bottom')
# for i, (x, y) in enumerate(zip(path_num, evo_mse)):
#     plt.text(x + offset_x, y + offset_y, f'{y:.3f}', fontsize=14, ha='center', va='bottom')

for i, (x, y) in enumerate(zip(path_num, gcn_mae)):
    print(i)
    if i==1:
        plt.text(x + 0.3, y + 0.02, f'{y:.3f}', fontsize=15, ha='center', va='bottom')
    elif i==2:
        plt.text(x - 0.3, y + 0.02, f'{y:.3f}', fontsize=15, ha='center', va='bottom')
    else:
        plt.text(x + offset_x, y + 0.02, f'{y:.3f}', fontsize=15, ha='center', va='bottom')
for i, (x, y) in enumerate(zip(path_num, evo_mae)):
    plt.text(x + offset_x, y - 0.08, f'{y:.3f}', fontsize=15, ha='center', va='bottom')


# 设置标题和标签
# plt.title('Comparison of GCN-branch and EVO-branch(GCN-based)', fontsize=14)
plt.xlabel('Path Num', fontsize=20)
plt.ylabel('Metrics', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# 显示图例
# 设置图例位置
plt.legend(ncol=4, fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 1.25))


# 调整布局
plt.tight_layout()

# 保存为PNG文件
plt.savefig('/home/data1/lk/project/mol_tree/outputs_visual/comparison_plot_custom_colors.pdf', dpi=500)

# 显示图表
plt.show()
