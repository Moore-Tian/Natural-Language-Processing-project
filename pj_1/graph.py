import matplotlib.pyplot as plt
import numpy as np
import re

# 从文本文件中读取数据
data = {}
with open("score.txt", "r") as file:
    lines = file.readlines()
    for i in range(0, len(lines), 4):
        model_name = lines[i].strip()
        print(lines[i + 1].split(":"))
        analogy_score = float(lines[i + 1].split(":")[1].strip())
        
        similarity_scores_line = lines[i+2].strip()
        pearsonR_result = re.search(r"PearsonRResult\(statistic=(.*?), pvalue=(.*?)\)", similarity_scores_line)
        pearsonR_statistic = float(pearsonR_result.group(1))
        pearsonR_pvalue = float(pearsonR_result.group(2))
        
        significance_result = re.search(r"SignificanceResult\(statistic=(.*?), pvalue=(.*?)\)", similarity_scores_line)
        significance_statistic = float(significance_result.group(1))
        significance_pvalue = float(significance_result.group(2))
        
        data[model_name] = {
            "analogy_score": analogy_score,
            "pearsonR_statistic": pearsonR_statistic,
            "pearsonR_pvalue": pearsonR_pvalue,
            "significance_statistic": significance_statistic,
            "significance_pvalue": significance_pvalue
        }

# 提取项目名称和得分数据
project_names = ["analogy_score", "pearsonR_statistic", "significance_statistic"]
pvalue_names = ["pearsonR_pvalue", "significance_pvalue"]
model_names = list(data.keys())
scores = {project: [data[model][project] for model in model_names] for project in project_names}
pvalues = {pvalue: [np.log10(data[model][pvalue]) for model in model_names] for pvalue in pvalue_names}

# 绘制两个子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# 绘制得分的柱状图
x = np.arange(len(model_names))
width = 0.15

for i, project in enumerate(project_names):
    ax1.bar(x + (i * width), scores[project], width, label=project)

ax1.set_ylabel('Scores')
ax1.set_title('Scores Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names)
ax1.legend()

# 绘制p-value的柱状图
for i, pvalue in enumerate(pvalue_names):
    ax2.bar(x + (i * width), pvalues[pvalue], width, label=pvalue)

ax2.set_ylabel('log10(p-value)')
ax2.set_title('p-value Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names)
ax2.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()