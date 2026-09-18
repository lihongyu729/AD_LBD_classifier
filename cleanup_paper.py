#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

with open('论文.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 第1步：删除重复的 SSM 数学公式和 3.1.2 选择性扫描机制
content = re.sub(
    r'SSM 通过隐状态 \$h\(t\) \\in \\mathbb{R}\^N\$ 对系统进行建模.*?三维选择性状态更新机制使模型能够对LBD中的局部异常信号进行动态增强，阻止无关区域噪声传播并建立非均匀空间依赖。\n',
    '',
    content,
    flags=re.DOTALL
)

# 第2步：删除重复的 3.2 子模块设计及其子章节
content = re.sub(
    r'### 3\.2子模块设计\nMedMambaSS3M 的主干设计可拆分为三个子模块.*?#### 3\.2\.3 Classification Head\n在最后一个 SS3MBlock3D 输出后.*?的嵌入向量以供原型/对比学习使用。\n\n',
    '',
    content,
    flags=re.DOTALL
)

# 第3步：删除重复的第二个 3.3 损失函数及 3.3.1 起的内容
pattern = r'### 3\.3 损失函数\n在我们的研究框架中，模型的训练包含两个核心阶段.*?#### 3\.3\.1预训练阶段：重建损失 \(Reconstruction Loss\)\n'
matches = list(re.finditer(pattern, content, re.DOTALL))
if len(matches) > 1:
    # 保留第一个，删除后续的重复
    for match in matches[1:]:
        content = content[:match.start()] + content[match.end():]

with open('论文.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ 论文清理完成：已删除重复的SSM理论和 3.2/3.3 子章节")
