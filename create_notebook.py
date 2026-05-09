#!/usr/bin/env python3
"""
将所有Python训练脚本转换为单个Jupyter Notebook文件，适用于Google Colab GPU运行
"""

import json
import os

def create_notebook():
    notebook = {
        'cells': [],
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python',
                'version': '3.8.0'
            },
            'colab': {
                'provenance': [],
                'gpuType': 'T4'
            },
            'accelerator': 'GPU'
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }

    # Cell 1: Markdown介绍
    notebook['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': '''# 文本分类模型训练工具集 - Google Colab GPU版本

## 📋 目录
本Notebook包含多个文本分类模型训练脚本，整合为单一文件便于在Google Colab中使用GPU运行。

### 模型列表
1. **基础LSTM分类器** (`train_scratch_lstm.py`) - 基础双向LSTM文本分类
2. **ULMFiT模型** (`train_ulmfit_scratch.py`) - ULMFiT预训练+微调
3. **强LSTM模型** (`train_strong_lstm.py`) - 多层BiLSTM+CNN特征
4. **ELMo风格模型** (`train_elmo_scratch.py`) - 双向语言模型融合
5. **n-gram混合模型** (`train_ulmfit_ngram.py`) - ULMFiT+n-gram特征
6. **VAT半监督模型** (`train_ulmfit_vat.py`) - 虚拟对抗训练
7. **自训练模型** (`selftrain_from_checkpoint.py`) - 伪标签自训练
8. **继续微调** (`continue_ulmfit_finetune.py`) - 从检查点继续微调
9. **全量微调** (`finetune_all_from_checkpoint.py`) - 全参数微调
10. **生成提交** (`make_submission_from_checkpoint.py`) - 从检查点生成预测
11. **NB-LSTM混合** (`nb_lstm_hybrid.py`) - 朴素贝叶斯+LSTM集成

## ⚙️ Colab设置
1. 点击 **运行时** → **更改运行时类型** → 选择 **GPU**
2. 上传您的数据文件（train.csv, test.csv, train_unlabel.csv）
3. 根据需要修改各部分的参数并运行单元格

---
'''
    })

    # Cell 2: 环境设置
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': '''# ============================================
# 第一部分：环境设置和依赖安装
# ============================================
# 功能：安装必要的依赖包，设置运行环境

# 检查GPU是否可用
import torch
print(f"PyTorch版本：{torch.__version__}")
print(f"CUDA可用：{torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备：{torch.cuda.get_device_name(0)}")
    print(f"CUDA版本：{torch.version.cuda}")

# 设置环境变量（避免某些OpenMP重复加载问题）
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")'''
    })

    # 读取源文件
    files_content = {}
    import_files = [
        'train_scratch_lstm.py',
        'train_ulmfit_scratch.py',
        'train_strong_lstm.py',
        'train_elmo_scratch.py',
        'train_ulmfit_ngram.py',
        'train_ulmfit_vat.py',
        'selftrain_from_checkpoint.py',
        'continue_ulmfit_finetune.py',
        'finetune_all_from_checkpoint.py',
        'make_submission_from_checkpoint.py',
        'nb_lstm_hybrid.py'
    ]

    for fname in import_files:
        with open(f'/workspace/{fname}', 'r', encoding='utf-8') as f:
            files_content[fname] = f.read()

    # Cell 3: 基础工具函数（来自train_scratch_lstm.py）
    base_code = files_content['train_scratch_lstm.py']
    # 移除main函数和argparse部分，只保留类和函数定义
    base_cells_source = f'''# ============================================
# 第二部分：通用工具函数（来自train_scratch_lstm.py）
# ============================================
# 功能：提供文本处理、数据加载、词汇构建、LSTM模型等基础功能
# 这是所有其他模型的基础依赖模块

{base_code}'''

    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': base_cells_source
    })

    # Cell 4: ULMFiT模型（来自train_ulmfit_scratch.py）
    ulmfit_code = files_content['train_ulmfit_scratch.py']
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': f'''# ============================================
# 第三部分：ULMFiT模型（来自train_ulmfit_scratch.py）
# ============================================
# 功能：实现ULMFiT（Universal Language Model Fine-tuning）模型
# 包括：单向LSTM编码器、语言模型预训练、分类器微调
# 使用方法：运行此单元格后，可以使用ULMFiTClassifier类进行训练

{ulmfit_code}'''
    })

    # Cell 5: 强LSTM模型（来自train_strong_lstm.py）
    strong_code = files_content['train_strong_lstm.py']
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': f'''# ============================================
# 第四部分：强LSTM模型（来自train_strong_lstm.py）
# ============================================
# 功能：实现增强的LSTM分类器
# 特性：多层BiLSTM、CNN多尺度特征、嵌入层通道dropout
# 使用方法：运行此单元格后，可以使用StrongTextLSTMClassifier类

{strong_code}'''
    })

    # Cell 6: ELMo风格模型（来自train_elmo_scratch.py）
    elmo_code = files_content['train_elmo_scratch.py']
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': f'''# ============================================
# 第五部分：ELMo风格模型（来自train_elmo_scratch.py）
# ============================================
# 功能：实现类似ELMo的双向语言模型融合分类器
# 特性：前向+后向LM预训练、多层特征混合、多种池化策略
# 使用方法：需要先运行ULMFiT部分获取预训练权重

{elmo_code}'''
    })

    # Cell 7: n-gram混合模型（来自train_ulmfit_ngram.py）
    ngram_code = files_content['train_ulmfit_ngram.py']
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': f'''# ============================================
# 第六部分：n-gram混合模型（来自train_ulmfit_ngram.py）
# ============================================
# 功能：结合ULMFiT和n-gram特征的混合模型
# 特性：哈希n-gram特征、可学习的特征混合权重
# 使用方法：需要提供ULMFiT预训练的检查点

{ngram_code}'''
    })

    # Cell 8: VAT半监督模型（来自train_ulmfit_vat.py）
    vat_code = files_content['train_ulmfit_vat.py']
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': f'''# ============================================
# 第七部分：VAT半监督模型（来自train_ulmfit_vat.py）
# ============================================
# 功能：虚拟对抗训练（Virtual Adversarial Training）
# 特性：利用无标签数据进行半监督学习、对抗扰动增强
# 使用方法：需要提供预训练的ULMFiT检查点

{vat_code}'''
    })

    # Cell 9: 自训练模型（来自selftrain_from_checkpoint.py）
    selftrain_code = files_content['selftrain_from_checkpoint.py']
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': f'''# ============================================
# 第八部分：自训练模型（来自selftrain_from_checkpoint.py）
# ============================================
# 功能：伪标签自训练（Self-Training with Pseudo Labels）
# 特性：高置信度样本选择、加权损失函数
# 使用方法：需要提供预训练模型的检查点

{selftrain_code}'''
    })

    # Cell 10: 继续微调（来自continue_ulmfit_finetune.py）
    continue_code = files_content['continue_ulmfit_finetune.py']
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': f'''# ============================================
# 第九部分：继续微调（来自continue_ulmfit_finetune.py）
# ============================================
# 功能：从已有检查点继续微调ULMFiT分类器
# 特性：阈值校准、模型融合
# 使用方法：需要提供ULMFiT预训练的检查点

{continue_code}'''
    })

    # Cell 11: 全量微调（来自finetune_all_from_checkpoint.py）
    finetune_code = files_content['finetune_all_from_checkpoint.py']
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': f'''# ============================================
# 第十部分：全量微调（来自finetune_all_from_checkpoint.py）
# ============================================
# 功能：对所有参数进行微调
# 特性：AdamW优化器、余弦退火学习率调度
# 使用方法：需要提供预训练模型的检查点

{finetune_code}'''
    })

    # Cell 12: 生成提交（来自make_submission_from_checkpoint.py）
    submission_code = files_content['make_submission_from_checkpoint.py']
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': f'''# ============================================
# 第十一部分：生成提交（来自make_submission_from_checkpoint.py）
# ============================================
# 功能：从检查点加载模型并生成测试集预测
# 特性：简单直接的预测流程
# 使用方法：需要提供训练好的模型检查点

{submission_code}'''
    })

    # Cell 13: NB-LSTM混合（来自nb_lstm_hybrid.py）
    nb_code = files_content['nb_lstm_hybrid.py']
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': f'''# ============================================
# 第十二部分：NB-LSTM混合模型（来自nb_lstm_hybrid.py）
# ============================================
# 功能：朴素贝叶斯与LSTM的集成模型
# 特性：n-gram哈希特征、网格搜索最优融合权重
# 使用方法：需要ULMFiT或LSTM模型的检查点

{nb_code}'''
    })

    # Cell 14: 使用示例
    notebook['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': '''## 📝 使用示例

### 示例1：训练基础LSTM分类器
```python
# 首先确保已上传数据文件
# 然后运行以下代码（需要根据实际文件名调整参数）

args = argparse.Namespace(
    train="train.csv",
    unlabel="train_unlabel.csv",
    test="test.csv",
    out_dir="runs_scratch_lstm",
    submission="submission.csv",
    seq_len=320,
    min_count=2,
    max_vocab=90000,
    emb_dim=256,
    hidden_dim=224,
    dropout=0.35,
    batch_size=128,
    epochs=8,
    lr=2e-3,
    weight_decay=1e-4,
    grad_clip=1.0,
    warmup_ratio=0.08,
    label_smoothing=0.0,
    valid_ratio=0.1,
    seed=2026,
    amp=True,
    include_test_vocab=True
)

# 调用main函数开始训练
# main()  # 取消注释以运行
```

### 示例2：训练ULMFiT模型
```python
# ULMFiT分为两个阶段：语言模型预训练 + 分类器微调
# 第一阶段：预训练语言模型
args = argparse.Namespace(
    train="train.csv",
    unlabel="train_unlabel.csv",
    test="test.csv",
    out_dir="runs_ulmfit_scratch",
    submission="submission_ulmfit.csv",
    seq_len=512,
    bptt=80,
    min_count=3,
    max_vocab=60000,
    emb_dim=256,
    hidden_dim=256,
    layers=2,
    dropout=0.35,
    word_dropout=0.04,
    lm_batch_size=64,
    batch_size=96,
    lm_epochs=2,
    clf_epochs=6,
    lm_lr=0.0015,
    clf_lr=0.001,
    weight_decay=0.0002,
    alpha=1e-4,
    beta=1e-4,
    grad_clip=0.25,
    label_smoothing=0.04,
    valid_ratio=0.1,
    seed=2029,
    amp=True,
    skip_lm=False,
    skip_predict=False
)

# 调用main函数开始训练
# main()  # 取消注释以运行
```

### 提示
- 每个模型都有独立的参数配置
- 建议先在小数据集上测试代码
- 使用GPU可以显著加速训练过程
- 检查点会保存在指定的out_dir目录中

---
'''
    })

    # 保存notebook
    with open('/workspace/text_classification_models.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("✅ Notebook创建成功！")
    print("=" * 60)
    print(f"📁 文件位置：/workspace/text_classification_models.ipynb")
    print(f"📊 单元格数量：{len(notebook['cells'])}")
    print("")
    print("📋 包含的模块:")
    print("   1. 环境设置")
    print("   2. 通用工具函数 (train_scratch_lstm.py)")
    print("   3. ULMFiT模型 (train_ulmfit_scratch.py)")
    print("   4. 强LSTM模型 (train_strong_lstm.py)")
    print("   5. ELMo风格模型 (train_elmo_scratch.py)")
    print("   6. n-gram混合模型 (train_ulmfit_ngram.py)")
    print("   7. VAT半监督模型 (train_ulmfit_vat.py)")
    print("   8. 自训练模型 (selftrain_from_checkpoint.py)")
    print("   9. 继续微调 (continue_ulmfit_finetune.py)")
    print("   10. 全量微调 (finetune_all_from_checkpoint.py)")
    print("   11. 生成提交 (make_submission_from_checkpoint.py)")
    print("   12. NB-LSTM混合模型 (nb_lstm_hybrid.py)")
    print("   13. 使用示例说明")
    print("")
    print("🚀 使用方法:")
    print("   1. 将text_classification_models.ipynb上传到Google Colab")
    print("   2. 选择 运行时 -> 更改运行时类型 -> GPU")
    print("   3. 上传数据文件 (train.csv, test.csv, train_unlabel.csv)")
    print("   4. 按顺序运行单元格")
    print("=" * 60)

if __name__ == "__main__":
    create_notebook()
