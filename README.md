# MiniMaxM2TinyModelGenerator

## 项目简介

MiniMaxM2TinyModelGenerator 是一个用于生成 MiniMax 模型微型版本的工具。该工具基于 MiniMaxAI/MiniMax-M2 原始模型，通过大幅缩减模型参数来创建一个轻量级的随机初始化版本，主要用于开发测试、模型架构验证等场景。

## 模型参数对比

| 参数 | 原始模型 | 微型版本 |
|------|----------|----------|
| 隐藏层大小 | - | 256 |
| 中间层大小 | - | 128 |
| 注意力头数 | - | 2 |
| 键值头数 | - | 1 |
| 隐藏层数 | - | 2 |
| 专家数量 | - | 8 |
| 每次选择专家数 | - | 2 |
| 头维度 | - | 128 |
| 旋转维度 | - | 64 |

## 环境要求

```bash
uv sync
```

## 使用方法

### 基本使用

直接运行脚本生成微型模型：

```bash
uv run generator.py
```

### 输出说明

运行后会在 `./minimax-m2-tiny-random` 目录下生成以下文件：

```
minimax-m2-tiny-random/
├── config.json              # 模型配置文件
├── model.safetensors         # 模型权重文件
├── tokenizer.json           # 分词器配置
├── tokenizer_config.json    # 分词器参数
├── special_tokens_map.json  # 特殊标记映射
└── generation_config.json   # 生成配置（如果存在）
```

## 技术特点

### 混合专家架构
- 保持原始模型的 MoE 架构设计
- 包含 Lightning Attention 和标准 Attention 两种注意力机制
- 专家网络门控模块使用 FP32 精度确保训练稳定性

### 权重初始化策略
- 使用正态分布（均值=0，标准差=0.2）进行随机初始化
- 在 CPU 上进行初始化以确保跨设备一致性
- 门控层保持 FP32 精度

### 模型兼容性
- 完全兼容 Transformers 库的接口
- 支持原始模型的自定义代码和配置
- 保持生成配置的兼容性

## 注意事项

1. **随机性**：模型权重为随机初始化，需要训练后才能正常使用
2. **用途限制**：主要用于开发测试，不适用于生产环境
3. **内存使用**：虽然是微型版本，但仍需要足够的内存来加载模型
4. **精度设置**：模型主体使用 bfloat16，门控层使用 float32

## 自定义配置

如需修改模型参数，可以编辑 `generator.py` 中的配置部分：

```python
config_json['hidden_size'] = 256        # 隐藏层大小
config_json['num_hidden_layers'] = 2    # 层数
config_json['num_attention_heads'] = 2  # 注意力头数
# ... 其他参数
```

## 许可证

请遵循原始 MiniMax 模型的许可证要求。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个工具。