# MiniMaxM2TinyModelGenerator

## Params

```python
config_json['head_dim'] = 128
config_json['hidden_size'] = 256
config_json['intermediate_size'] = 128
config_json['num_attention_heads'] = 2
config_json['num_experts_per_tok'] = 2
config_json['num_hidden_layers'] = 2
config_json['num_key_value_heads'] = 1
config_json['num_local_experts'] = 8
config_json['rotary_dim'] = 64
config_json['tie_word_embeddings'] = True
```

## Usage

Install:

```bash
uv sync
```

Generate:

```bash
uv run generator.py
```

## Output

The model will be generated in the `./minimax-m2-tiny-random` directory.

```
minimax-m2-tiny-random/
├── config.json              # 模型配置文件
├── model.safetensors         # 模型权重文件
├── tokenizer.json           # 分词器配置
├── tokenizer_config.json    # 分词器参数
├── special_tokens_map.json  # 特殊标记映射
└── generation_config.json   # 生成配置（如果存在）
```

## License

It uses the same license as the original MiniMax-M2 model.
