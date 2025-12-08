import json
from pathlib import Path

import torch

import accelerate
from huggingface_hub import file_exists, hf_hub_download
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed,
)

source_model_id = "MiniMaxAI/MiniMax-M2"
save_folder = "./minimax-m2-tiny-random"

processor = AutoTokenizer.from_pretrained(source_model_id)
processor.save_pretrained(save_folder)

with open(hf_hub_download(source_model_id, filename='config.json', repo_type='model'), 'r', encoding='utf-8') as f:
    config_json = json.load(f)

config_json["attn_type_list"] = [0, 1]  # one lightning, one attention
for k, v in config_json['auto_map'].items():
    config_json['auto_map'][k] = f'{source_model_id}--{v}'
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

with open(f"{save_folder}/config.json", "w", encoding='utf-8') as f:
    json.dump(config_json, f, indent=2)

config = AutoConfig.from_pretrained(
    save_folder,
    trust_remote_code=True,
)
print(config)
automap = config_json['auto_map']
torch.set_default_dtype(torch.bfloat16)
model = AutoModelForCausalLM.from_config(config)
torch.set_default_dtype(torch.float32)
# according to source model, gat is in FP32
for i in range(config.num_hidden_layers):
    model.model.layers[i].block_sparse_moe.gate.float()
if file_exists(filename="generation_config.json", repo_id=source_model_id, repo_type='model'):
    model.generation_config = GenerationConfig.from_pretrained(
        source_model_id, trust_remote_code=True,
    )
set_seed(42)
model = model.cpu()  # cpu is more stable for random initialization across machines
with torch.no_grad():
    for name, p in sorted(model.named_parameters()):
        torch.nn.init.normal_(p, 0, 0.2)
        print(name, p.shape)
model.save_pretrained(save_folder)
print(model)
with open(f"{save_folder}/config.json", "r", encoding='utf-8') as f:
    config_json = json.load(f)
    config_json['auto_map'] = automap
with open(f"{save_folder}/config.json", "w", encoding='utf-8') as f:
    json.dump(config_json, f, indent=2)
for python_file in Path(save_folder).glob('*.py'):
    python_file.unlink()