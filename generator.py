import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed,
)

MODEL_ID = "MiniMaxAI/MiniMax-M2"
SAVE_FOLDER_TMP = Path("./minimax-m2-tiny-random-tmp")
SAVE_FOLDER = Path("./minimax-m2-tiny-random")


def prepare_tokenizer_and_generation_config(base_model_id: str, save_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(save_dir)
    generation_config = GenerationConfig.from_pretrained(
        base_model_id,
        trust_remote_code=True,
    )
    generation_config.save_pretrained(save_dir)


def build_tiny_config(base_model_id: str, save_dir: Path):
    with open(
        hf_hub_download(base_model_id, filename="config.json", repo_type="model"),
        "r",
        encoding="utf-8",
    ) as f:
        config_json = json.load(f)

    config_json["head_dim"] = 128
    config_json["hidden_size"] = 256
    config_json["intermediate_size"] = 128
    config_json["num_attention_heads"] = 2
    config_json["num_experts_per_tok"] = 2
    config_json["num_hidden_layers"] = 2
    config_json["num_key_value_heads"] = 1
    config_json["num_local_experts"] = 8
    config_json["rotary_dim"] = 64
    config_json["tie_word_embeddings"] = True

    config_json.pop("auto_map", None)

    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2, ensure_ascii=False)


def init_random_tiny_model(save_dir: Path):
    config = AutoConfig.from_pretrained(
        save_dir,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_config(config)

    set_seed(42)
    model = model.cpu()
    print(model)

    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            torch.nn.init.normal_(p, mean=0.0, std=0.2)

    model.save_pretrained(save_dir)
    print("=" * 100)
    print("Stage 1 model saved to:", save_dir)


def reload_with_quant_and_resave(save_dir_tmp: Path, save_dir: Path):
    print("=" * 100)
    print("Stage 2: reload with quant and resave")

    model = AutoModelForCausalLM.from_pretrained(
        save_dir_tmp,
        trust_remote_code=True,
    )

    model.save_pretrained(save_dir)
    print("Stage 2 model saved to:", save_dir)


def test_loaded_model(save_dir: Path):
    print("=" * 100)
    print("Testing loaded model")
    model = AutoModelForCausalLM.from_pretrained(
        save_dir,
        trust_remote_code=True,
    )
    print(model)


if __name__ == "__main__":
    SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
    prepare_tokenizer_and_generation_config(MODEL_ID, SAVE_FOLDER_TMP)
    build_tiny_config(MODEL_ID, SAVE_FOLDER_TMP)
    init_random_tiny_model(SAVE_FOLDER_TMP)
    reload_with_quant_and_resave(SAVE_FOLDER_TMP, SAVE_FOLDER)
    test_loaded_model(SAVE_FOLDER)
