# coding=utf-8
# Copyright 2025 The LG AI Research and HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert old EXAONE model to EXAONE 4.0 compatible model."""

import argparse

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.exaone4.configuration_exaone4 import Exaone4Config
from transformers.models.exaone4.modeling_exaone4 import Exaone4ForCausalLM


def remap_key(key, reorder_qk_norm, verbose=False):
    rename_mapping = {
        "transformer": "model",
        "h": "layers",
        "ln_f": "norm",
        "ln_1": "input_layernorm" if not reorder_qk_norm else "post_attention_layernorm",
        "ln_2": "pre_feedforward_layernorm" if not reorder_qk_norm else "post_feedforward_layernorm",
        "out_proj": "o_proj",
        "c_fc_0": "gate_proj",
        "c_fc_1": "up_proj",
        "c_proj": "down_proj",
        "wte": "embed_tokens",
    }

    attrs = key.split(".")
    new_attrs = []
    for attr in attrs:
        if attr in rename_mapping:
            new_attrs.append(rename_mapping[attr])
        else:
            new_attrs.append(attr)
    final = ".".join(new_attrs)
    final = final.replace("attn.attention", "self_attn")
    if verbose:
        print(f"Renamed Key: {key} -> {final}")
    return final


def convert_config(old_config, verbose=False):
    cfg = old_config.to_dict()
    print(cfg)

    config_rename_mapping = {
        "num_layers": "num_hidden_layers",
        "activation_function": "hidden_act",
        "layer_norm_epsilon": "rms_norm_eps",
    }
    config_drop_names = [
        "auto_map",
        "embed_dropout",
    ]
    config_update_dict = {
        "reorder_qk_norm": False,
        "architectures": ["Exaone4ForCausalLM"],
        "sliding_window": None,
        "sliding_window_pattern": None,
        "tokenizer_class": "GPT2Tokenizer",
    }

    for old_name, new_name in config_rename_mapping.items():
        val = cfg.pop(old_name)
        cfg[new_name] = val
        if verbose:
            print(f"Renamed Config: {old_name} -> {new_name} ({val})")
    for name in config_drop_names:
        cfg.pop(name)
        if verbose:
            print(f"Dropped Config: {name}")
    for key, value in config_update_dict.items():
        cfg[key] = value
        if verbose:
            print(f"Set Config: {key} -> {value}")

    return Exaone4Config.from_dict(cfg)


def convert_model(old_model, new_config, verbose=False):
    with torch.device("meta"):
        new_model = Exaone4ForCausalLM(new_config)

    empty_state_dict = {
        name: torch.zeros_like(param, dtype=new_config.torch_dtype, device="cpu")
        for name, param in new_model.state_dict().items()
    }
    new_model = Exaone4ForCausalLM.from_pretrained(
        None, state_dict=empty_state_dict, config=new_config, torch_dtype=new_config.torch_dtype
    )
    print("Empty EXAONE model created")

    state_dict = old_model.state_dict()

    del old_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = remap_key(key, new_config.reorder_qk_norm, verbose)
        if new_key in new_model.state_dict():
            new_state_dict[new_key] = value

    missing_keys, unexpected_keys = new_model.load_state_dict(new_state_dict, strict=False)
    print(f"Weight loaded. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    if verbose and (missing_keys or unexpected_keys):
        print(f"Missing keys: {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}")
        print(f"Unexpected keys: {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}")

    return new_model


def main(args):
    print(f"Converting old EXAONE model ({args.old_model}) to EXAONE 4.0 compatible model ({args.new_model})...")
    old_model = AutoModelForCausalLM.from_pretrained(args.old_model, trust_remote_code=True)
    new_config = convert_config(old_model.config, args.verbose)
    print(new_config)
    new_model = convert_model(old_model, new_config, args.verbose)
    print(new_model)
    new_model.save_pretrained(args.new_model)

    tokenizer = AutoTokenizer.from_pretrained(args.old_model)
    tokenizer.save_pretrained(args.new_model)
    print("Conversion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_model", type=str, required=True)
    parser.add_argument("--new_model", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    main(args)
