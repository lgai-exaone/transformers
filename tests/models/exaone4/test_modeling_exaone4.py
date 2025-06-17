# coding=utf-8
# Copyright 2025 The LG AI Research and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch EXAONE 4.0 model."""

import unittest

import pytest
from packaging import version
from parameterized import parameterized

from transformers import (
    AutoTokenizer,
    Exaone4Config,
    GenerationConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        Exaone4ForCausalLM,
        Exaone4ForQuestionAnswering,
        Exaone4ForSequenceClassification,
        Exaone4ForTokenClassification,
        Exaone4Model,
    )


class Exaone4ModelTester:
    config_class = Exaone4Config
    if is_torch_available():
        model_class = Exaone4Model
        for_causal_lm_class = Exaone4ForCausalLM
        for_sequence_class = Exaone4ForSequenceClassification
        for_token_class = Exaone4ForTokenClassification

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        dtype=torch.float32,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="silu",
        max_position_embeddings=512,
        attention_dropout=0.0,
        reorder_qk_norm=True,
        sliding_window=50,
        sliding_window_pattern="LG",
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.dtype = dtype
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.reorder_qk_norm = reorder_qk_norm
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return Exaone4Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            attention_dropout=self.attention_dropout,
            reorder_qk_norm=self.reorder_qk_norm,
            sliding_window=self.sliding_window,
            sliding_window_pattern=self.sliding_window_pattern,
            torch_dtype=self.dtype,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = Exaone4Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class Exaone4ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Exaone4Model,
            Exaone4ForCausalLM,
            Exaone4ForSequenceClassification,
            Exaone4ForQuestionAnswering,
            Exaone4ForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": Exaone4Model,
            "question-answering": Exaone4ForQuestionAnswering,
            "text-classification": Exaone4ForSequenceClassification,
            "text-generation": Exaone4ForCausalLM,
            "zero-shot": Exaone4ForSequenceClassification,
            "token-classification": Exaone4ForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez
    model_split_percents = [0.5, 0.6]

    def setUp(self):
        self.model_tester = Exaone4ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Exaone4Config, hidden_size=37)

    @unittest.skip("Failing because of unique cache (HybridCache)")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @unittest.skip("EXAONE 4.0 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("EXAONE 4.0 has HybridCache which is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @pytest.mark.generate
    @unittest.skip("EXAONE 4.0 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("EXAONE 4.0 has HybridCache which is not compatible with dola decoding")
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip("EXAONE 4.0 has HybridCache and doesn't support continue from past kv")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip("EXAONE 4.0 has HybridCache and doesn't support low_memory generation")
    def test_beam_search_low_memory(self):
        pass

    @unittest.skip("EXAONE 4.0 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("EXAONE 4.0 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("EXAONE 4.0 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip(
        "EXAONE 4.0 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support."
    )
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip(
        "EXAONE 4.0 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support."
    )
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip(
        "EXAONE 4.0 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support."
    )
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @unittest.skip("EXAONE 4.0 has HybridCache which auto-compiles. Compile and FA2 don't work together.")
    def test_eager_matches_fa2_generate(self):
        pass

    @unittest.skip(
        reason="HybridCache can't be gathered because it is not iterable. Adding a simple iter and dumping `distributed_iterator`"
        " as in Dynamic Cache doesnt work. NOTE: @gante all cache objects would need better compatibility with multi gpu setting"
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_Exaone4_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = self.model_tester.for_sequence_class(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_Exaone4_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = self.model_tester.for_sequence_class(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_Exaone4_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = self.model_tester.for_sequence_class(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_Exaone4_token_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        token_labels = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.num_labels)
        model = self.model_tester.for_token_class(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=token_labels)
        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_labels),
        )


@require_torch
class Exaone4IntegrationTest(unittest.TestCase):
    TEST_MODEL_ID = "/home/junwon_hwang/exaone_models/v40/exaone_32b_4k_qknorm_exaone"  # temporary model

    def tearDown(self):
        # TODO (joao): automatic compilation, i.e. compilation when `cache_implementation="static"` is used, leaves
        # some memory allocated in the cache, which means some object is not being released properly. This causes some
        # unoptimal memory usage, e.g. after certain tests a 7B model in FP16 no longer fits in a 24GB GPU.
        # Investigate the root cause.
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_logits(self):
        input_ids = [405, 7584, 79579, 76636, 2907, 94640, 373]
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID, device_map="auto", torch_dtype=torch.float16, attn_implementation="eager"
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()

        EXPECTED_MEAN = torch.tensor([[48.7355, 39.5970, 31.9457, 15.9101, 4.9173, 23.4806, 9.7128]])
        EXPECTED_SLICE = torch.tensor(
            [
                48.0000,
                50.0625,
                48.7812,
                49.5000,
                52.5625,
                50.2812,
                51.9062,
                51.7500,
                51.6562,
                51.8438,
                51.9688,
                51.7812,
                51.7500,
                51.7188,
                51.4375,
                52.5938,
                51.9375,
                51.9062,
                52.2188,
                53.0000,
                53.0312,
                52.5000,
                52.6562,
                53.0000,
                53.0312,
                53.1875,
                52.9062,
                53.3750,
                53.5312,
                53.3750,
            ]
        )

        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-4, rtol=1e-4)
        del model
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_logits_bf16(self):
        input_ids = [405, 7584, 79579, 76636, 2907, 94640, 373]
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()

        EXPECTED_MEAN = torch.tensor([[49.0620, 39.1866, 31.7134, 15.9796, 4.8780, 23.5991, 9.5953]])
        EXPECTED_SLICE = torch.tensor(
            [
                48.2500,
                50.5000,
                49.0000,
                49.7500,
                53.0000,
                50.7500,
                52.2500,
                52.0000,
                52.0000,
                52.2500,
                52.2500,
                52.0000,
                52.0000,
                52.0000,
                51.7500,
                52.7500,
                52.2500,
                52.2500,
                52.5000,
                53.2500,
                53.2500,
                52.7500,
                53.0000,
                53.2500,
                53.5000,
                53.5000,
                53.2500,
                53.7500,
                53.7500,
                53.7500,
            ]
        )

        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-4, rtol=1e-4)
        del model
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_generation(self):
        EXPECTED_TEXT = 'Tell me about the Miracle on the Han river.\n\nThe Miracle on the Han river is a term used to describe the rapid economic growth and development of South Korea since the 1960s. It refers to the transformation of the country from a poor, agrarian society to a modern, industrialized nation with a strong economy.\n\nThe term "Miracle on the Han river" was coined by British journalist and writer, James E. Fraser, in 1960, after he visited South Korea and was impressed by the rapid changes he saw. He compared the economic growth of South Korea to the "Miracle of the Rhine" in Germany after World War II'
        prompt = "Tell me about the Miracle on the Han river."
        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID)
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID, device_map="auto", torch_dtype=torch.float16, attn_implementation="eager"
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=128, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, text)
        del model
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_torch_sdpa
    def test_model_generation_bf16_sdpa(self):
        EXPECTED_TEXT = "Tell me about the Miracle on the Han river.\n\nThe Miracle on the Han River is a term used to describe the rapid economic growth and development of South Korea since the 1960s. It refers to the transformation of the country from a poor, agrarian society to a modern, industrialized nation with a strong economy.\n\nThe term \"Miracle on the Han River\" was coined by British journalist and writer, James E. Moore, in 1960, after he visited South Korea and was impressed by the country's rapid economic development. He used the term to describe the remarkable growth of the country's economy, which had grown at an"
        prompt = "Tell me about the Miracle on the Han river."
        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID)
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=128, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, text)
        del model
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_torch_accelerator
    @require_flash_attn
    def test_model_generation_long_flash(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [433, 9055]
        input_ids = [433, 9055] * 2048
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID, device_map="auto", torch_dtype=torch.float16, attn_implementation="flash_attention_2"
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)

        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())
        del model
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_torch_accelerator
    @require_torch_sdpa
    def test_model_generation_beyond_sliding_window(self):
        EXPECTED_TEXT_COMPLETION = " the weather, and the people here.import { Component, OnInit } from '@angular/core';\nimport { ActivatedRoute, Router }"
        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID)
        prompt = "This is a nice place. " * 700 + "I really enjoy the scenery,"
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID, device_map="auto", torch_dtype=torch.float16, attn_implementation="sdpa"
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        generated_ids = model.generate(input_ids, max_new_tokens=32, temperature=0)
        text = tokenizer.decode(generated_ids[0, -32:], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
        del model
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_export_static_cache(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
            convert_and_export_with_cache,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID, padding_side="right")
        EXPECTED_TEXT_COMPLETION = ["The Deep Learning is 100% based on the concept of Neural Networks.\n\n"]
        max_generation_length = tokenizer(EXPECTED_TEXT_COMPLETION, return_tensors="pt", padding=True)[
            "input_ids"
        ].shape[-1]

        # Load model
        device = "cpu"
        dtype = torch.bfloat16
        cache_implementation = "static"
        attn_implementation = "sdpa"
        batch_size = 1
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID,
            device_map=device,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            generation_config=GenerationConfig(
                use_cache=True,
                cache_implementation=cache_implementation,
                max_length=max_generation_length,
                cache_config={
                    "batch_size": batch_size,
                    "max_cache_len": max_generation_length,
                },
            ),
        )

        prompt = ["The Deep Learning is "]
        prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        prompt_token_ids = prompt_tokens["input_ids"]
        max_new_tokens = max_generation_length - prompt_token_ids.shape[-1]

        # Static Cache + export
        exported_program = convert_and_export_with_cache(model)
        ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
            exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
        )
        ep_generated_text = tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)
