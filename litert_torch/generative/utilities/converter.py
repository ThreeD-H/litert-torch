# Copyright 2024 The LiteRT Torch Authors.
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
# ==============================================================================

"""Common utility functions for model conversion."""

import enum
import os
import pathlib
import tempfile
from typing import Callable, Dict, Optional, Union

import transformers
from absl import flags
from litert_torch._convert import interface as converter_utils
from litert_torch.generative.layers import kv_cache as kv_utils
from litert_torch.generative.layers import lora as lora_utils
import litert_torch.generative.layers.model_config as cfg
from litert_torch.generative.quantize import quant_attrs
from litert_torch.generative.quantize import quant_recipes
from litert_torch.generative.utilities import export_config as export_config_lib
from litert_torch.generative.utilities import litertlm_builder
from litert_torch.generative.utilities import loader
from litert_torch.quantize import quant_config as qcfg
import torch

ExportConfig = export_config_lib.ExportConfig


class ExportableModule(torch.nn.Module):

    def __init__(self, module, **extra_kwargs):
        super().__init__()
        self.module = module
        self.extra_kwargs = extra_kwargs

    def forward(self, *export_args, **export_kwargs):
        full_kwargs = {**export_kwargs, **self.extra_kwargs}
        return self.module(*export_args, **full_kwargs)


class QuantizationName(str, enum.Enum):
    """Strings for all supported quantization recipes.

    none: No quantization.
    dynamic_int8: Dynamic range quantization with int8 weights.
    weight_only_int8: Weight only quantization with int8 weights.
    fp16: Float16 quantization.
    dynamic_int4_block32: Dynamic range quantization with int4 weights and block
    size of 32, better model quality but slower inference.
    dynamic_int4_block128: Dynamic range quantization with int4 weights and block
    size of 128, faster inference but worse model quality.
    """

    NONE = 'none'
    DYNAMIC_INT8 = 'dynamic_int8'
    WEIGHT_ONLY_INT8 = 'weight_only_int8'
    FP16 = 'fp16'
    DYNAMIC_INT4_BLOCK32 = 'dynamic_int4_block32'
    DYNAMIC_INT4_BLOCK128 = 'dynamic_int4_block128'


def define_conversion_flags(
        model_name: str,
        default_mask_as_input: bool = False,
        default_transpose_kv_cache: bool = False,
):
    """Defines common flags used for model conversion."""

    flags.DEFINE_string(
        'checkpoint_path',
        os.path.join(pathlib.Path.home(), f'Downloads/llm_data/{model_name}'),
        'The path to the model checkpoint, or directory holding the checkpoint.',
    )
    flags.DEFINE_string(
        'output_path',
        '/tmp/',
        'The path to export the tflite model.',
    )
    flags.DEFINE_string(
        'output_name_prefix',
        f'{model_name}',
        'The prefix of the output tflite model name.',
    )
    flags.DEFINE_integer(
        'prefill_seq_len',
        64,
        'List of the maximum sizes of prefill input tensors.',
    )
    flags.DEFINE_integer(
        'decode_batch_size',
        1,
        'The batch size for the decode signature.',
    )
    flags.DEFINE_integer(
        'kv_cache_max_len',
        512,
        'The maximum size of KV cache buffer, including both prefill and decode.',
    )
    flags.DEFINE_bool(
        'use_quantize',
        False,
        'How the model should be quantized. Set to "none" to disable '
        'quantization. See `QuantDtype` for supported quantization types.',
    )
    flags.DEFINE_multi_integer(
        'lora_ranks',
        None,
        'If set, the model will be converted with the provided list of LoRA '
        'ranks.',
    )
    flags.DEFINE_bool(
        'mask_as_input',
        default_mask_as_input,
        'If true, the mask will be passed in as input. Otherwise, mask will be '
        'built by the model internally.',
    )
    flags.DEFINE_bool(
        'transpose_kv_cache',
        default_transpose_kv_cache,
        'If true, the model will be converted with transposed KV cache.',
    )
    flags.DEFINE_bool(
        'custom_checkpoint_loader',
        False,
        'If true, the conversion script will use a custom checkpoint loader '
        'which will read a checkpoint from a remote source.',
    )
    flags.DEFINE_bool(
        'gpu_dynamic_shapes',
        False,
        'It is to support dynamic shapes on GPU effectively. If true, the graph '
        'sets the actual kv_cache size and prefill lengths when the graph is '
        'initialized for inference based on the flags, `kv_cache_max_len` and '
        '`prefill_seq_lens` as the maximum of kv_cache size and prefill lengths '
        'in the graph.',
    )
    flags.DEFINE_bool(
        'export_gpu_dynamic_shape_verifications',
        False,
        'If true, the conversion script will export signatures used only for '
        'verification of GPU dynamic shapes.',
    )
    flags.DEFINE_string(
        'backend',
        'qnn',
        'How the model use backends '
        'backend：cpu,qnn',
    )
    return flags


# Context length for verifying GPU dynamic shapes.
_CONTEXT_LENGTH_TO_VERIFY_MAGIC_NUMBERS = 1280
# Long prefill length for verifying GPU dynamic shapes.
_LONG_PREFILL_LENGTH_TO_VERIFY_MAGIC_NUMBERS = 1024
# Short prefill length for verifying GPU dynamic shapes.
_SHORT_PREFILL_LENGTH_TO_VERIFY_MAGIC_NUMBERS = 64


def is_magic_number_(num: int) -> bool:
    """Returns true if the number is a magic number, i.e. prime number > 10."""
    if num < 10:
        return False
    if num % 2 == 0:
        return False
    for i in range(3, int(num / 2), 2):
        if num % i == 0:
            return False
    return True


def get_magic_number_for(org_number: int) -> int:
    """Returns the magic number for the given original number."""
    while not is_magic_number_(org_number):
        org_number += 1
    return org_number


def get_mask_cache_size_from_flags() -> int:
    """Returns the mask cache size according to the flags."""
    if flags.FLAGS.mask_as_input:
        return 0
    if flags.FLAGS.gpu_dynamic_shapes:
        return get_magic_number_for(flags.FLAGS.kv_cache_max_len)
    return flags.FLAGS.kv_cache_max_len


def get_quant_recipe_from_flag(
        quantize: str,
        model_config: cfg.ModelConfig,
) -> Optional[qcfg.QuantConfig]:
    """Processes the quantization flag and returns the corresponding recipe.

    Args:
        quantize: The quantization type.

    Returns:
        The quantization recipe, or None if no quantization is needed.

    Raises:
        ValueError: If the quantization type is not supported.
    """
    match quantize:
        case QuantizationName.NONE:
            return None
        case QuantizationName.DYNAMIC_INT8:
            return quant_recipes.full_dynamic_recipe(mcfg=model_config)
        case QuantizationName.WEIGHT_ONLY_INT8:
            return quant_recipes.full_weight_only_recipe(mcfg=model_config)
        case QuantizationName.FP16:
            return quant_recipes.full_fp16_recipe()
        case QuantizationName.DYNAMIC_INT4_BLOCK32:
            return quant_recipes.full_dynamic_recipe(
                mcfg=model_config,
                weight_dtype=quant_attrs.Dtype.INT4,
                granularity=quant_attrs.Granularity.BLOCKWISE_32,
            )
        case QuantizationName.DYNAMIC_INT4_BLOCK128:
            return quant_recipes.full_dynamic_recipe(
                mcfg=model_config,
                weight_dtype=quant_attrs.Dtype.INT4,
                granularity=quant_attrs.Granularity.BLOCKWISE_128,
            )
        case _:
            raise ValueError(f'Unsupported quantization flag: {quantize}')


def create_quantize_suffix(quantize: str) -> str:
    """Creates a suffix for the output file name based on the quantization type.

    Args:
        quantize: The quantization type.

    Returns:
        A string representing the quantization suffix.

    Raises:
        ValueError: If the quantization type is not supported.
    """
    match quantize:
        case QuantizationName.NONE:
            return 'f32'
        case QuantizationName.DYNAMIC_INT8:
            return 'q8'
        case QuantizationName.WEIGHT_ONLY_INT8:
            return 'q8_wo'
        case QuantizationName.FP16:
            return 'fp16'
        case QuantizationName.DYNAMIC_INT4_BLOCK32:
            return 'q4_block32'
        case QuantizationName.DYNAMIC_INT4_BLOCK128:
            return 'q4_block128'
        case _:
            raise ValueError(f'Unsupported quantization flag: {quantize}')


def _build_mask(mask_len, kv_cache_max_len, causal_mask_value) -> torch.Tensor:
    if isinstance(mask_len, list):
        return [
            _build_mask(i, kv_cache_max_len, causal_mask_value) for i in mask_len
        ]

    mask = torch.full(
        (mask_len, kv_cache_max_len), causal_mask_value, dtype=torch.float32
    )
    return torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)


def convert_to_tflite(
        pytorch_model: torch.nn.Module,
        output_path: str,
        output_name_prefix: str,
        prefill_seq_len: Union[int, list[int]],
        kv_cache_max_len: int,
        pixel_values_size: torch.Size = None,
        pixel_seq_len: int = 0,
        quantize: str = 'dynamic_int8',
        config: cfg.ModelConfig = None,
        lora_ranks: Optional[list[int]] = None,
        export_config: ExportConfig = None,
        extra_model: torch.nn.Module = None,
        extra_prefill_seq_lens: list[int] = None,
        extra_kv_cache_max_len: int = 0,
        extra_signature_prefix: str = '',
):
    """Converts a nn.Module model to multi-signature tflite model.

    A PyTorch model will be converted to a tflite model with several signatures:
      * "prefill_[prefill_seq_len]" (or "prefill" if only one prefill_seq_len is
          passed),
      * "prefill_[preill_seq_len]_pixel" (or "prefill_pixel" if only one
          prefill_seq_len is passed) if num_pixel_values > 0, and
      * "decode".

    "prefill_[prefill_seq_len]" (or "prefill" if only one prefill_seq_len is
    passed) signature takes as a sample input:
      * a tensor of shape [1, prefill_seq_len] of token sequence,
      * a tensor of shape [1, prefill_seq_len] of token positions, and
      * an external KV cache.

    If num_pixel_values > 0, "prefill_[prefill_seq_len]_pixel" (or "prefill_pixel"
    if only one prefill_seq_len is passed) signature takes as a sample input:
      * a tensor of shape [1, prefill_seq_len] of token sequence,
      * a tensor of shape [1, prefill_seq_len] of token positions,
      * an external KV cache, and
      * a tensor of shape [1, num_pixel_values] of pixel values.

    "decode" signature takes as a sample input:
      * a tensor of shape [1, 1] of token sequence,
      * a tensor of shape [1, 1] of the token position, and
      * an external KV cache.

    The final tflite model will be exported to tflite_path.

    Args:
        pytorch_model (torch.nn.Module): PyTorch model to convert to tflite.
        output_path (str): The path to export the tflite model.
        output_name_prefix (str): The prefix of the tflite model name.
        prefill_seq_len (Union[int, list[int]]): The prefill sequence length to
          use. If a list, the model will have multiple prefill signatures.
        kv_cache_max_len (int): The maximum size of KV cache buffer, including
          both prefill and decode.
        pixel_values_size (torch.Size, optional): The size of pixel values to pass
          to the model. If None, the model is not expected to take pixel values.
        pixel_seq_len (int, optional): The length of pixel tokens, or pixel
          embeddings generated by the image encoder with pixel values. The actual
          length of prefill_seq_len will be added by pixel_seq_len when pixel
          values are passed.
        quantize (str, optional): The quantization type. Defaults to
          'dynamic_int8'.
        config (cfg.ModelConfig, optional): The model config used to configure KV
          cache. If None, it uses the config of the pytorch_model.
        lora_ranks (list[int], optional): The ranks of the LORA layers. If None,
          no LoRA signatures will be added.
        export_config (ExportConfig, optional): The export configuration. If None,
          it uses the default export configuration.
        extra_model (torch.nn.Module, optional): PyTorch model to export in
          addition to the pytorch_model. This model can have different
          prefill_seq_lens and kv_cache_max_len.
        extra_prefill_seq_lens (list[int], optional): The prefill sequence
          lengths for extra_model. Meaningful only when extra_model is not None.
        extra_kv_cache_max_len (int, optional): The maximum size of KV cache
          buffer for extra_model. Meaningful only when extra_model is not None.
        extra_signature_prefix (str, optional): The prefix of the extra model
          signatures. Meaningful only when extra_model is not None.
    """
    # pylint: disable=protected-access
    torch._dynamo.config.cache_size_limit = 64

    config = config if config else pytorch_model.config
    prefill_seq_lens = (
        [prefill_seq_len] if isinstance(prefill_seq_len, int) else prefill_seq_len
    )
    loras = [None]
    if lora_ranks is not None:
        for rank in lora_ranks:
            lora = lora_utils.LoRA.zeros(rank, config)
            loras.append(lora)

    quant_suffix = create_quantize_suffix(quantize)
    kv_size = kv_cache_max_len
    lora_suffix = (
        '' if not lora_ranks else f'_lora{",".join(map(str, lora_ranks))}'
    )

    if pixel_values_size is not None:
        assert pixel_seq_len > 0, 'pixel_seq_len must be greater than 0'
        max_prefill_seq_len = max(prefill_seq_lens)
        assert kv_size > max_prefill_seq_len + pixel_seq_len, (
            f'The KV cache size ({kv_size}) must be greater than the maximum '
            f'prefill sequence length ({max_prefill_seq_len}) + pixel sequence '
            f'length ({pixel_seq_len})'
        )

    if export_config is not None:
        if export_config.decode_batch_size > 1:
            output_name_prefix += f'_dbs{export_config.decode_batch_size}'

    output_filename = (
        f'{output_name_prefix}_{quant_suffix}_ekv{kv_size}{lora_suffix}.tflite'
    )
    output_file = os.path.join(output_path, output_filename)

    converter = converter_utils.Converter()
    _add_signatures(
        converter,
        pytorch_model,
        prefill_seq_lens,
        kv_cache_max_len,
        pixel_values_size,
        pixel_seq_len,
        config,
        loras,
        export_config,
    )

    if extra_model is not None and extra_prefill_seq_lens:
        _add_signatures(
            converter,
            extra_model,
            extra_prefill_seq_lens,
            extra_kv_cache_max_len,
            pixel_values_size,
            pixel_seq_len,
            config,
            loras,
            export_config,
            signature_prefix=extra_signature_prefix,
        )

    edge_model = converter.convert(
        quant_config=get_quant_recipe_from_flag(quantize, config),
    )
    edge_model.export(output_file)
    return output_file


def _add_signatures(
        converter: converter_utils.Converter,
        pytorch_model: torch.nn.Module,
        prefill_seq_lens: list[int],
        kv_cache_max_len: int,
        pixel_values_size: torch.Size,
        pixel_seq_len: int,
        config: cfg.ModelConfig,
        loras: list[None | lora_utils.LoRA],
        export_config: ExportConfig,
        signature_prefix: str = '',
):
    """Helper function to export a model to tflite."""
    prefill_tokens_list = []
    prefill_input_pos_list = []
    for seq_len in prefill_seq_lens:
        prefill_tokens_list.append(torch.full((1, seq_len), 0, dtype=torch.int))
        prefill_input_pos_list.append(torch.arange(0, seq_len, dtype=torch.int))

    prefill_pixel_values = None
    prefill_tokens_list_with_pixel = []
    prefill_input_pos_list_with_pixel = []
    if pixel_values_size is not None:
        prefill_pixel_values = torch.full(pixel_values_size, 0, dtype=torch.float32)
        for seq_len in prefill_seq_lens:
            prefill_tokens_list_with_pixel.append(
                torch.full((1, seq_len + pixel_seq_len), 0, dtype=torch.int)
            )
            prefill_input_pos_list_with_pixel.append(
                torch.arange(0, seq_len + pixel_seq_len, dtype=torch.int)
            )

    prefill_masks = None
    if export_config.mask_as_input:
        prefill_masks = _build_mask(
            prefill_seq_lens, kv_cache_max_len, config.causal_mask_value
        )
        if not isinstance(prefill_masks, list):
            prefill_masks = [prefill_masks]
        assert len(prefill_masks) == len(prefill_seq_lens)

    decode_token = torch.tensor(
        [[0] for _ in range(export_config.decode_batch_size)], dtype=torch.int
    )
    decode_input_pos = torch.tensor([0], dtype=torch.int)
    prefill_kv = kv_utils.KVCache.from_model_config(
        kv_cache_max_len, config, kv_layout=export_config.kvcache_layout
    )
    decode_kv = kv_utils.KVCache.from_model_config(
        kv_cache_max_len,
        config,
        batch_size=export_config.decode_batch_size,
        kv_layout=export_config.kvcache_layout,
    )

    # For export, we create a module that captures any non-exportable,
    # arugments, e.g. the generation config object.
    mod = ExportableModule(pytorch_model, export_config=export_config).eval()

    for lora in loras:
        for i in range(len(prefill_seq_lens)):
            prefill_seq_len = prefill_seq_lens[i]
            prefill_signature_name = f'{signature_prefix}prefill_{prefill_seq_len}'

            sample_kwargs = {
                'tokens': prefill_tokens_list[i],
                'input_pos': prefill_input_pos_list[i],
                'kv_cache': prefill_kv,
            }
            if prefill_masks is not None:
                sample_kwargs['mask'] = prefill_masks[i]

            if lora is not None:
                prefill_signature_name += f'_lora_r{lora.get_rank()}'
                sample_kwargs['lora'] = lora

            converter.add_signature(
                prefill_signature_name,
                mod,
                sample_kwargs=sample_kwargs,
            )

            if prefill_pixel_values is not None:
                sample_pixel_kwargs = {
                    'tokens': prefill_tokens_list_with_pixel[i],
                    'input_pos': prefill_input_pos_list_with_pixel[i],
                    'kv_cache': prefill_kv,
                    'pixel_values': prefill_pixel_values,
                }
                # mask should be built internally when pixel values are passed.
                if lora is not None:
                    sample_pixel_kwargs['lora'] = lora
                converter.add_signature(
                    prefill_signature_name + '_pixel',
                    mod,
                    sample_kwargs=sample_pixel_kwargs,
                )

        sample_kwargs = {
            'tokens': decode_token,
            'input_pos': decode_input_pos,
            'kv_cache': decode_kv,
        }
        if export_config.mask_as_input:
            # Note that the decode mask is not a correct causal mask, but it is okay
            # for the conversion purpose because only the shape matters in conversion.
            # A correct causal mask of decode for a given token position of decode, it
            # should be built like:
            #
            #  torch.triu(mask, diagonal=decode_position).unsqueeze(0).unsqueeze(0)
            #
            sample_kwargs['mask'] = _build_mask(
                1, kv_cache_max_len, config.causal_mask_value
            )
        if lora is not None:
            sample_kwargs['lora'] = lora

        decode_signature_name = f'{signature_prefix}decode'
        if lora is not None:
            decode_signature_name += f'_lora_r{lora.get_rank()}'
        converter.add_signature(
            decode_signature_name,
            mod,
            sample_kwargs=sample_kwargs,
        )


from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import generate_htp_compiler_spec, generate_qnn_executorch_compiler_spec, \
    to_edge_transform_and_lower_to_qnn
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.runtime import Runtime


def build_and_convert_to_pte_from_flags(model_builder: Callable[
    [str, Callable[[str], Dict[str, torch.Tensor]], int], torch.nn.Module
],
                                        checkpoint_path: str = None,
                                        output_name_prefix: str = None,
                                        export_config: ExportConfig = None):
    if export_config is None:
        export_config = ExportConfig()
        export_config.output_logits_on_prefill = True
        export_config.mask_as_input = True
    backend = flags.FLAGS.backend
    quantize = flags.FLAGS.use_quantize
    if checkpoint_path is None:
        checkpoint_path = flags.FLAGS.checkpoint_path
    if output_name_prefix is None:
        output_name_prefix = flags.FLAGS.output_name_prefix
    if export_config is None:
        export_config = export_config_lib.get_from_flags()

    pytorch_model = model_builder(
        checkpoint_path,
        loader.maybe_get_custom_loader(
            checkpoint_path, flags.FLAGS.custom_checkpoint_loader
        ),
        get_mask_cache_size_from_flags(),
    )
    prefill_seq_len = flags.FLAGS.prefill_seq_len
    kv_cache_max_len = flags.FLAGS.kv_cache_max_len
    print(
        f"backend = {backend}, quantize = {quantize}, prefill_seq_len = {prefill_seq_len}, kv_cache_max_len = {kv_cache_max_len}")
    config = pytorch_model.config
    prefill_kv = kv_utils.KVCache.from_model_config(
        kv_cache_max_len, config, kv_layout=export_config.kvcache_layout
    )
    mod = ExportableModule(pytorch_model, export_config=export_config).eval()
    prefill_masks = None
    if export_config.mask_as_input:
        prefill_masks = _build_mask(
            prefill_seq_len, kv_cache_max_len, config.causal_mask_value
        )
    sample_kwargs = {
        'tokens': torch.full((1, prefill_seq_len), 0, dtype=torch.int),
        'input_pos': torch.arange(0, prefill_seq_len, dtype=torch.int),
        'kv_cache': prefill_kv,
    }
    if prefill_masks is not None:
        sample_kwargs['mask'] = prefill_masks
    print(f"示例输入创建完毕~ sample_kwargs = {sample_kwargs}")
    backend_options = generate_htp_compiler_spec(
        use_fp16=True if not quantize else False,
        use_multi_contexts=False,
        use_weight_sharing=False
    )
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=QcomChipset.SA8295,
        backend_options=backend_options,
        shared_buffer=True
    )
    if quantize:
        print(f"开始导出ExportedProgram...")
        exported_program = torch.export.export(mod, args=tuple(), kwargs=sample_kwargs, dynamic_shapes=None)
        print(f"ExportedProgram导出完毕~")
        quantizer = QnnQuantizer()
        quantizer.set_default_quant_config(
            quant_dtype=QuantDtype.use_16a8w,
            is_qat=False,
            is_conv_per_channel=False,
            is_linear_per_channel=False
        )
        print(f"prepare_pt2e...")
        prepared_model = prepare_pt2e(exported_program.module(), quantizer)
        prepared_model(**sample_kwargs)
        print(f"convert_pt2e...")
        quantized_model = convert_pt2e(prepared_model)
        print(f"量化完毕~")
        sample_inputs = (
            sample_kwargs["tokens"],
            sample_kwargs["input_pos"],
            sample_kwargs["kv_cache"],
        )
        if prefill_masks is not None:
            sample_inputs = (*sample_inputs, sample_kwargs["mask"])

        program = to_edge_transform_and_lower_to_qnn(
            quantized_model,
            compiler_specs=compile_spec,
            args=sample_inputs,
            skip_mutable_buffer=True
        ).to_executorch()
    else:
        if 'qnn' == backend:
            program = to_edge_transform_and_lower_to_qnn(
                mod,
                compiler_specs=compile_spec,
                inputs=sample_kwargs,
                skip_mutable_buffer=True
            ).to_executorch()
        else:
            print(f"开始导出ExportedProgram...")
            exported_program = torch.export.export(mod, args=tuple(), kwargs=sample_kwargs, dynamic_shapes=None)
            print(f"ExportedProgram导出完毕~")
            program = to_edge_transform_and_lower(
                exported_program,
                partitioner=[XnnpackPartitioner()]
            ).to_executorch()
    file_name = f"{output_name_prefix}.pte" if output_name_prefix is not None else f"model_{backend}.pte"
    with open(file_name, "wb") as f:
        f.write(program.buffer)
    print(f"ExportedProgram转换和 lowers完毕~")
    if backend == 'cpu':
        print(f"开始验证edge模型")
        runtime = Runtime.get()
        method = runtime.load_program(file_name).load_method("forward")

        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_path)
        prompt = "你是谁"
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to("cpu")
        tokens = model_inputs["input_ids"]
        tokens = torch.nn.functional.pad(tokens, (0, prefill_seq_len - tokens.size(-1)), "constant",
                                         tokenizer.pad_token_id)
        input_pos = torch.arange(0, prefill_seq_len, dtype=torch.int)
        actual_seq_len = model_inputs["input_ids"].size(-1)
        prefill_masks = create_causal_mask(actual_seq_len, kv_cache_max_len, prefill_seq_len)
        print(f"tokens={tokens}, input_pos={input_pos}, prefill_kv={prefill_kv}, prefill_masks={prefill_masks}")
        flattened_kv_cache = prefill_kv.flatten()
        inputs = [tokens.int(), input_pos] + flattened_kv_cache + [prefill_masks]
        outputs = method.execute(inputs)
        logits = outputs[0]
        last_real_pos = int(actual_seq_len) - 1
        next_token_logits = logits[:, last_real_pos:last_real_pos + 1, :]
        next_token = next_token_logits.argmax(dim=-1)
        next_token_ids = next_token.squeeze(0).tolist()
        print(f"actual_seq_len={actual_seq_len}, last_real_pos={last_real_pos}, next_token_ids={next_token_ids}")
        wrapper_text = tokenizer.decode(next_token_ids, skip_special_tokens=True)
        print(f"edge模型验证完毕~ wrapper_text={wrapper_text}")
        visualize_mask(prefill_masks, actual_seq_len)
    return file_name


def create_causal_mask(actual_seq_len, kv_cache_max_len, mask_len):
    """
    创建因果掩码矩阵

    Args:
        actual_seq_len: 真实序列长度（键的有效长度）
        kv_cache_max_len: KV缓存最大长度
        mask_len: 掩码序列长度

    Returns:
        mask: 形状为 [1, 1, mask_len, kv_cache_max_len] 的掩码矩阵
    """
    # 初始化为全 -inf，形状 (mask_len, kv_cache_max_len)
    mask = torch.full((mask_len, kv_cache_max_len), float('-inf'), dtype=torch.float32)

    if actual_seq_len > 0:
        # 有效查询长度：不超过实际序列长度
        q_len = min(mask_len, actual_seq_len)
        # 创建布尔下三角掩码，表示允许关注的位置
        # 形状 (q_len, actual_seq_len)，下三角（含对角线）为 True
        allow_positions = torch.tril(torch.ones(q_len, actual_seq_len, dtype=torch.bool))
        # 将允许的位置设为 0（注意力分数加 0 不变）
        mask[:q_len, :actual_seq_len][allow_positions] = 0

    # 增加 batch 和 head 维度
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


def visualize_mask(mask, actual_seq_len, title="Causal Mask Visualization"):
    """
    可视化掩码矩阵

    Args:
        mask: 掩码矩阵 [1, 1, mask_len, kv_cache_max_len]
        actual_seq_len: 真实序列长度
        title: 图表标题
    """
    # 提取矩阵（去掉批次和头维度）
    mask_2d = mask[0, 0].numpy()
    #
    # # 创建图形
    # fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    #
    # # 1. 完整矩阵可视化
    # im1 = axes[0].imshow(mask_2d, cmap='viridis', aspect='auto')
    # axes[0].set_title(f'{title}\n完整矩阵 (形状: {mask_2d.shape})')
    # axes[0].set_xlabel('KV Cache位置')
    # axes[0].set_ylabel('序列位置')
    # plt.colorbar(im1, ax=axes[0])
    #
    # # 添加实际序列长度的标记
    # if actual_seq_len > 0:
    #     axes[0].axhline(y=actual_seq_len - 0.5, color='r', linestyle='--', alpha=0.7,
    #                     label=f'实际序列长度={actual_seq_len}')
    #     axes[0].axvline(x=actual_seq_len - 0.5, color='r', linestyle='--', alpha=0.7)
    #     axes[0].legend()
    #
    # # 2. 只显示实际序列部分（前actual_seq_len行和列）
    # if actual_seq_len > 0:
    #     actual_part = mask_2d[:actual_seq_len, :actual_seq_len]
    #     im2 = axes[1].imshow(actual_part, cmap='viridis', aspect='equal')
    #     axes[1].set_title(f'实际序列部分 (形状: {actual_part.shape})')
    #     axes[1].set_xlabel('序列位置 (列)')
    #     axes[1].set_ylabel('序列位置 (行)')
    #     plt.colorbar(im2, ax=axes[1])
    #
    #     # 添加网格线
    #     axes[1].grid(True, alpha=0.3)
    #
    # plt.tight_layout()
    # plt.show()

    # 打印矩阵的数值信息
    print(f"\n掩码矩阵统计信息:")
    print(f"形状: {mask.shape}")
    print(f"实际序列长度: {actual_seq_len}")
    print(f"非-inf元素数量: {torch.sum(mask != float('-inf')).item()}")
    print(f"0值元素数量: {torch.sum(mask == 0).item()}")
    print(f"-inf元素数量: {torch.sum(mask == float('-inf')).item()}")

    # 打印前几行和前几列的示例
    print(f"\n矩阵前{actual_seq_len + 1}行前{actual_seq_len + 1}列示例:")
    print(mask_2d[:actual_seq_len + 1, :actual_seq_len + 1])


def build_and_convert_to_tflite_from_flags(
        model_builder: Callable[
            [str, Callable[[str], Dict[str, torch.Tensor]], int], torch.nn.Module
        ],
        checkpoint_path: str = None,
        output_name_prefix: str = None,
):
    """Builds a nn.Module model and converts it according to the flags."""
    if checkpoint_path is None:
        checkpoint_path = flags.FLAGS.checkpoint_path
    if output_name_prefix is None:
        output_name_prefix = flags.FLAGS.output_name_prefix

    pytorch_model = model_builder(
        checkpoint_path,
        loader.maybe_get_custom_loader(
            checkpoint_path, flags.FLAGS.custom_checkpoint_loader
        ),
        get_mask_cache_size_from_flags(),
    )

    # Extra model for GPU dynamic shape verification if needed.
    extra_model = None
    extra_prefill_seq_lens = None
    extra_kv_cache_max_len = 0
    if flags.FLAGS.gpu_dynamic_shapes:
        prefill_seq_lens = [
            get_magic_number_for(l) for l in flags.FLAGS.prefill_seq_lens
        ]
        kv_cache_max_len = get_magic_number_for(flags.FLAGS.kv_cache_max_len)

        if flags.FLAGS.export_gpu_dynamic_shape_verifications:
            extra_kv_cache_max_len = _CONTEXT_LENGTH_TO_VERIFY_MAGIC_NUMBERS
            if extra_kv_cache_max_len > flags.FLAGS.kv_cache_max_len:
                extra_kv_cache_max_len = flags.FLAGS.kv_cache_max_len
            extra_model = model_builder(
                checkpoint_path,
                loader.maybe_get_custom_loader(
                    checkpoint_path, flags.FLAGS.custom_checkpoint_loader
                ),
                extra_kv_cache_max_len,
            )
            extra_prefill_seq_lens = []
            if extra_kv_cache_max_len > _SHORT_PREFILL_LENGTH_TO_VERIFY_MAGIC_NUMBERS:
                extra_prefill_seq_lens.append(
                    _SHORT_PREFILL_LENGTH_TO_VERIFY_MAGIC_NUMBERS
                )
            if extra_kv_cache_max_len > _LONG_PREFILL_LENGTH_TO_VERIFY_MAGIC_NUMBERS:
                extra_prefill_seq_lens.append(
                    _LONG_PREFILL_LENGTH_TO_VERIFY_MAGIC_NUMBERS
                )
    else:
        prefill_seq_lens = flags.FLAGS.prefill_seq_lens
        kv_cache_max_len = flags.FLAGS.kv_cache_max_len

    convert_to_tflite(
        pytorch_model,
        output_path=flags.FLAGS.output_path,
        output_name_prefix=output_name_prefix,
        prefill_seq_len=prefill_seq_lens,
        kv_cache_max_len=kv_cache_max_len,
        quantize=flags.FLAGS.quantize,
        lora_ranks=flags.FLAGS.lora_ranks,
        export_config=export_config_lib.get_from_flags(),
        extra_model=extra_model,
        extra_prefill_seq_lens=extra_prefill_seq_lens,
        extra_kv_cache_max_len=extra_kv_cache_max_len,
        extra_signature_prefix='test_' if extra_model is not None else '',
    )


def convert_to_litert(
        pytorch_model: torch.nn.Module,
        output_path: str,
        output_name_prefix: str,
        prefill_seq_len: Union[int, list[int]],
        kv_cache_max_len: int,
        pixel_values_size: torch.Size = None,
        pixel_seq_len: int = 0,
        quantize: str = 'dynamic_int8',
        config: cfg.ModelConfig = None,
        lora_ranks: Optional[list[int]] = None,
        export_config: ExportConfig = None,
        output_format: str = 'tflite',
        **kwargs,
):
    """Converts a nn.Module model to multi-signature tflite model and pack it."""
    with tempfile.TemporaryDirectory() as workdir:
        if output_format == 'litertlm':
            tflite_model_output_path = workdir
        else:
            tflite_model_output_path = output_path
        tflite_model_path = convert_to_tflite(
            pytorch_model,
            tflite_model_output_path,
            output_name_prefix,
            prefill_seq_len,
            kv_cache_max_len,
            pixel_values_size,
            pixel_seq_len,
            quantize,
            config,
            lora_ranks,
            export_config,
        )
        if output_format == 'litertlm':
            tokenizer_model_path = kwargs.pop('tokenizer_model_path', None)
            hf_tokenizer_model_path = kwargs.pop('hf_tokenizer_model_path', None)
            litertlm_builder.build_litertlm(
                tflite_model_path=tflite_model_path,
                workdir=workdir,
                output_path=output_path,
                context_length=kv_cache_max_len,
                tokenizer_model_path=tokenizer_model_path,
                hf_tokenizer_model_path=hf_tokenizer_model_path,
                **kwargs,
            )
