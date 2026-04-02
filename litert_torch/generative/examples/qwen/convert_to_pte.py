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

"""Example of converting Qwen 2.5 models to multi-signature tflite model."""
import torch
import transformers
from absl import app
from litert_torch.generative.examples.qwen import qwen
from litert_torch.generative.utilities import converter

flags = converter.define_conversion_flags('qwen')

_MODEL_SIZE = flags.DEFINE_enum(
    'model_size',
    '0.5b',
    ['0.5b', '1.5b', '3b'],
    'The size of the model to convert.',
)

_BUILDER = {
    '0.5b': qwen.build_0_5b_model,
    '1.5b': qwen.build_1_5b_model,
    '3b': qwen.build_3b_model,
}


def main(_):
    checkpoint_path = "/home/donghao/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
    file_name = converter.build_and_convert_to_pte_from_flags(model_builder=_BUILDER[_MODEL_SIZE.value],
                                                              checkpoint_path=checkpoint_path)


if __name__ == '__main__':
    app.run(main)
