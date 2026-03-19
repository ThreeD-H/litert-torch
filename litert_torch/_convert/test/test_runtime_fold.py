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
"""Tests for runtime fold."""

from litert_torch._convert import runtime_fold as runtime_fold_lib
from absl.testing import absltest as googletest
from ai_edge_litert.tools.model_utils import model_builder
from ai_edge_litert.tools.model_utils import testing
from ai_edge_litert.tools.model_utils import transform as mu_transform
from ai_edge_litert.tools.model_utils.dialect import mlir
from ai_edge_litert.tools.model_utils.dialect import tfl


runtime_fold = runtime_fold_lib.runtime_fold


class RuntimeFoldTest(testing.ModelUtilsTestCase):
  """Tests for runtime fold."""

  def test_fold_constants(self):

    @model_builder.build_module_from_py_func(mlir.RankedTensorType([2], "f32"))
    def module(x):
      c1 = tfl.const([1.0, 2.0])
      c2 = tfl.const([3.0, 4.0])
      cst = tfl.add(c1, c2)
      z = tfl.add(x, cst)
      return z

    ir_module = mu_transform.model_utils_to_mlir(module)
    ir_module = runtime_fold(self.ir_context, ir_module)
    self.assert_filecheck(
        ir_module,
        """
        CHECK: %[[CST:.*]] = arith.constant
        CHECK: %[[ADD:.*]] = tfl.add %arg0, %[[CST]] {fused_activation_function = "NONE"} : tensor<2xf32>
        CHECK: return %[[ADD]]
        """,
    )

  def test_fold_multiple_constants(self):
    @model_builder.build_module_from_py_func(mlir.RankedTensorType([2], "f32"))
    def module(x):
      c11 = tfl.const([1.0, 2.0])
      c12 = tfl.const([3.0, 4.0])
      cst1 = tfl.add(c11, c12)
      x = tfl.add(x, cst1)
      c21 = tfl.const([5.0, 6.0])
      c22 = tfl.const([7.0, 8.0])
      cst2 = tfl.add(c21, c22)
      x = tfl.mul(x, cst2)
      return x

    ir_module = mu_transform.model_utils_to_mlir(module)
    ir_module = runtime_fold(self.ir_context, ir_module)
    self.assert_filecheck(
        ir_module,
        """
        CHECK: %[[CST2:.*]] = arith.constant
        CHECK: %[[CST1:.*]] = arith.constant
        CHECK: %[[ADD:.*]] = tfl.add %arg0, %[[CST1]] {fused_activation_function = "NONE"} : tensor<2xf32>
        CHECK: %[[MUL:.*]] = tfl.mul %[[ADD]], %[[CST2]] {fused_activation_function = "NONE"} : tensor<2xf32>
        CHECK: return %[[MUL]]
        """,
    )


if __name__ == "__main__":
  googletest.main()
