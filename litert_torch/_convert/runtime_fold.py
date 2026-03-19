# Copyright 2026 The LiteRT Torch Authors.
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
"""Fold constants via LiteRT and ModelUtils."""

import copy
import gc
from typing import cast

from litert_torch import progress
from ai_edge_litert.mlir import ir
from xdsl import irdl

from ai_edge_litert.tools import model_utils as mu
from ai_edge_litert.tools.model_utils import transform as mu_transform
from ai_edge_litert.tools.model_utils.dialect import func
from ai_edge_litert.tools.model_utils.dialect import mlir
from ai_edge_litert.tools.model_utils.dialect import stablehlo
from ai_edge_litert.tools.model_utils.dialect import tfl
# pylint: disable=g-direct-tensorflow-import
from ai_edge_litert import interpreter as interpreter_lib
# pylint: enable=g-direct-tensorflow-import

__all__ = ["runtime_fold"]

_CST_ONLY_SIGNATURE = "_CST_ONLY_SIGNATURE"
SSAValue = irdl.SSAValue


def log(message: str):
  progress.log(f"Runtime Fold: {message}")


class RuntimeFoldPass(mu.core.ModulePassBase):
  """The const fold pass via LiteRT."""

  name = "runtime-fold-pass"
  description = "Runtime fold pass"

  def __init__(self, *, fold_dequantize: bool = False):
    super().__init__()
    self.fold_dequantize = fold_dequantize

  def _extract_constants(self, fn: func.FuncOp):
    cst_ops = []
    cst_values: set[SSAValue] = set()
    for op in fn.ops:
      if not self.fold_dequantize and "tfl.dequantize" in op.name:
        continue

      if all(x in cst_values for x in op.operands):
        cst_values.update(op.results)
        cst_ops.append(op)

    output_cst_values = []
    for val in cst_values:
      owner = val.owner
      if (
          owner
          and "tfl.no_value" not in owner.name
          and not isinstance(owner, tfl.ConstOp)
          and any(use.operation not in cst_ops for use in val.uses)
      ):
        output_cst_values.append(val)

    return cst_ops, output_cst_values

  def _extract_composite_decomps(
      self, module: mlir.ModuleOp
  ) -> list[func.FuncOp]:
    decomps = []
    for op in module.walk():
      if isinstance(op, stablehlo.CompositeOp):
        decomps.append(op.decomposition_func)
    return decomps

  def _create_cst_only_module(
      self,
      cst_ops: list[mu.core.MlirOpBase],
      output_cst_values: list[SSAValue],
      composite_decomps: list[func.FuncOp],
  ):
    @mu.model_builder.build_module_from_py_func()
    def cst_only_module():
      value_mapping = {}
      outputs = []
      for op in cst_ops:
        cloned_op = mlir.MlirOp(
            name=op.name,
            operands=[value_mapping[x] for x in op.operands],
            result_types=[x.type for x in op.results],
            attributes=op.attributes,
        )
        for orig_res, cloned_res in zip(op.results, cloned_op.results):
          value_mapping[orig_res] = cloned_res
          if orig_res in output_cst_values:
            outputs.append(cloned_res)
      return outputs

    cst_only_module = cast(mlir.ModuleOp, cst_only_module)
    sig = mu.SignatureBuilder(cst_only_module.ops[0])
    sig.name = _CST_ONLY_SIGNATURE
    sig.output_names = [f"output_{i}" for i in range(len(output_cst_values))]

    prev = cst_only_module.ops[0]
    for decomp in composite_decomps:
      cloned = copy.deepcopy(decomp)
      cloned.detach()
      cst_only_module.body.block.insert_op_after(cloned, prev)

    cst_only_module.cleanup()
    return cst_only_module

  def _patch_trivial_const_op(self, module: mlir.ModuleOp):
    for op in module.walk():
      if "tfl.dequantize" in op.name:
        f32_tensor = op.results[0]
        if cast(mlir.RankedTensorType, f32_tensor.type).elty != "f32":
          continue
        with mu.OpBuildingContext(op, insert_after=True):
          x = tfl.add(f32_tensor, tfl.const(1e-30))
          f32_tensor.replace_by(x)
          x.owner.operands[0] = f32_tensor

  def _litert_run(self, module: mlir.ModuleOp):
    cst_module_content = mu.write_flatbuffer(module)
    interpreter = interpreter_lib.Interpreter(model_content=cst_module_content)
    named_outputs = interpreter.get_signature_runner(_CST_ONLY_SIGNATURE)()
    outputs = []
    for i in range(len(named_outputs)):
      outputs.append(named_outputs[f"output_{i}"])
    return outputs

  def _fold_module(self, module: mlir.ModuleOp):
    if not isinstance(module, mlir.ModuleOp):
      raise ValueError(f"Input is not a module. Got {type(module)}.")

    composite_decomps = self._extract_composite_decomps(module)
    composite_decomps_set = set(composite_decomps)

    cst_ops = []
    output_cst_values = []

    for fn in module.ops:
      if isinstance(fn, func.FuncOp) and fn not in composite_decomps_set:
        local_cst_ops, local_output_cst_values = self._extract_constants(fn)
        cst_ops.extend(local_cst_ops)
        output_cst_values.extend(local_output_cst_values)

    if not output_cst_values:
      log("No constants found to fold.")
      return

    log(
        f"Generating constant-only module with {len(output_cst_values)}"
        " constants"
    )
    cst_module = self._create_cst_only_module(
        cst_ops, output_cst_values, composite_decomps
    )

    log("Patching trivial constant ops for LiteRT compatibility")
    self._patch_trivial_const_op(cst_module)

    log("Running constant-only model via LiteRT")
    cst_np_arrs = self._litert_run(cst_module)

    assert len(cst_np_arrs) == len(output_cst_values)

    log(f"Replacing {len(output_cst_values)} constants in module")
    for val, arr in zip(output_cst_values, cst_np_arrs):
      ir_attr = ir.DenseResourceElementsAttr.get_from_buffer(
          memoryview(arr),
          f"runtime_fold_{id(arr)}",
          val.type.to_mlir(),
      )
      attr = mlir.MlirAttribute(ir_attr)
      with mu.OpBuildingContext(val.owner):
        new_cst = tfl.const(attr)
        val.replace_by(new_cst)

    log("Running post-folding optimize pass")
    mu.passes.tfl.OptimizePass()(module)

  def call(self, module: mlir.ModuleOp):
    self._fold_module(module)
    module.cleanup()

    # Clear imtermediate constant numpy arrays for conversion mem efficiency.
    gc.collect()


def runtime_fold(
    ir_context: ir.Context,
    module_op: mlir.ModuleOp,
    *,
    fold_dequantize: bool = False,
) -> mlir.ModuleOp:
  """Runtime fold constants via LiteRT."""
  with ir_context:
    mu_module = mu_transform.mlir_to_model_utils(module_op)
    RuntimeFoldPass(fold_dequantize=fold_dequantize)(mu_module)
    return mu_transform.model_utils_to_mlir(mu_module, ir_context)
