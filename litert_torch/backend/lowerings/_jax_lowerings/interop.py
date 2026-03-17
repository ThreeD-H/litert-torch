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
"""Interop functions for converting between torch and jax.

Branched from https://github.com/google/torchax.
"""

import functools
from typing import Optional, ParamSpec

import jax
import jax.numpy as jnp
import numpy as np
import torch

P = ParamSpec('P')


TORCH_DTYPE_TO_JAX = {
    # NO_MAPPING        : jnp.float0.dtype (signless scalar int),
    torch.bool: jnp.bool_.dtype,
    # NO_MAPPING        : jnp.int4.dtype,
    torch.int8: jnp.int8.dtype,
    torch.int16: jnp.int16.dtype,
    torch.int32: jnp.int32.dtype,
    torch.int64: jnp.int64.dtype,
    torch.long: jnp.int64.dtype,
    # NO_MAPPING        : jnp.uint4
    torch.uint8: jnp.uint8.dtype,
    torch.uint16: jnp.uint16.dtype,
    torch.uint32: jnp.uint32.dtype,
    torch.uint64: jnp.uint64.dtype,
    # NO_MAPPING        : jnp.float8_e4m3b11fnuz.dtype,
    torch.float8_e4m3fn: jnp.float8_e4m3fn.dtype,
    # NO_MAPPING        : jnp.float8_e4m3fnuz.dtype,
    torch.float8_e5m2: jnp.float8_e5m2.dtype,
    # NO_MAPPING        : jnp.float8_e5m2fnuz.dtype,
    torch.bfloat16: jnp.bfloat16.dtype,
    torch.half: jnp.float16.dtype,
    torch.float16: jnp.float16.dtype,
    torch.float32: jnp.float32.dtype,
    torch.float64: jnp.float64.dtype,
    torch.double: jnp.double.dtype,
    torch.complex64: jnp.complex64.dtype,
    torch.complex128: jnp.complex128.dtype,
    None: None,
}


def t2j_dtype(dtype):
  if dtype not in TORCH_DTYPE_TO_JAX:
    raise RuntimeError(
        f'Attempting to convert unknown type: {dtype} to jax type,'
    )
  return TORCH_DTYPE_TO_JAX[dtype]


def convert_dtype(use_default_dtype: bool = True):
  """Converts `dtype` kwarg of function from torch to JAX.

  Args:
    use_default_dtype: Whether to use torch default dtype if none is provided.

  Returns:
    A decorator that wraps a JAX implementation of a torch function.
  """

  def decorator(func):

    @functools.wraps(func)
    def wrapper(
        *args: P.args, dtype: Optional[torch.dtype] = None, **kwargs: P.kwargs
    ):
      if not dtype and use_default_dtype:
        dtype = torch.get_default_dtype()
      if isinstance(dtype, torch.dtype):
        jax_dtype = t2j_dtype(dtype)
      else:
        jax_dtype = dtype

      return func(*args, dtype=jax_dtype, **kwargs)

    return wrapper

  return decorator


def maybe_convert_constant_dtype(val, dtype):
  """Optionally converts scalar constant's dtype using numpy.

  Use in cases where you require a constant and can't handle a traced array.
  """
  if val and dtype:
    if isinstance(val, jax.Array):
      return maybe_convert_constant_dtype(val.item(), dtype)

    return np.array(val, dtype)

  return val


def promote_int_input(f):
  """If the first argument is an int array, promote it to float32."""

  @functools.wraps(f)
  def wrapper(x: jax.Array, *args: P.args, **kwargs: P.kwargs):
    if x.dtype in [jnp.int8, jnp.int16, jnp.int32, jnp.int64]:
      x = x.astype(t2j_dtype(torch.get_default_dtype()))

    return f(x, *args, **kwargs)

  return wrapper
