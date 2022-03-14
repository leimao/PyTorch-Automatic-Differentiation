# https://pytorch.org/functorch/stable/notebooks/jacobians_hessians.html
# Requires PyTorch >= 1.11

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.benchmark import Timer
import torch.autograd.forward_ad as fwAD
from functorch import vmap, vjp, jvp, jacrev, jacfwd


# Linear layer mapping from R^N to R^M
def predict(weight, bias, x):
    return F.linear(x, weight, bias)


B = 4
N = 16
M = 32
weight = torch.randn(M, N)
bias = torch.randn(M)
x = torch.randn(B, N)  # feature vector

# Reverse Mode

primal = x.clone().requires_grad_()
diagonal_matrix = torch.eye(M)
cotangents = torch.stack([cotangent.repeat(B, 1) for cotangent in diagonal_matrix])

# Method 1
# Use PyTorch autograd reverse mode + `for` loop.
rev_jacobian = []
# 1 forward pass.
output = predict(weight, bias, primal)
# M backward pass.
for cotangent in cotangents:
    # Compute vjp, where v = cotangent
    (jacobian_row, ) = torch.autograd.grad(outputs=(output, ),
                                           inputs=(primal, ),
                                           grad_outputs=(cotangent, ),
                                           retain_graph=True)
    rev_jacobian.append(jacobian_row)
# jacobian: [M, B, N]
jacobian = torch.stack(rev_jacobian)
# jacobian: [B, M, N]
jacobian = jacobian.transpose(1, 0)

# Run a sanity check for the Jacobian.
primal = x.clone().requires_grad_()
output = predict(weight, bias, primal)
# This will not work.
# output.backward()
# As PyTorch gradient compute always assume the function has scalar output.
external_grad = torch.ones_like(output)
# This is equivalent to
# output.sum().backward()
output.backward(gradient=external_grad)
grad = primal.grad
assert torch.allclose(jacobian.sum(dim=1), grad)

# Set the jacobian from method 1 as the reference.

# Method 2
# Using functorch vjp + vmap.
_, vjp_fn = vjp(partial(predict, weight, bias), primal)
# In PyTorch autograd backward, 
# in order to compute the grad, there is no need to compute to compute Jacobian.
assert torch.allclose(vjp_fn(external_grad)[0], grad)
# A vectorized implementation for computing Jacobian using vjp.
(rev_jacobian, ) = vmap(vjp_fn)(cotangents)
rev_jacobian = rev_jacobian.transpose(1, 0)
assert torch.allclose(rev_jacobian, jacobian)

# Method 3
# Use functorch jacrev + vmap.
# A vectorized implementation for computing Jacobian using vjp.
# https://pytorch.org/functorch/stable/generated/functorch.vmap.html#functorch.vmap
compute_batch_jacobian = vmap(jacrev(predict, argnums=(2, )), in_dims=(None, None, 0))
(rev_jacobian, ) = compute_batch_jacobian(weight, bias, primal)
assert torch.allclose(rev_jacobian, jacobian)

# Forward Mode

primal = x.clone().requires_grad_()
diagonal_matrix = torch.eye(N)
tangents = torch.stack([cotangent.repeat(B, 1) for cotangent in diagonal_matrix])

# Method 1
# Use PyTorch autograd forward mode + `for` loop.
fwd_jacobian = []
with fwAD.dual_level():
    # N forward pass
    for tangent in tangents:
        dual_input = fwAD.make_dual(primal, tangent)
        # Tensors that do not not have an associated tangent are automatically
        # considered to have a zero-filled tangent of the same shape.
        dual_output = predict(weight, bias, dual_input)
        # Unpacking the dual returns a namedtuple with ``primal`` and ``tangent``
        # as attributes
        jacobian_column = fwAD.unpack_dual(dual_output).tangent
        fwd_jacobian.append(jacobian_column)
fwd_jacobian = torch.stack(fwd_jacobian).permute(1, 2, 0)
torch.allclose(fwd_jacobian, jacobian)

# Method 2
# Using functorch vjp + `for` loop.
fwd_jacobian = []
# No functorch vmap for jvp
for tangent in tangents:
    _, jacobian_column = jvp(func=partial(predict, weight, bias), primals=(primal, ), tangents=(tangent, ))
    fwd_jacobian.append(jacobian_column)
fwd_jacobian = torch.stack(fwd_jacobian).permute(1, 2, 0)
torch.allclose(fwd_jacobian, jacobian)

# Method 3
# Use functorch jacfwd.
compute_batch_jacobian = vmap(jacfwd(predict, argnums=(2, )), in_dims=(None, None, 0))
(fwd_jacobian, ) = compute_batch_jacobian(weight, bias, primal)
assert torch.allclose(fwd_jacobian, jacobian)

# Measure Performance

cpu = torch.device("cpu")
cuda = torch.device("cuda:0")
for device in [cuda, cpu]:
    for B, N, M in [(4, 16, 10240), (4, 10240, 16)]:
        print(f"B: {B}, N: {N}, M: {M}, Device: {device}")
        weight = torch.randn(M, N).to(device)
        bias = torch.randn(M).to(device)
        x = torch.randn(B, N).to(device)

        using_fwd = Timer(
            stmt="vmap(jacfwd(predict, argnums=(2, )), in_dims=(None, None, 0))(weight, bias, x)",
            globals=globals())
        using_bwd = Timer(
            stmt="vmap(jacrev(predict, argnums=(2, )), in_dims=(None, None, 0))(weight, bias, x)",
            globals=globals())

        jacfwd_timing = using_fwd.timeit(100)
        jacrev_timing = using_bwd.timeit(100)

        print(f"Forward mode jacfwd time: {jacfwd_timing.mean * 1000:.5f} ms")
        print(f"Reverse mode jacrev time: {jacrev_timing.mean * 1000:.5f} ms")
