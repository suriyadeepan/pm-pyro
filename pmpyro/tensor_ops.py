"""pmpyro/tensor_ops.py

Tensor Operation Utilities
"""
import torch


def to_tensor(*args):
  """Convert arrays into torch tensors"""
  tensors = []
  for arg in args:
    if not isinstance(arg, type(torch.tensor([9.6]))):
      arg = arg if arg is None else torch.tensor(arg).float()
    tensors.append(arg)
  return tensors


def numpy(t):
  """Convert torch tensor into numpy array"""
  return t.cpu().detach().numpy()


def var_dim(var):
  """Find the shape associated with random variable"""
  shape = var.fn.shape()
  if len(shape) == 0:
    return 1
  if len(shape) > 1:
    raise Exception(f'Cannot handle high-dimensional variable {var}')
  return shape[0]
