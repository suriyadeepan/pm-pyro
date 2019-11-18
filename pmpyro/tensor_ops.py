import torch


def to_tensor(*args):
  tensors = []
  for arg in args:
    if not isinstance(arg, type(torch.tensor([9.6]))):
      arg = arg if arg is None else torch.tensor(arg).float()
    tensors.append(arg)
  return tensors


def numpy(t):
  return t.cpu().detach().numpy()


def var_dim(var):
  shape = var.fn.shape()
  if len(shape) == 0:
    return 1
  if len(shape) > 1:
    raise Exception(f'Cannot handle high-dimensional variable {var}')
  return shape[0]
