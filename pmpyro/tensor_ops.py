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
