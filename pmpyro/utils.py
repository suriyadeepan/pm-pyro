"""pmpyro/utils.py

General Utilities
"""

def mode(l):
  """Find the most frequent item in list"""
  return max(set(l), key=l.count)


def count_samples(trace):
  """Count the number of samples in a trace"""
  return mode([ trace[z].size(0) for z in trace ])


def get_last_n_from_trace(trace, n):
  """Get last `n` samples from trace"""
  return { var : trace[var][-n:] for var in trace }
