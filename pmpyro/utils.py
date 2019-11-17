def mode(l):
  return max(set(l), key=l.count)


def count_samples(trace):
  return mode([ trace[z].size(0) for z in trace ])


def get_last_n_from_trace(trace, n):
  return { var : trace[var][-n:] for var in trace }
