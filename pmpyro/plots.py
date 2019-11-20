"""pmpyro/plots.py

Tools for managing and plotting traces.
"""
from pmpyro.inference import sample_posterior_predictive
from pmpyro.inference import disentangle_trace
from pmpyro.model import PmlikeModel as Context
from pmpyro.tensor_ops import *

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib as mpl
import pymc3 as pm
import torch


"""Configuration for matplotlib"""
# Plot style
plt.style.use('ggplot')
# Color Palette
PM_BLUE = '#0e688a'  # PyMC3 colors
PM_RED = '#e26553'
PM_GRAY = '#504a4e'
PYRO_YELLOW = '#fecd08'  # Pyro colors
PYRO_ORANGE = '#f26722'
PYRO_RED = '#ed1f24'
# Font settings
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 12}
mpl.rc('font', **font)
mpl.rcParams['text.color'] = PM_GRAY
mpl.rcParams['axes.labelcolor'] = PM_GRAY
mpl.rcParams['xtick.color'] = PM_GRAY
mpl.rcParams['ytick.color'] = PM_GRAY
# set default figure size
mpl.rcParams['figure.figsize'] = (10.0, 8.0)


def resolve_var_names(var_names, trace, vectors):
  """Break down vector valued variables `vectors` in `trace` into multiple scalars.

  Eg : `{ 'beta' : [ [0.7, 2.3], [ ... ], .. ] }` resolves to `{ 'beta_0' : [ 0.7, ... ], 'beta_1' : [ 2.3, ... ] }

  """
  if var_names is None:
    return None
  for name in var_names:
    if name not in trace:
      if name in vectors:
        var_names.remove(name)  # remove from var_names
        var_names.extend(  # add scalar vars from trace
            [ var for var in trace if name in var ])
      else:
        raise Exception(f'"{name}" not in trace!')
  return var_names


def traceplot(trace, *args, **kwargs):
  """Resolves variables in trace and plot trace"""
  # convert vector-valued variables into multiple scalars
  trace, vectors = disentangle_trace(trace)
  # resolve variable names
  var_names = resolve_var_names(kwargs.get('var_names'), trace, vectors)
  if var_names is not None:
    kwargs['var_names'] = var_names
  pm.traceplot(trace, *args, **kwargs)  # `pymc3.traceplot`


def plot_posterior(trace, *args, **kwargs):
  """Plot posterior of random variables.

  Default behaviour is to plot all random variables.
  Variables can be selectively plotted by mentioning them in `var_names` argument.
  """
  # convert vector-valued variables into multiple scalars
  trace, vectors = disentangle_trace(trace)
  # resolve variable names
  var_names = resolve_var_names(kwargs.get('var_names'), trace, vectors)
  if var_names is not None:
    kwargs['var_names'] = var_names
  pm.plot_posterior(trace, *args, **kwargs)  # `pymc3.traceplot`


def make_ppc_legend():
  """Helper to make ppc plot legend"""
  ppc = mpatches.Patch(color=PYRO_ORANGE, alpha=0.65,
      label='Posterior Predictive')
  obs = mpatches.Patch(color=PM_BLUE, alpha=0.65,
      label='Observations')
  plt.legend(handles=[ppc, obs])


def plot_posterior_predictive(*args, **kwargs):
  """Plot Posterior Predictive samples.

  Set transparency using `alpha`.
  Optionally observations can be plotted using `obs` dictionary.
  Data are passed as positional arguments which will be used in `pmpyro.inference.sample_posterior_predictive`.
  Additional keyword arguments to `pmpyro.inference.sample_posterior_predictive` are required.
  """
  # get alpha and obs from kwargs
  alpha = kwargs.get('alpha', 0.01)
  obs = kwargs.get('obs', {})
  # remove alpha and obs from kwargs
  #  before passing to `sample_posterior_predictive`
  kwargs = { k : kwargs[k] for k in kwargs if k not in ['obs', 'alpha'] }
  if kwargs.get('model') is None:  # get model from context if necessary
    model = Context.get_context()
  if len(args) == 0:   # if no explicit samples are given:
    args = model.args[:-1] + (None,)  #  consider the last argument to model
  args = to_tensor(*args)  # convert to torch tensors
  trace = kwargs.get('trace')  # get trace
  ppc = sample_posterior_predictive(*args, **kwargs)
  for pname in ppc:
    pvar = ppc[pname]
    for i, var in enumerate(list(args)):
      if var is not None:  # filter-out `None`
        #if var.dim() != 1 or pvar.dim() != 2:
        #  raise Exception('Only scalars allowed!')
        if var.dim() > 1:
          vvars = expand_feature(var, pvar.size(1))
        else:
          vvars = [var]
        plt.ylabel(pname, fontsize=12)
        plt.xlabel(f'x_{i}', fontsize=12)
        for vvar in vvars:
          for pvar_t in pvar:
            plt.scatter(vvar, pvar_t, c=PYRO_ORANGE, alpha=alpha)
          if pname in obs:
            plt.scatter(vvar, obs[pname], c=PM_BLUE, alpha=min(0.65, 30.*alpha))
          make_ppc_legend()
          plt.figure()
  # return posterior predictive samples
  return ppc
