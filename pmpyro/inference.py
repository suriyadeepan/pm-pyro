from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.infer.mcmc.util import predictive
from pyro.infer.mcmc import NUTS, MCMC, HMC
import pyro
import arviz as az

from pmpyro.model import PmlikeModel as Context
from pmpyro.tensor_ops import *
from pmpyro.utils import *

from tqdm import trange
import os


def sample(draws=500, model=None, warmup_steps=None, num_chains=1, kernel='nuts'):
  # get model from context
  if model is None:
    model = Context.get_context()
  stfn = model.stfn  # get stochastic function from model
  data = model.args  # get data
  # make nuts kernel
  kernels = {
      'nuts' : NUTS(stfn, adapt_step_size=True), 
      'hmc'  : HMC(stfn)
      }
  # if not num_chains:    # figure out number of chains
  #   num_chains = max(os.cpu_count() -1, 2)
  if not warmup_steps:  # figure out warm-up steps
    warmup_steps = int(0.3 * draws)
  # run MCMC
  mcmc = MCMC(kernels[kernel], num_samples=draws,
      warmup_steps=warmup_steps, num_chains=num_chains)
  mcmc.run(*data)
  # get num samples
  num_samples = num_chains * draws
  return mcmc.get_samples()


def disentangle_trace(trace):
  updated_trace = {}
  num_samples = count_samples(trace)
  vectors = []
  for var in trace:
    var_trace = trace[var]
    if var_trace.dim() > 1:  # vector
      if var_trace.dim() > 2:
        raise Exception('{var} : trace plot cannot handle dim > 2!')
      if var_trace.size(0) != num_samples:
        raise Exception(f'{var} : number of samples mismatch!')
      # disentangle vector variable traces
      vectors.append(var)
      var_trace = var_trace.transpose(1, 0)
      for i in range(var_trace.size(0)):
        scalar_name = f'{var}_{i}'
        updated_trace[scalar_name] = numpy(var_trace[i])
    else:  # scalar
      updated_trace[var] = numpy(trace[var])
  return updated_trace, vectors


def entangle_trace(trace, variables):
  dvars, vectors = disentangle_variables(variables)
  vec_vars = []
  for vector in vectors:
    vec_vars.extend(vectors[vector])
  # select the scalars scalars
  etrace = { var : trace[var] for var in dvars if var not in vec_vars }
  for var in vectors:
    vvars = vectors[var]
    evar = torch.stack([ trace[vvar] for vvar in vvars ]).transpose(1, 0)
    etrace[var] = evar
  return etrace


def sample_posterior_predictive(*args, trace=None, samples=100, model=None):
  if trace is None:  # check for a valid trace
    raise Exception(f'Need a valid trace; got trace={trace} instead!')
  if model is None:  # get model from context if necessary
    model = Context.get_context()
  num_pred_samples = min(samples, count_samples(trace))
  pred_samples = get_last_n_from_trace(trace, num_pred_samples)
  stfn = model.stfn
  if len(args) == 0:   # if no explicit samples are given:
    args = model.args[:-1] + (None,)  #  consider the last argument to model
  args = to_tensor(*args)  # convert to torch tensors
  ppc = predictive(stfn, pred_samples, *args)
  return ppc


def summary(trace, burn_in=0):
  num_samples = count_samples(trace)
  if burn_in > 0:
    trace = get_last_n_from_trace(trace, num_samples - burn_in)
  trace, _ = disentangle_trace(trace)
  return az.summary(trace)


def fit(model=None, lr=1e-2, epochs=10000, autoguide=None):
  if model is None:
    model = Context.get_context()
  stfn = model.stfn  # get stochastic function from model
  data = model.args  # get data
  # clear param store
  pyro.clear_param_store()
  # select auto-guide
  if autoguide is None:
    autoguide = AutoDiagonalNormal
  # create guide
  guide = autoguide(stfn) 
  # create elbo loss fn
  elbo_fn = pyro.infer.JitTraceGraph_ELBO()
  # create Adam optimizer
  optimiser = pyro.optim.Adam({"lr": lr})
  # create an SVI object
  svi = pyro.infer.SVI(stfn, guide, optimiser, elbo_fn)

  elbo = []
  with trange(epochs) as tbar:
    tbar.set_description('svi')
    for step in tbar:
      # run a step of svi
      step_loss = svi.step(*data)
      tbar.set_postfix(loss=step_loss)
      elbo.append(step_loss)

  # get variational distribution
  vardist = guide.get_posterior()
  return vardist, elbo


def disentangle_variables(variables):
  dvars, vectors = [], {}
  for var in variables:
    dims = var_dim(var)
    if dims == 1:
      dvars.append(var.name)
    else:
      vectors[var.name] = []
      for i in range(dims):
        dvars.append(f'{var.name}_{i}')
        vectors[var.name].append(dvars[-1])
  return dvars, vectors


def sample_variational(vardist, samples=100, model=None):
  if model is None:
    model = Context.get_context()
  psamples = vardist.sample([samples, ])
  psamples = psamples.transpose(1, 0)
  dfvars, vectors = disentangle_variables(model.free_variables())
  # disentangled trace
  dtrace = { var : psample for var, psample in zip(dfvars, psamples) }
  # entangle trace
  trace = entangle_trace(dtrace, model.free_variables())
  return trace
