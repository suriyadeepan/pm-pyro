"""pmpyro/inference.py

Collection of Inference methods.
Sampling-based methods like NUTS and HMC are available. Pyro's Stochastic Variational Inference is also available.
"""
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer.mcmc.util import predictive
from pyro.infer.mcmc import NUTS, MCMC, HMC
import pyro
import arviz as az

from pmpyro.model import PmlikeModel as Context
from pmpyro.tensor_ops import *
from pmpyro.utils import *

from tqdm import trange
import os


def sample(draws=500, model=None,
    warmup_steps=None, num_chains=1, kernel='nuts'):
  """Markov-chain Monte Carlo sampling.

  Sampling should be run within the context of a model or the model should be passed as an argument `model` explicitly.
  Number of samples is given by `draws` which defaults to `500`.
  Warm-up steps are assumed to be 30% of sample count.
  MCMC kernel can be selected by setting `kernel`. `hmc` and `nuts` are available.
  `pmpyro.inference.sample` returns a trace of samples

  """
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
  """Separate vector-valued variable traces into multiple appropriatedly named scalars.

  Returns the updated trace and a list of vector-valued variable names.
  """
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
  """Combine disentangled variables into their original vector-valued states

  `trace` is a python dictionary of disentangled variable traces.
  `variables` is a list of free variables from the model. 

  `pmpyro.inference.entangle_trace` returns an entangled trace.
  """
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
  """Generate Posterior Predictive samples given new data points
  
  Data are given a positional arguments in `args`.
  `trace` from sampling or variational inference is required.
  Number of samples to generate is given by `samples`.
  The function should be run within the context of the model or a model should be explicitly passed as an argument `model`.
  """
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
  """Summary of Random Variables from the `trace`
  
  `burn_in` period ignores the initial super-noisy samples.
  """
  num_samples = count_samples(trace)  # count the samples in trace
  if burn_in > 0:  # if there is burn-in period, filter out those samples
    trace = get_last_n_from_trace(trace, num_samples - burn_in)
  trace, _ = disentangle_trace(trace)  # get disentangle_trace for summary
  return az.summary(trace)  # use arviz's summary function


def fit(model=None, lr=1e-2, epochs=10000, autoguide=None):
  """Wrapper to pyro's SVI (Stochastic Variation Inference) inference engine.

  Requires to be run within a model's context or a model to be passed explicitly via `model`.
  The Variational Family of distributions (guide) can be set via `autoguide`. By default Diagonal Normal family is chosen.
  Control optimization by using the arguments `lr` and `epochs`
  """
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
  """Create multiple scalar variables names to replace vector-values variable names.

  `variables` is a list of free variables associated with the model.
  """
  dvars, vectors = [], {}
  for var in variables:
    dims = var_dim(var)  # get the shape of the variable
    if dims == 1:  # if it is a scalar
      dvars.append(var.name)
    else:  # if it is a vector
      vectors[var.name] = []
      for i in range(dims):
        dvars.append(f'{var.name}_{i}')  # add created scalar
        vectors[var.name].append(dvars[-1])  # track the vector-valued vars
  return dvars, vectors


def sample_variational(vardist, samples=100, model=None):
  """Generate samples from the variational distribution.
  
  `vardist` is a variational family of distributions.
  Requires to be run within a model's context or a model to be passed explicitly via `model`.
  Number of samples to generate is given by `samples`.
  """
  if model is None:
    model = Context.get_context()
  psamples = vardist.sample([samples, ])  # sample from variational dist
  psamples = psamples.transpose(1, 0)  # [ d, n ] -> [ n, d ]
  dfvars, vectors = disentangle_variables(model.free_variables())
  # disentangled trace
  dtrace = { var : psample for var, psample in zip(dfvars, psamples) }
  # entangle trace
  trace = entangle_trace(dtrace, model.free_variables())
  return trace
