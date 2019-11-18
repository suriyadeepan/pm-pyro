import pmpyro as pm
import pyro.distributions as dist
import pyro
import torch

import pytest


@pytest.fixture
def normal_data():
  return 3. * torch.randn([1000, ]) + 4.


@pytest.fixture
def normal_normal():
  def model(data):
    mu = pyro.sample('mu', dist.Normal(0., 1.))
    sigma = pyro.sample('sigma', dist.HalfCauchy(5.))
    with pyro.plate('observe_data'):
      pyro.sample('obs', dist.Normal(mu, sigma), obs=data)
  return model


def test_model(normal_normal):
  from pmpyro.model import PmlikeModel
  model = PmlikeModel(stfn=normal_normal)


def test_pm_like(normal_normal, normal_data):
  with pm.pm_like(normal_normal, normal_data) as model:
    assert len(model.variables) == 3


@pytest.fixture
def pm_model(normal_normal, normal_data):
  return pm.pm_like(normal_normal, normal_data)


def test_sample(pm_model):
  with pm_model:
    trace = pm.sample(100)
    assert trace is not None
    assert len(trace) == len(pm_model.variables) - 1 


@pytest.fixture
def pm_trace(pm_model):
  with pm_model:
    trace = pm.sample(100)
    return trace


def test_traceplot(pm_trace):
  pm.traceplot(pm_trace)


def test_plot_posterior(pm_trace):
  pm.plot_posterior(pm_trace)


def test_sample_posterior_predictive(normal_data, pm_trace, pm_model):
  with pm_model:
    num_samples = 10
    ppc = pm.sample_posterior_predictive(normal_data,
        trace=pm_trace, samples=num_samples)
    obs_vars = pm_model.observed_variables()
    assert len(ppc) == len(obs_vars)
    assert len(ppc[obs_vars[0].name]) == num_samples


def test_summary(pm_trace, pm_model):
  assert len(pm.summary(pm_trace)) == len(pm_model.free_variables())


def test_variational(pm_model):
  with pm_model:
    vardist, elbo = pm.fit(epochs=10)
    trace = pm.sample_variational(vardist, 10)
    fvars = pm_model.free_variables()
    assert len(trace) == len(fvars)
    assert len(trace[fvars[0].name]) == 10
