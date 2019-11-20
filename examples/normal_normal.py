from pmpyro import pm_like
import pmpyro as pm
import pyro
import pyro.distributions as dist

import matplotlib.pyplot as plt
plt.style.use('ggplot')


def normal_normal(x):
  mu = pyro.sample('mu', dist.Normal(1., 1.))
  x_obs = pyro.sample('obs', dist.Normal(mu, scale_real), obs=x)
  return x_obs


if __name__ == '__main__':
  # make data with `Normal(4., 1.)`
  loc_real, scale_real = 4., 1.
  X = dist.Normal(loc_real, scale_real).rsample([100, ])
  with pm_like(normal_normal, X) as model:
    var_dist, elbo = pm.fit(epochs=100)
    trace = pm.sample_variational(var_dist, samples=1000)
    pm.plot_posterior(trace)
    plt.show()
