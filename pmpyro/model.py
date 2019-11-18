"""pmpyro/model.py

Abstractions to manage models and associated random variables
"""
from pmpyro.tensor_ops import *
import threading
import pyro


class Context:
  """Functionality for objects that put themselves in a context using the `with` statement.
  """ 
  contexts = threading.local()  # thread-local storage

  def __enter__(self):
    cls = type(self)  # get a handle to Class `Context`
    contexts = cls.get_contexts()  # call @classmethod `get_contexts`
    contexts.append(self)  # add instance to contexts
    return self

  def __exit__(self, typ, value, traceback):
    cls = type(self)  # get a handle to Class `Context`
    contexts = cls.get_contexts()  # call @classmethod `get_contexts`
    contexts.pop()  # remove instance from contexts stack

  @classmethod
  def get_contexts(cls):
    if not hasattr(cls.contexts, 'stack'):  # does `Context.contexts.stack` exist?
      cls.contexts.stack = []  # create and return an empty stack
    return cls.contexts.stack

  @classmethod
  def get_context(cls):
    """Return the deepest context on the stack."""
    contexts = cls.get_contexts()  # get all contexts
    if len(contexts) == 0:
      raise Exception("Context stack is empty!")
    return contexts[-1]  # return the deepest context


class Variable:
  """Wrapper for Random Variables associated with the model"""

  def __init__(self, *args, **kwargs):
    for name in kwargs:
      setattr(self, name, kwargs.get(name))

  def sample(self, *args, **kwargs):
    """Calls `pmpyro.model.Variable.fn.sample`"""
    return self.fn.sample(*args, **kwargs)

  def __repr__(self):
    return self.name


class PmlikeModel(Context):
  """Encapsulates variables of a bayesian model. Thin wrapper over pyro's stochastic functions.
  """

  def __init__(self, *args, stfn=None):
    """ Observations/data are passed as positional arguments (`args`).
    Pyro's stochastic function is passed as a keyword argument `stfn`.
    """
    self.stfn = stfn
    self.args = args
    self.named_variables = {}  # maintain named variables
    self.variables = []  # maintain a list of variables

  def add_variable(self, var):
    """Add variable `var` to the model"""
    if var.name in self.named_variables:
      raise Exception('Model contains {} already!'.format(var.name))
    self.named_variables[var.name] = var
    self.variables.append(var)
    setattr(self, var.name, var)

  def observed_variables(self):
    """Get a list of observed variables"""
    return [ var for var in self.variables if var.is_observed ]

  def free_variables(self):
    """Get a list of free/unobserved variables"""
    return [ var for var in self.variables if not var.is_observed ]


def get_trace(stfn, *data):
  """Get a trace of sample sites in a stochastic function
  
  Data are passed as positional arguments `data`.
  """
  # get trace of sample sites using a trace messenger
  trace = pyro.poutine.trace(stfn).get_trace(*data)
  variables = []
  for name in trace.nodes:  # iterate through the trace
    if name is not '_INPUT' and name is not '_RETURN':
      variables.append(trace.nodes[name])
  return variables


def pm_like(stfn, *args):
  """Create a PyMC3-esque wrapper over a stochastic function.

  Data are passed as positional arguments `args`
  """
  data = to_tensor(*args)
  trace = get_trace(stfn, *data)
  # create a model wrapper
  pm_model = PmlikeModel(*data, stfn=stfn)
  for msg in trace:  # iterate through messages
    var = Variable(**msg)  # make variables from messages
    pm_model.add_variable(var)   # add variable to model
  return pm_model
