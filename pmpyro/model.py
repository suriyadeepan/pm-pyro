from pmpyro.tensor_ops import *
import threading
import pyro


class Context:
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
    contexts = cls.get_contexts()  # get all contexts
    if len(contexts) == 0:
      raise Exception("Context stack is empty!")
    return contexts[-1]  # return the deepest context


class Variable:

  def __init__(self, *args, **kwargs):
    for name in kwargs:
      setattr(self, name, kwargs.get(name))

  def sample(self, *args, **kwargs):
    return self.fn.sample(*args, **kwargs)

  def __repr__(self):
    return self.name


class PmlikeModel(Context):

  def __init__(self, *args, stfn=None, variables=[]):
    self.stfn = stfn
    self.args = args
    self.variables = variables
    self.named_variables = {}
    self.variables = []

  def add_variable(self, var):
    if var.name in self.named_variables:
      raise Exception('Model contains {} already!'.format(var.name))
    self.named_variables[var.name] = var
    self.variables.append(var)
    setattr(self, var.name, var)

  def observed_variables(self):
    return [ var for var in self.variables if var.is_observed ]

  def free_variables(self):
    return [ var for var in self.variables if not var.is_observed ]


def get_trace(stfn, *data):
  # get trace of sample sites
  trace = pyro.poutine.trace(stfn).get_trace(*data)
  variables = []
  for name in trace.nodes:
    if name is not '_INPUT' and name is not '_RETURN':
      variables.append(trace.nodes[name])
  return variables


def pm_like(stfn, *args):
  data = to_tensor(*args)
  trace = get_trace(stfn, *data)
  inf_model = PmlikeModel(*data, stfn=stfn)
  for msg in trace:
    var = Variable(**msg)
    inf_model.add_variable(var) 
  return inf_model
