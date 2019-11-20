from setuptools import setup, find_packages

setup(
    long_description_content_type='text/markdown',
    name='pm-pyro',
    author='Suriyadeepan Ramamoorthy',
    version='0.3.3',
    packages=['pmpyro'],
    python_requires='>=3.5',
    url='https://github.com/suriyadeepan/pm-pyro',
    license='GNU Affero General Public License v3 or later (AGPLv3+)',
    long_description=open('README.md').read(),
    description='A PyMC3-like Interface for Pyro Stochastic Functions',
    install_requires=['arviz', 'pymc3', 'pyro-ppl']
)
