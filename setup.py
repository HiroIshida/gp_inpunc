from setuptools import setup, find_packages
setup(
    name='gp_uncinp',
    version='0.0.1',
    description='Sample package for Python-Guide.org',
    license=license,
    install_requires=['numpy', 'scipy'],
    packages=find_packages(exclude=('tests', 'docs'))
)
