from setuptools import setup

install_requires = [
    'Flask',
    'Flask_Mail',
    'numpy',
    'pandas',
    'scikit_learn',
    'scikit_plot'
]

setup(
    name = 'pymlapigen',
    version='1',
    description='Aplicación TFG Adrián Ruiz Parra',
    author='Adrián Ruiz',
    author_email='adruizp@alumnos.unex.es',
    packages=['pymlapigen'],
    include_package_data = True,
    install_requires = install_requires,
    entry_points={"console_scripts": ["pymlapigen=pymlapigen.cli:cli"]}
)