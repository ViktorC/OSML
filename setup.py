from setuptools import setup

install_requires = ['numpy', 'pandas']

setup_requires = ['pytest-runner']

test_requires = ['pytest', 'coverage', 'scipy', 'sklearn']

setup(
    name='osml',
    version='1.0',
    description='A library of old school machine learning algorithms',
    author='Viktor Csomor',
    author_email='viktor.csomor@gmail.com',
    packages=['osml'],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=install_requires + test_requires
)
