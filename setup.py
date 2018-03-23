from setuptools import setup,find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='probabilistic_regression_tools',
      version='0.1.0a0',
      description='Probabilistic Regression Tools.',
      long_description=readme(),
      keywords='machine learning energy timeseries probabilistic regression',
      author='Andre Gensler',
      packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
      install_requires=[
          'numpy',
          'sklearn',
          'scipy',
      ],
      test_suite='nose.collector',
      tests_require=['nose','pandas'],
      classifiers=[
      'Development Status :: 3 - Alpha',
      'Programming Language :: Python :: 3',
      'Intended Audience :: Developers',
      ]
      )

