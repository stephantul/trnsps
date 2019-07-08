# -*- coding: utf-8 -*-
"""Setup file."""
from setuptools import setup
from setuptools import find_packages


setup(name='trnsps',
      version="1.0.0",
      description='generate transpositions, deletions and insertions.',
      author='Stéphan Tulkens',
      author_email='stephan.tulkens@uantwerpen.be',
      url='https://github.com/stephantul/trnsps',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy'],
      classifiers=[
          'Intended Audience :: Developers',
          'Programming Language :: Python :: 3'],
      keywords='machine learning',
      zip_safe=True,
      python_requires='>=3')
