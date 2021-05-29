from distutils.core import setup, Extension
import numpy

CAnnoPred = Extension('CAnnoPred',sources = ['c/CAnnoPred.c'])

setup (name = 'CAnnoPred',version = '1.0',description = 'AnnoPred algorithm',ext_modules = [CAnnoPred],include_dirs=[numpy.get_include()])