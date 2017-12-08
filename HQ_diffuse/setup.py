from setuptools import setup, Extension
from Cython.Build import cythonize
from glob import glob
import os
import numpy
# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
#import distutils.sysconfig
#cfg_vars = distutils.sysconfig.get_config_vars()
#for key, value in cfg_vars.items():
#    if type(value) == str:
#        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

#libs=[path for path in os.environ['LD_LIBRARY_PATH'].split(':') if path]



#-------------Dissociation and Recombination Module------------------
files = [	'cython/HQ_diffuse.pyx',
			'src/utility.cxx' ]
modules = [
        Extension('HQ_diffuse',
        		 sources=files, 
        		 language="c++", 
        		 #library_dirs=libs,
                 include_dirs=[numpy.get_include()],
        		 extra_compile_args=["-stdlib=libc++", '-march=native', '-fPIC'],
        		 #extra_compile_args=["-std=c++11", '-march=native', '-fPIC'],
        		 # use -stdlib=libc++ on MaxOS otherwise -std=c++11
        		 libraries=["m", "gsl", "gslcblas"]),
]


setup(
        ext_modules=cythonize(modules)
)
