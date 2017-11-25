# cython: c_string_type=str, c_string_encoding=ascii
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.math cimport *
import cython, os, h5py
import numpy as np
cimport numpy as np

#### ----------- Import C++ fucntions and class for Xsection and rates------------------
cdef extern from "../src/TransLorentzRotate.h":
	# lorentz_transform
	cdef vector[double] lorentz_transform(vector[double] momentum_4, vector[double] velocity_3)
	cdef vector[double] rotation_transform(vector[double] vector_3, double theta, double phi)
	cdef vector[double] angle_find(vector[double] vector_3)



#### ----------- Provide a direct python wrapper for these functions for tests

## ----------------- Lorentz transformation ---------------- ##
def lorentz(np.ndarray[double, ndim=1] p4, 
			np.ndarray[double, ndim=1] v3):
	return np.array(lorentz_transform(p4, v3))
	
## --------------- end of Lorentz transformation ------------ ##


## --------------- Rotation ---------------- ##
def rotation(np.ndarray[double, ndim=1] v3, double theta, double phi):
	return np.array(rotation_transform(v3, theta, phi))

## ------------ end of rotation ------------ ##


## --------------- find theta and phi of a 3-vector ------------------ ##
def angle(np.ndarray[double, ndim=1] v3):
	return np.array(angle_find(v3))

## ----------------- end of finding theta and phi -------------------- ##