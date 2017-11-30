# cython: c_string_type=str, c_string_encoding=ascii
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport *
import cython, os, h5py
import numpy as np
cimport numpy as np

#------------------Import C++ fucntions and class for Xsection and rates------------------
cdef extern from "../src/DisRec.h":
	# function relates to integrated decay rate
	cdef double R_1S_decay(double vabs, double T)

	# function relates to decay sampling
	cdef double dRdq_1S(double q, void * params)
	cdef double dRdq_1S_small_v(double q, void * params_)
	cdef double qdRdq_1S_u(double u, void * params)
	cdef double find_max(double(*f)(double x, void * params), double * params, double xL_, double xR_)
	cdef double decay_sample_1S_dRdq(double v, double T, double maximum)
	cdef double decay_sample_1S_costheta_q(double q, double v, double T)
	cdef double decay_sample_1S_final_p(double q)
	
	# function relates to recombine rate
	cdef double RtimesV_1S_reco(double v, double T, double p)
	cdef double dist_position(double r)
	
	# function relates to recombine sampling
	cdef double reco_sample_1S_q(double p)
	cdef double reco_sample_1S_costheta(double v, double T, double q)

	# small v cut
	cdef double small_number


	
cdef extern from "../src/utility.h":
	cdef double E1S


# Provide a direct python wrapper for these functions for tests
def pyE1S():
	return E1S

def pyR_1S_decay(double vabs, double T):
	return R_1S_decay(vabs, T)

def pydRdq_1S(double q, double v, double T):
	cdef double * params = <double*>malloc(2*sizeof(double))
	v = np.max([small_number, v])
	cdef double gamma = 1./sqrt(1. - v*v)
	params[0] = gamma*(1.+v)/T
	params[1] = gamma*(1.-v)/T
	return dRdq_1S(q, params)

def pyR_1S_reco(double vabs, double T, double p_rel):
	return RtimesV_1S_reco(vabs, T, p_rel)


pyR_1S_decay = np.vectorize(pyR_1S_decay)
pydRdq_1S = np.vectorize(pydRdq_1S)
pyR_1S_reco = np.vectorize(pyR_1S_reco)

@cython.boundscheck(False)
@cython.wraparound(False)



## --------------------- 2-d interpolation ----------------- ##
cdef double interp2d(np.ndarray[np.double_t, ndim=2] grid,
					 double x, double y, 
					 double xmin, double xmax, double dx, Nx,
					 double ymin, double ymax, double dy, Ny):
	x = np.min([np.max([x, xmin]), xmax])
	y = np.min([np.max([y, ymin]), ymax])
	cdef double rx, ry, res=0.
	cdef size_t ix, iy, i, j
	rx, ix = np.modf((x - xmin)/dx) 
	ry, iy = np.modf((y - ymin)/dy)
	cdef double wx[2], wy[2]
	wx[0] = 1.-rx; wx[1] = rx
	wy[0] = 1.-ry; wy[1] = ry
	for i in range(2):
		for j in range(2):
			res += grid[(ix+i)%Nx, (iy+j)%Ny]*wx[i]*wy[j]
	return res


## --------------------- 3-d interpolation ----------------- ##
cdef double interp3d(np.ndarray[np.double_t, ndim=3] grid,
					 double x, double y, double z,
					 double xmin, double xmax, double dx, Nx,
					 double ymin, double ymax, double dy, Ny,
					 double zmin, double zmax, double dz, Nz):
	x = np.min([np.max([x, xmin]), xmax])
	y = np.min([np.max([y, ymin]), ymax])
	z = np.min([np.max([z, zmin]), zmax])
	cdef double rx, ry, rz, res=0.
	cdef size_t ix, iy, iz, i, j, k
	rx, ix = np.modf((x - xmin)/dx) 
	ry, iy = np.modf((y - ymin)/dy)
	rz, iz = np.modf((z - zmin)/dz)
	cdef double wx[2], wy[2], wz[2]
	wx[0] = 1.-rx; wx[1] = rx
	wy[0] = 1.-ry; wy[1] = ry
	wz[0] = 1.-rz; wz[1] = rz
	for i in range(2):
		for j in range(2):
			for k in range(2):
				res += grid[(ix+i)%Nx, (iy+j)%Ny, (iz+k)%Nz] *wx[i]*wy[j]*wz[k]
	return res



cdef class DisRec(object):
	cdef double vmin, vmax, dv
	cdef double Tmin, Tmax, dT
	cdef p_rel_log_min, p_rel_log_max, dp_rel
	cdef size_t N_v, N_T, N_p_rel
	cdef np.ndarray R_1S_dis
	cdef np.ndarray qdRdq_1S_max
	cdef np.ndarray RV_1S_reco
	
	def __cinit__(self, table_folder='tables', overwrite=False):
		if not os.path.exists(table_folder):
			os.mkdir(table_folder)
		fname = table_folder+'/Diquark-table.hdf5'
		if not os.path.exists(fname):
			f = h5py.File(fname, 'w')
		else:
			f = h5py.File(fname, 'a')
		
		gpname = 'c-system'
		cdef double params[2]
		if (gpname in f) and (not overwrite):
			print ("loading T(1S)+g <-> c+c rate table")
			gp = f[gpname]
			self.vmin, self.vmax, self.N_v = gp.attrs['v-min-max-N']
			self.Tmin, self.Tmax, self.N_T = gp.attrs['T-min-max-N']
			self.p_rel_log_min, self.p_rel_log_max, self.N_p_rel = gp.attrs['p_rel_log-min-max-N']
			self.R_1S_dis = gp['R_1S_dis'].value
			self.qdRdq_1S_max = gp['qdRdq_1S_max'].value
			self.RV_1S_reco = gp['RV_1S_reco'].value
		else:
			if gpname in f:
				del f[gpname]
			gp = f.create_group(gpname)
			self.vmin = 0.01; self.vmax = 0.999; self.N_v = 100
			self.Tmin = 0.15; self.Tmax = 0.5; self.N_T = 30
			# here we use p_rel in GeV; ln(1000)=6.9
			self.p_rel_log_min = -4.0; self.p_rel_log_max = 1.0; self.N_p_rel = 100

			
			## Initialize dissociation rate table
			print ("generating T(1S)+g -> c+c rate table")
			varray = np.linspace(self.vmin, self.vmax, self.N_v)
			Tarray = np.linspace(self.Tmin, self.Tmax, self.N_T)
			grid_v, grid_T = np.meshgrid(varray, Tarray)
			self.R_1S_dis = np.transpose(pyR_1S_decay(grid_v, grid_T))
			self.qdRdq_1S_max = np.zeros_like(self.R_1S_dis)
			for iv, v in enumerate(varray):
				params[0] = v
				for iT, T in enumerate(Tarray):
					params[1] = T
					self.qdRdq_1S_max[iv, iT] = find_max(&qdRdq_1S_u, params, 0., 4.)


			## Initialize recombination rate*vol table
			print ("generating c+c -> T(1S)+g rate*vol table, vol in fm^3")
			p_relarray = np.exp( np.linspace(self.p_rel_log_min, self.p_rel_log_max, self.N_p_rel) )
			grd_v, grd_T, grd_p_rel = np.meshgrid(varray, Tarray, p_relarray)
			self.RV_1S_reco = pyR_1S_reco(grd_v, grd_T, grd_p_rel).transpose(1,0,2)
			
			
			## store the disso and reco rates in datasets
			gp.create_dataset('R_1S_dis', data=self.R_1S_dis)
			gp.create_dataset('qdRdq_1S_max', data=self.qdRdq_1S_max)
			gp.create_dataset('RV_1S_reco', data = self.RV_1S_reco)
			gp.attrs.create('v-min-max-N', 
							np.array([self.vmin, self.vmax, self.N_v]))
			gp.attrs.create('T-min-max-N', 
							np.array([self.Tmin, self.Tmax, self.N_T]))
			gp.attrs.create('p_rel_log-min-max-N', 
							np.array([self.p_rel_log_min, self.p_rel_log_max, self.N_p_rel]))
		
		self.dv = (self.vmax-self.vmin)/(self.N_v - 1.)
		self.dT = (self.Tmax-self.Tmin)/(self.N_T - 1.)
		self.dp_rel = (self.p_rel_log_max - self.p_rel_log_min)/(self.N_p_rel - 1.)

		print ("done")
	
	
	
	cpdef get_R_1S_dis(self, double v, double T):
		return interp2d(self.R_1S_dis, v, T, 
						self.vmin, self.vmax, self.dv, self.N_v,
						self.Tmin, self.Tmax, self.dT, self.N_T)
	
	
	cpdef get_qdRdq_1S_max(self, double v, double T):
		return interp2d(self.qdRdq_1S_max, v, T, 
						self.vmin, self.vmax, self.dv, self.N_v,
						self.Tmin, self.Tmax, self.dT, self.N_T)


	cpdef pydecay_sample_1S_init(self, double v, double T):
		cdef double q = decay_sample_1S_dRdq(v, T, self.get_qdRdq_1S_max(v, T))
		cdef double costheta_q = decay_sample_1S_costheta_q(q, v, T)
		return q, costheta_q, (2.*M_PI*rand())/RAND_MAX
	
	
	cpdef pydecay_sample_1S_final(self, double q):
		cdef double p_rel = decay_sample_1S_final_p(q)
		return p_rel, 2.*(rand()/RAND_MAX-0.5), (2.*M_PI*rand())/RAND_MAX
	
	
	cpdef get_R_1S_reco(self, double v, double T, double p_rel, double r):
		small_number = 0.000001		# add this small number to p_rel in case p_rel = 0
		cdef double Rvol = interp3d(self.RV_1S_reco, v, T, np.log(p_rel+small_number),
									self.vmin, self.vmax, self.dv, self.N_v,
									self.Tmin, self.Tmax, self.dT, self.N_T,
									self.p_rel_log_min, self.p_rel_log_max, self.dp_rel, self.N_p_rel)
		return Rvol * dist_position(r)
		# no factor of 2 in dist_position, add that when judging the theta function
		

	cpdef pyreco_sample_1S_final(self, double v, double T, double p_rel):
		cdef double q = reco_sample_1S_q(p_rel)
		cdef double costheta_q = reco_sample_1S_costheta(v, T, q)
		return q, -costheta_q, (2.*M_PI*rand())/RAND_MAX
		# here we return the momentum of the formed Upsilon(1S)


