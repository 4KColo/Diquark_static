# I have removed the first line because it leads to errors in the encoding
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport *
import numpy as np
cimport numpy as np
import HqEvo

C1 = 0.197327			# 0.197 GeV*fm = 1


cdef extern from "../src/utility.h":
	cdef double product4(vector[double] A, vector[double] B)
	cdef void rotate_back_from_D(vector[double] Ap, vector[double] A, double Dx, double Dy, double Dz)
	cdef void boost4_By3(vector[double] Ap, vector[double] A, vector[double] v)
	cdef void boost4_By3_back(vector[double] Ap, vector[double] A, vector[double] va)


cdef class HQ_diff:
	cdef object hqsample
	def __cinit__(self, double Mass):
		self.hqsample = HqEvo.HqEvo(
		options={'transport': {'2->2':True, '2->3':False, '3->2':False},
			 'mass': Mass, 
			 'Nf': 3,
			 'Kfactor': 1.0,
			 'Tc': 0.154,
			 'mD': {'mD-model': 0, 'mTc': 5., 'slope': 0.5, 'curv': -1.2}},
		table_folder="./tables"+str(Mass),
		refresh_table=True	)

		# this function needs to be cdef so that we can use the reference copy
	cpdef (int, double, vector[double]) update_HQ_LBT(self, vector[double] p1_lab, vector[double] v3cell, double Temp, double mean_dt23_lab, double mean_dt32_lab):
		# Define local variables
		cdef double s, L1, L2, Lk, x2, xk, a1=0.6, a2=0.6
		cdef double dt_cell, dt23_com
		cdef size_t i=0
		cdef int channel
		cdef vector[double] p1_cell, p1_cell_Z, p1_com, \
			 p1_com_Z_new, p1_com_new, \
			 p1_cell_Z_new, p1_cell_new,\
			 pnew, fs, Pcom,	 \
			 dx23_cell, dx32_cell, \
			 v3com, pbuffer

		# Boost p1 (lab) to p1 (cell)
		boost4_By3(p1_cell, p1_lab, v3cell)

		# displacement vector within dx23_lab and dx32_lab seen from cell frame
		dx23_cell.resize(4)
		dx32_cell.resize(4)
		for i in range(4):
			dx23_cell[i] = p1_cell[i]/p1_lab[0]*mean_dt23_lab
			dx32_cell[i] = p1_cell[i]/p1_lab[0]*mean_dt32_lab

		# Sample channel in cell, return channl index and evolution time seen from cell
		channel, dt_cell = self.hqsample.sample_channel(p1_cell[0], Temp, dx23_cell[0], dx32_cell[0])
		# Boost evolution time back to lab frame
		cdef double dtHQ = p1_lab[0]/p1_cell[0]*dt_cell
		
		# If not scattered, return channel=-1, evolution time in Lab frame and origin al p1(lab)
		if channel < 0:
			pnew = p1_lab
		else:
			# Sample initial state and return initial state particle four vectors 
			# Imagine rotate p1_cell to align with z-direction, construct p2_cell_align, ...				
			self.hqsample.sample_initial(channel, p1_cell[0], Temp, dx23_cell[0], dx32_cell[0])
			p1_cell_Z = self.hqsample.IS[0]
			# Center of mass frame of p1_cell_align and other particles, and take down orientation of p1_com
			Pcom.resize(4)
			for i in range(4):
				Pcom[i] = 0.
			for pp in self.hqsample.IS:
				for i in range(4):
					Pcom[i] += pp[i]
			s = Pcom[0]**2 - Pcom[1]**2 - Pcom[2]**2 - Pcom[3]**2
			v3com.resize(3)
			for i in range(3):
				v3com[i] = Pcom[i+1]/Pcom[0];

			boost4_By3(p1_com, p1_cell_Z, v3com)
			if channel in [4,5]: # channel=4,5 for 3 -> 2 kinetics
				L1 = sqrt(p1_com[0]**2 - self.M**2)
				boost4_By3(pbuffer, self.hqsample.IS[1], v3com)
				L2 = pbuffer[0]
				boost4_By3(pbuffer, self.hqsample.IS[2], v3com)
				Lk = pbuffer[0]
				x2 = L2/(L1+L2+Lk)
				xk = Lk/(L1+L2+Lk)
				a1 = x2 + xk; 
				a2 = (x2 - xk)/(1. - a1)
			dt23_com = p1_com[0]/p1_cell_Z[0]*dx23_cell[0]

			# Sample final state momentum in Com frame, with incoming paticles on z-axis
			self.hqsample.sample_final(channel, s, Temp, dt23_com, a1, a2)
			p1_com_Z_new = self.hqsample.FS[0]
			# Rotate final states back to original Com frame (not z-axis aligened)
			rotate_back_from_D(p1_com_new, p1_com_Z_new, p1_com[1], p1_com[2], p1_com[3])
			# boost back to cell frame z-align
			boost4_By3_back(p1_cell_Z_new, p1_com_new, v3com)	   
			# rotate back to original cell frame
			rotate_back_from_D(p1_cell_new, p1_cell_Z_new, p1_cell[1], p1_cell[2], p1_cell[3])
			# boost back to lab frame
			boost4_By3_back(pnew, p1_cell_new, v3cell)

		# return channel, dt in lab frame, updated momentum of heavy quark
		# the channels are: 0:Qq2Qq，1:Qg2Qg, 2:Qq2Qqg, 3:Qg2Qgg, 4:Qqg2Qq, 5:Qgg2Qg
		return channel, dtHQ, pnew
		

	cpdef (vector[double]) update_dt(self, double lab_time_step, vector[double] p1_lab, vector[double] v3cell, double Temp, double mean_dt23_lab, double mean_dt32_lab):
		# lab_time_step is in fm/c
		cdef double lab_time = 0.0		# in fm/c
		cdef int channel
		cdef vector[double] pnew = p1_lab
		cdef double dt_lab
		while lab_time < lab_time_step:
			channel, dt_lab, pnew = self.update_HQ_LBT(pnew, v3cell, Temp, mean_dt23_lab, mean_dt32_lab)
			lab_time += dt_lab * C1		# convert the GeV^-1 to fm/c
		return pnew
	
	
	
	
