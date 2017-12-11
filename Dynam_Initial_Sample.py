#!/usr/bin/env python
import numpy as np
import random as rd
import h5py
import LorRot
# Parameters: 	norm=125, nucleon-width=0.5fm, fluctuation=1,
#				(for norm=?, see 1605.03954)
# 				grid-max=15.05fm, (grid value is at the center of grid) 
#				grid-step=0.1fm, b-min=0fm, b-max=5.0fm
# EoS: HotQCD, e(T=Tc) = 0.329 Gev/fm^3, eta/s = 0.08+1*(T-Tc), bulk_norm=0.01, bulk_width=0.01

def interpolate_2d(table, r1, r2, i1, i2, max1, max2):
	# r = where in between, i = index, max = max index
	w1 = [1.0-r1, r1]
	w2 = [1.0-r2, r2]
	interpolation = 0.0
	for i in range(2):
		for j in range(2):
			interpolation += table[(int(i1)+i)%max1][(int(i2)+j)%max2] *w1[i]*w2[j]
	return interpolation

M = 1.3		# mass of charm quark in GeV
twopi = 2.0*np.pi

Xsect = {'2760': 2.564}				# mb
Taa = {'2760': {'0-10': 23.0}}		# mb^-1
class Dynam_Initial_Sample:
	def __init__(self, energy_GeV = 2760.0, centrality_str = '0-10'):
	
		### -------- store the position information -------- ###
		file_TAB = h5py.File('ic-'+centrality_str+'-avg.hdf5','r')
		Tab = np.array(file_TAB['TAB_0'].value)
		dx = 0.1 	# fm
		dy = 0.1	# fm
		Nx, Ny = Tab.shape
		Tab_flat = Tab.reshape(-1)
		T_tot = np.sum(Tab_flat)
		T_AA_mb = Taa[str(int(energy_GeV))][centrality_str]
		T_norm = Tab_flat/T_tot
		T_accum = np.zeros_like(T_norm, dtype=np.double)
		for index, value in enumerate(T_norm):
			T_accum[index] = T_accum[index-1] + value
		file_TAB.close()
			
		
		### -------- store the momentum information -------- ###
		ymin = 0.0
		ymax = 8.0
		ny = 41
		pTmin = 0.1
		pTmax = 29.1
		npT = 100
		dy = (ymax - ymin)/ny
		dpT = (pTmax - pTmin)/npT
		y_list = np.linspace(ymin, ymax, ny)
		pT_list = np.linspace(pTmin, pTmax, npT)
				
		data = np.fromfile('FONLL-M=1.3GeV-spectra2760GeVPbPb-dsigma-dpt2-dy.dat', dtype=float, sep=" ")
		
		dsigma_dpT2dy = np.zeros((100,41))
		for i in range(100):	# i is pT
			for j in range(41):	# j is y
				dsigma_dpT2dy[i][j] = data[(41*i + j)*3 + 2]
		dsigma_normed = dsigma_dpT2dy/(np.max(dsigma_dpT2dy)+0.001)
		
		N = Xsect[str(int(energy_GeV))] * T_AA_mb	# averaged number of charm produced
		Nsam = int(N) + 1	# sampled number

		
		### ---------- sample initial conditions --------- ###
		p4_Q = []
		x3_Q = []
		t23 = []
		t32 = []
		dt_loss = [] # loss due to finite time truncation of HQ_diffusion
		
		### ---------- sample postions --------- ###
		for i in range(Nsam):
			r_N = rd.uniform(0.0, Nsam+0.0)
			if r_N <= N:
				## positions
				r_xy = rd.uniform(0.0, 1.0)
				i_r = np.searchsorted(T_accum, r_xy)
				i_x = np.floor((i_r+0.0)/Ny)
				i_y = i_r - i_x*Ny
				i_x += np.random.rand()
				i_y += np.random.rand()
				x = (i_x - Nx/2.)*dx
				y = (i_y - Ny/2.)*dy
				x3_Q.append(np.array([x,y,0.0]))
		self.x3_Q = np.array(x3_Q)
		len_x = len(self.x3_Q)
		
		### ---------- sample momenta --------- ###
		for i in range(len_x):
			while True:
				y_try = rd.uniform(-8.0, 8.0)
				pT_try = rd.uniform(0.1, 29.1)
				y_try = abs(y_try)
				i_y = np.searchsorted(y_list, y_try)
				i_pT = np.searchsorted(pT_list, pT_try)
				r_y = (y_try - y_list[i_y])/dy
				r_pT = (pT_try - pT_list[i_pT])/dpT
				
				dsigma_try = np.random.rand()
				if dsigma_try < interpolate_2d(dsigma_normed, r_pT, r_y, i_pT, i_y, npT, ny):
					mT = np.sqrt(M**2 + pT_try**2)
					pz = mT * np.sinh(y_try)
					costheta = rd.uniform(-1.0, 1.0)
					phi = rd.uniform(0.0, twopi)
					px = pT_try * costheta * np.cos(phi)
					py = pT_try * costheta * np.sin(phi)
					energy = pz/np.tanh(y_try)
					p4_Q.append([energy, px, py, pz])
					break
		self.p4_Q = np.array(p4_Q)
		
		### ---------- sample t23 and t32 --------- ###
		for i in range(len_x):
			t23.append(0.1)
			t32.append(0.1)
			dt_loss.append(0.0)
		self.t23 = np.array(t23)
		self.t32 = np.array(t32)
		self.dt_loss = np.array(dt_loss)
	def Qinit_p(self):
		return self.p4_Q
	
	def Qinit_x(self):
		return self.x3_Q
	
	def Qinit_t23(self):
		return self.t23
	
	def Qinit_t32(self):
		return self.t32
	
	def Qinit_dt_loss(self):
		return self.dt_loss