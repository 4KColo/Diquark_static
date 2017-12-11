#!/usr/bin/env python
import numpy as np
import h5py
import csv
import random as rd
from scipy.spatial import cKDTree
from Medium_Read import hydro_reader
from Dynam_Initial_Sample import Dynam_Initial_Sample
import DisRec
import LorRot
import HQ_diffuse

#### ---------------------- some constants -----------------------------
alpha_s = 0.4				  # 0.4 for charmonium and 0.3 for bottom
M = 1.3 					  # GeV c-quark
a_B = 3.0/(alpha_s*M)
E_1S = alpha_s**2*M/9.0		  # T(1S), here is magnitude, true value is its negative
M_1S = M*2.0 - E_1S  		  # mass of T(1S)
C1 = 0.197327                 # 0.197 GeV*fm = 1
R_search = 3.0				  # (fm), pair-search radius in the recombination
T_1S = 0.4					  # melting temperature of T_1S = 400 MeV
Tc = 0.154				  	  # critical temperature of QGP


class QQ_evol:
####---- input the medium_type when calling the class ----####
	def __init__(self, medium_type = 'dynamical', centrality_str_given = '0-10', energy_GeV = 2760.0, recombine = True, HQ_scat = False):
		self.type = medium_type		
		self.recombine = recombine
		self.HQ_scat = HQ_scat
		self.centrality = centrality_str_given
		self.Ecm = energy_GeV
		## -------------- create the hydro reader ------------ ##
		self.hydro = hydro_reader( hydro_file_path = 'HydroData'+centrality_str_given+'.h5' )
		## -------------- create the rates reader ------------ ##
		self.rates = DisRec.DisRec()
		## -------------- create HQ-diff rates reader -------- ##
		if self.HQ_scat == True:
			self.HQ_event = HQ_diffuse.HQ_diff(Mass = M)
			self.Hdi_event = HQ_diffuse.HQ_diff(Mass = M_1S)

####---- initialize Q and T1S -- currently we only study T(1S) ----####
	def initialize(self):
		if self.type == 'dynamical':
			## ----- create dictionaries to store momenta, positions, id ----- ##
			self.Qlist = {'4-momentum': [], '3-position': [], 'id': 5, 'last_t23': [], 'last_t32': [], 'last_scatter_dt':[]}
			self.T1Slist = {'4-momentum': [], '3-position': [], 'id': 533, 'last_form_time': [], 'last_t23': [], 'last_t32': [], 'last_scatter_dt':[}

			
			## -------------- create init p,x sampler ------------ ##
			self.init = Dynam_Initial_Sample(energy_GeV = self.Ecm, centrality_str = self.centrality)
			## --------- sample initial momenta and positions -------- ##
			self.Qlist['4-momentum'] = self.init.Qinit_p()
			self.Qlist['3-position'] = self.init.Qinit_x()
			self.Qlist['last_t23'] = self.init.Qinit_t23()
			self.Qlist['last_t32'] = self.init.Qinit_t32()
			self.Qlist['last_scatter_dt'] = self.init.Qinit_dt_loss()
			## ------------- store the current time ------------ ##
			self.t = 0.0

			
#### ---------------- event evolution function ------------------ ####	
#---- tau0 = 0.6 fm/c is the hydro starting time; before that, free stream
#---- this is done via Medium_Read and T >= Tc, if tau < 0.6, Medium_read gives T = 0
	def run(self, dt = 0.04):
		len_Q = len(self.Qlist['4-momentum'])
		len_T1S = len(self.T1Slist['4-momentum'])
				
		### ----------- free stream these particles ------------###
		for i in range(len_Q):
			v3_Q = self.Qlist['4-momentum'][i][1:]/self.Qlist['4-momentum'][i][0]
			self.Qlist['3-position'][i] = self.Qlist['3-position'][i] + dt * v3_Q
		for i in range(len_T1S):
			v3_T1S = self.T1Slist['4-momentum'][i][1:]/self.T1Slist['4-momentum'][i][0]
			self.T1Slist['3-position'][i] = self.T1Slist['3-position'][i] + dt * v3_T1S	
		### ----------- end of free stream particles -----------###


		###!!! update the time here, otherwise z would be bigger than t !!!###
		self.t += dt

		### ------------- heavy quark diffusion --------------- ###
		if self.HQ_scat == True:
			for i in range(len_Q):
				#print self.t, self.Qlist['3-position'][i]
				T_Vxyz = self.hydro.cell_info(self.t, self.Qlist['3-position'][i])
				if T_Vxyz[0] >= Tc:
					timer = 0.0
					p_Q = self.Qlist['4-momentum'][i]
					dt23 = max(0.0, self.t - self.Qlist['last_t23'][i])
					dt32 = max(0.0, self.t - self.Qlist['last_t32'][i])
					dt_real = dt + self.Qlist['last_scatter_dt'][i]
					while timer <= dt_real:
						channel, dtHQ, p_Q = self.HQ_event.update_HQ_LBT(p_Q, T_Vxyz[1:], T_Vxyz[0], dt23, dt32)
						if channel == 2 or channel == 3:
							t23 = self.t + timer
							self.Qlist['last_t23'][i] = t23
						if channel == 4 or channel == 5:
							t32 = self.t + timer
							self.Qlist['last_t32'][i] = t32
						timer += dtHQ*C1
					self.Qlist['last_scatter_dt'][i] = dt_real - timer
					self.Qlist['4-momentum'][i] = p_Q
		### ----------- end of heavy quark diffusion ---------- ###
		
		
		### ------------- heavy diquark diffusion ------------- ###
		if self.HQ_scat == True:
			for i in range(len_T1S):
				T_Vxyz = self.hydro.cell_info(self.t, self.T1Slist['3-position'][i])
				if T_Vxyz[0] >= Tc:
					timer = 0.0
					p_di = self.T1Slist['4-momentum'][i]
					dt23 = max(0.0, self.t - self.T1Slist['last_t23'][i])
					dt32 = max(0.0, self.t - self.T1Slist['last_t32'][i])
					dt_real = dt + self.T1Slist['last_scatter_dt'][i]
					while timer <= dt_real:
						channel, dtHdi, p_di = self.Hdi_event.update_HQ_LBT(p_di, T_Vxyz[1:], T_Vxyz[0], dt23, dt32)
						if channel == 2 or channel == 3:
							t23 = self.t + timer
							self.T1Slist['last_t23'][i] = t23
						if channel == 4 or channel == 5:
							t32 = self.t + timer
							self.T1Slist['last_t32'][i] = t32
						timer += dtHdi*C1
					self.T1Slist['last_scatter_dt'][i] = dt_real - timer
					self.T1Slist['4-momentum'][i] = p_di
		### ---------- end of heavy diquark diffusion --------- ###
		


		### -------------------- decay ------------------------ ###
		delete_T1S = []
		add_pQ = []
		add_xQ = []
		add_t23 = []
		add_t32 = []
		add_dt_last = []
		
		for i in range(len_T1S):
			T_Vxyz = self.hydro.cell_info(self.t, self.T1Slist['3-position'][i])		# temp, vx, vy, vz
			# only consider dissociation and recombination if in the de-confined phase
			if T_Vxyz[0] >= Tc:	
				p4_in_hydrocell = LorRot.lorentz(self.T1Slist['4-momentum'][i], T_Vxyz[1:])		# boost to hydro cell
				v3_in_hydrocell = p4_in_hydrocell[1:]/p4_in_hydrocell[0]
				v_in_hydrocell = np.sqrt(np.sum(v3_in_hydrocell**2))
				rate_decay = self.rates.get_R_1S_dis( v_in_hydrocell, T_Vxyz[0] )		# GeV
				
				# one half is for final-state identical particles
				if 0.5*rate_decay *dt* p4_in_hydrocell[0]/self.T1Slist['4-momentum'][i][0]/C1 >= np.random.rand(1):
				# dt = 0.04 is time in lab frame, dt' = dt*E'/E is time in hydro cell frame
					delete_T1S.append(i)
					# initial gluon sampling: incoming momentum
					q, costheta1, phi1 = self.rates.pydecay_sample_1S_init( v_in_hydrocell, T_Vxyz[0] )
					sintheta1 = np.sqrt(1. - costheta1**2)
					# final QQ sampling: relative momentum
					p_rel, costheta2, phi2 = self.rates.pydecay_sample_1S_final(q)
					sintheta2 = np.sqrt(1. - costheta2**2)
				
					# all the following three momentum components are in the quarkonium rest frame
					tempmomentum_g = np.array([q*sintheta1*np.cos(phi1), q*sintheta1*np.sin(phi1), q*costheta1])
					tempmomentum_Q = np.array([p_rel*sintheta2*np.cos(phi2), p_rel*sintheta2*np.sin(phi2), p_rel*costheta2])
					#tempmomentum_Qbar = -tempmomentum_Q	#(true in the quarkonium rest frame)
				
					# add the recoil momentum from the gluon
					recoil_p_Q1 = 0.5*tempmomentum_g + tempmomentum_Q
					recoil_p_Q2 = 0.5*tempmomentum_g - tempmomentum_Q
				
					# energy of Q and Qbar
					E_Q1 = np.sqrt(  np.sum(recoil_p_Q1**2) + M**2  )
					E_Q2 = np.sqrt(  np.sum(recoil_p_Q2**2) + M**2  )
				
					# Q1, Q2 momenta need to be rotated from the v = z axis to hydro cell frame
					# first get the rotation matrix angles
					theta_rot, phi_rot = LorRot.angle( v3_in_hydrocell )
					# then do the rotation
					rotmomentum_Q1 = LorRot.rotation(recoil_p_Q1, theta_rot, phi_rot)
					rotmomentum_Q2 = LorRot.rotation(recoil_p_Q2, theta_rot, phi_rot)
				
					# we now transform them back to the hydro cell frame
					momentum_Q1 = LorRot.lorentz(np.append(E_Q1, rotmomentum_Q1), -v3_in_hydrocell)		# final momentum of Q1
					momentum_Q2 = LorRot.lorentz(np.append(E_Q2, rotmomentum_Q2), -v3_in_hydrocell)		# final momentum of Q2
					# then transform back to the lab frame
					momentum_Q1 = LorRot.lorentz(momentum_Q1, -T_Vxyz[1:])
					momentum_Q2 = LorRot.lorentz(momentum_Q2, -T_Vxyz[1:])
				
					# positions of Q1 and Q2
					position_Q = self.T1Slist['3-position'][i]
					#position_Q2 = position_Q1
		
					# add x and p for the QQ to the temporary list
					add_pQ.append(momentum_Q1)
					add_pQ.append(momentum_Q2)
					add_xQ.append(position_Q)
					add_xQ.append(position_Q)
					add_t23.append(self.t)
					add_t23.append(self.t)
					add_t32.append(self.t)
					add_t32.append(self.t)
					add_dt_last.append(0.0)
					add_dt_last.append(0.0)
		### ------------------ end of decay ------------------- ###
		
		
		
		### ------------------ recombination ------------------ ###
		if self.recombine == True:
			delete_Q = []
			
			if len_Q != 0:
				pair_search = cKDTree(self.Qlist['3-position'])
				# for each Q1, obtain the Q2 indexes within R_search
				pair_list = pair_search.query_ball_point(self.Qlist['3-position'], r = R_search)
			
			for i in range(len_Q):					# loop over Q1
				len_recoQ2 = len(pair_list[i])
				reco_rate = []
				for j in range(len_recoQ2):			# loop over Q2 within R_search
					# positions in lab frame
					xQ1 = self.Qlist['3-position'][i]
					xQ2 = self.Qlist['3-position'][pair_list[i][j]]
					x_rel = xQ1 - xQ2
					x_CM = 0.5*( xQ1 + xQ2 )					
					T_Vxyz = self.hydro.cell_info(self.t, x_CM)
					#rdotp = np.sum( x_rel* (self.Qlist['4-momentum'][i][1:] - self.Qlist['4-momentum'][pair_list[i][j]][1:]) )
					if  T_Vxyz[0] >= Tc and pair_list[i][j] not in delete_Q and i != pair_list[i][j]:
					#if T_Vxyz[0] >= Tc and rdotp < 0.0 and pair_list[i][j] not in delete_Qbar and i != pair_list[i][j]:
						r_rel = np.sqrt(np.sum(x_rel**2))
						# CM energy in the lab frame
						p_CMlab = self.Qlist['4-momentum'][i][1:] + self.Qlist['4-momentum'][pair_list[i][j]][1:]
						E_CMlab = np.sqrt(np.sum(p_CMlab**2) + (2.*M)**2)
						
						# momenta in hydro cell frame
						pQ1 = LorRot.lorentz(self.Qlist['4-momentum'][i], T_Vxyz[1:])
						pQ2 = LorRot.lorentz(self.Qlist['4-momentum'][pair_list[i][j]], T_Vxyz[1:])
						
						# CM momentum and velocity in hydro cell frame
						p_CM = pQ1[1:] + pQ2[1:]		# M_tot = 2M
						p_CM_sqd = np.sum(p_CM**2)
						E_CM = np.sqrt(p_CM_sqd + (2.*M)**2)
						v_CM_abs = np.sqrt(p_CM_sqd)/E_CM
						v_CM = p_CM/E_CM
						
						# viewed in the CM frame
						pQ1_CM = LorRot.lorentz(pQ1, v_CM)
						pQ2_CM = LorRot.lorentz(pQ2, v_CM)
						
						# relative momentum inside CM frame
						p_rel = 0.5*(pQ1_CM - pQ2_CM)
						p_rel_abs = np.sqrt(np.sum(p_rel**2))
						
						reco_rate.append(self.rates.get_R_1S_reco(v_CM_abs, T_Vxyz[0], p_rel_abs, r_rel)*E_CM/E_CMlab)
						# we move the E_CMcell / E_CMlab above from below to avoid no definition
					else:	# rdotp >= 0
						reco_rate.append(0.)
				
				# get the recombine probability in the hydro cell frame
				# dt' in hydro cell = E_CMcell / E_CMlab * dt in lab frame
				# the factor of 2 is for the theta function normalization
				# one half avoiding double counting in the loops
				reco_prob = 0.5*6./9.*np.array(reco_rate)*dt/C1
				total_reco_prob = np.sum(reco_prob)
				reject_prob = np.random.rand(1)
				if total_reco_prob > reject_prob:
					delete_Q.append(i)		# remove this Q later
					# find the Qbar we need to remove
					a = 0.0
					for j in range(len_recoQ2):
						if a <= reject_prob <= a + reco_prob[j]:
							k = j
							break
						a += reco_prob[j]
					delete_Q.append(pair_list[i][k])
					
					# re-construct the reco event and sample initial and final states
					# positions and local temperature
					xQ1 = self.Qlist['3-position'][i]
					xQ2 = self.Qlist['3-position'][pair_list[i][k]]
					x_CM = 0.5*( xQ1 + xQ2 )					
					T_Vxyz = self.hydro.cell_info(self.t, x_CM)
					
					# momenta
					pQ1 = LorRot.lorentz(self.Qlist['4-momentum'][i], T_Vxyz[1:])
					pQ2 = LorRot.lorentz(self.Qlist['4-momentum'][pair_list[i][k]], T_Vxyz[1:])
					
					# CM momentum and velocity
					p_CM = pQ1[1:] + pQ2[1:]		# M_tot = 2M
					p_CM_sqd = np.sum(p_CM**2)
					E_CM = np.sqrt(p_CM_sqd + (2.*M)**2)
					v_CM_abs = np.sqrt(p_CM_sqd)/E_CM
					v_CM = p_CM/E_CM
						
					# viewed in the CM frame
					pQ1_CM = LorRot.lorentz(pQ1, v_CM)
					pQ2_CM = LorRot.lorentz(pQ2, v_CM)
						
					# relative momentum inside CM frame
					p_rel = 0.5*(pQ1_CM - pQ2_CM)
					p_rel_abs = np.sqrt(np.sum(p_rel**2))
				
					# calculate the final quarkonium momenta in the CM frame of QQbar
					q_T1S, costhetaT, phiT = self.rates.pyreco_sample_1S_final(v_CM_abs, T_Vxyz[0], p_rel_abs)
					sinthetaT = np.array(1.0-costhetaT**2)
					# get the 3-component of T1S momentum, where v_CM = z axis
					tempmomentum_T = np.array([q_T1S*sinthetaT*np.cos(phiT), q_T1S*sinthetaT*np.sin(phiT), q_T1S*costhetaT])
					E_T1S = np.sqrt( np.sum(tempmomentum_T**2)+M_1S**2 )
					
					# need to rotate the vector, v is not the z axis in hydro cell frame
					theta_rot, phi_rot = LorRot.angle(v_CM)
					rotmomentum_T = LorRot.rotation(tempmomentum_T, theta_rot, phi_rot)

					# lorentz back to the hydro cell frame
					momentum_T1S = LorRot.lorentz( np.append(E_T1S, rotmomentum_T), -v_CM )
					# lorentz back to the lab frame
					momentum_T1S = LorRot.lorentz( momentum_T1S, -T_Vxyz[1:] )
					
					# positions of the quarkonium
					position_T1S = x_CM
					
					# update the quarkonium list
					if len_T1S == 0:
						self.T1Slist['4-momentum'] = np.array([momentum_T1S])
						self.T1Slist['3-position'] = np.array([position_T1S])
						self.T1Slist['last_form_time'] = np.array([self.t])
						self.T1Slist['last_t23'] = np.array([self.t])
						self.T1Slist['last_t32'] = np.array([self.t])
						self.T1Slist['last_scatter_dt'] = np.array([0.0])
					else:
						self.T1Slist['4-momentum'] = np.append(self.T1Slist['4-momentum'], [momentum_T1S], axis=0)
						self.T1Slist['3-position'] = np.append(self.T1Slist['3-position'], [position_T1S], axis=0)
						self.T1Slist['last_form_time'] = np.append(self.T1Slist['last_form_time'], self.t)
						self.T1Slist['last_t23'] = np.append(self.T1Slist['last_t23'], self.t)
						self.T1Slist['last_t32'] = np.append(self.T1Slist['last_t32'], self.t)
						self.T1Slist['last_scatter_dt'] = np.append(self.T1Slist['last_scatter_dt'], 0.0)
						
			## now update Q1 and Q2 lists
			self.Qlist['4-momentum'] = np.delete(self.Qlist['4-momentum'], delete_Q, axis=0)
			self.Qlist['3-position'] = np.delete(self.Qlist['3-position'], delete_Q, axis=0)
			self.Qlist['last_t23'] = np.delete(self.Qlist['last_t23'], delete_Q)
			self.Qlist['last_t32'] = np.delete(self.Qlist['last_t32'], delete_Q)
			self.Qlist['last_scatter_dt'] = np.delete(self.Qlist['last_scatter_dt'], delete_Q)
		### -------------- end of recombination --------------- ###
		
		
		### -------------- update lists due to decay ---------- ###
		add_pQ = np.array(add_pQ)
		add_xQ = np.array(add_xQ)
		add_t23 = np.array(add_t23)
		add_t32 = np.array(add_t32)
		if len(add_pQ):	
			# if there is at least quarkonium decays, we need to update all the three lists
			self.T1Slist['3-position'] = np.delete(self.T1Slist['3-position'], delete_T1S, axis=0) # delete along the axis = 0
			self.T1Slist['4-momentum'] = np.delete(self.T1Slist['4-momentum'], delete_T1S, axis=0)
			self.T1Slist['last_form_time'] = np.delete(self.T1Slist['last_form_time'], delete_T1S)
			self.T1Slist['last_t23'] = np.delete(self.T1Slist['last_t23'], delete_T1S)
			self.T1Slist['last_t32'] = np.delete(self.T1Slist['last_t32'], delete_T1S)
			self.T1Slist['last_scatter_dt'] = np.delete(self.Qlist['last_scatter_dt'], delete_T1S)
			
			if len(self.Qlist['4-momentum']) == 0:
				self.Qlist['3-position'] = np.array(add_xQ)
				self.Qlist['4-momentum'] = np.array(add_pQ)
				self.Qlist['last_t23'] = np.array(add_t23)
				self.Qlist['last_t32'] = np.array(add_t32)
				self.Qlist['last_scatter_dt'] = np.array(add_dt_last)
			else:
				self.Qlist['3-position'] = np.append(self.Qlist['3-position'], add_xQ, axis=0)
				self.Qlist['4-momentum'] = np.append(self.Qlist['4-momentum'], add_pQ, axis=0)
				self.Qlist['last_t23'] = np.append(self.Qlist['last_t23'], add_t23)
				self.Qlist['last_t32'] = np.append(self.Qlist['last_t23'], add_t32)
				self.Qlist['last_scatter_dt'] = np.append(self.Qlist['last_scatter_dt'], add_dt_last)
		### ---------- end of update lists due to decay ------- ###
		
#### ----------------- end of evolution function ----------------- ####	

#### ----------------- define a clear function ------------------- ####
	def dict_clear(self):
		self.Qlist.clear()
		self.T1Slist.clear()





