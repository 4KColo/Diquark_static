#!/usr/bin/env python
import numpy as np
import h5py
import csv
import random as rd
from scipy.spatial import cKDTree
from Static_Initial_Sample import Static_Initial_Sample
from Static_HQ_evo import HQ_p_update
import DisRec
import LorRot


#### ---------------------- some constants -----------------------------
alpha_s = 0.3 				  # for charmonium
M = 1.29 					  # GeV c-quark
a_B = 3.0/(alpha_s*M)
E_1S = alpha_s**2*M/9.0		  # T(1S), here is magnitude, true value is its negative
M_1S = M*2.0 - E_1S  		  # mass of T(1S)
C1 = 0.197327                 # 0.197 GeV*fm = 1
R_search = 1.0				  # (fm), pair-search radius in the recombination
T_1S = 0.4					  # melting temperature of T_1S = 400 MeV


####--------- initial sample of heavy Q and Qbar using thermal distribution -------- ####
def thermal_dist(temp, mass, momentum):
    return momentum**2*np.exp(-np.sqrt(mass**2+momentum**2)/temp)


# sample according to the thermal distribution function
def thermal_sample(temp, mass):
    p_max = np.sqrt(2.0*temp**2 + 2.0*temp*np.sqrt(temp**2+mass**2))
    # most probably momentum
    p_uplim = 10.0*p_max
    y_uplim = thermal_dist(temp, mass, p_max)
    while True:
        p_try = rd.uniform(0.0, p_uplim)
        y_try = rd.uniform(0.0, y_uplim)
        if y_try < thermal_dist(temp, mass, p_try):
            break
        
    E = np.sqrt(p_try**2+mass**2)
    cos_theta = rd.uniform(-1.0, 1.0)
    sin_theta = np.sqrt(1.0-cos_theta**2)
    phi = rd.uniform(0.0, 2.0*np.pi)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    return np.array([ E, p_try*sin_theta*cos_phi, p_try*sin_theta*sin_phi, p_try*cos_theta ])

####---- end of initial sample of heavy Q and Qbar using thermal distribution ---- ####




####--------- initial sample of heavy Q and Qbar using uniform distribution ------ ####
def uniform_sample(Pmax, mass):
    px, py, pz = (np.random.rand(3)-0.5)*2*Pmax
    E = np.sqrt(mass**2 + px**2 + py**2 + pz**2)
    return np.array([E, px, py, pz])

####----- end of initial sample of heavy Q and Qbar using uniform distribution ---- ####



class QQbar_evol:
####---- input the medium_type when calling the class ----####
	def __init__(self, medium_type = 'static', temp_init = 0.3, recombine = True, HQ_scat = False):
		self.type = medium_type
		self.T = temp_init		# initial temperature in GeV	
		self.recombine = recombine
		self.HQ_scat = HQ_scat
		if self.HQ_scat == True:
			self.HQ_diff = HQ_p_update(Mass = M)
			self.T_diff = HQ_p_update(Mass = M_1S)
		## ---------- create the rates reader --------- ##
		self.rates = DisRec.DisRec()
		
####---- initialize Q, Qbar, Quarkonium -- currently we only study Upsilon(1S) ----####
	def initialize(self, N_Q = 100, N_Qbar = 100, N_T1S = 10, Lmax = 10.0, thermal_dist = True,
	fonll_dist = False, Fonll_path = False, uniform_dist = False, Pmax = 10.0, decaytestmode = False, P_decaytest = [0.0, 0.0, 5.0]):
		# initial momentum: thermal; Fonll (give the fonll file path), uniform in x,y,z (give the Pmax in GeV)
        # if decaytestmode: give initial Px,Py,Pz in GeV
		if self.type == 'static':
			## ----- initialize the clock to keep track of time
			self.t = 0.0
			## ----- store the box side length
			self.Lmax = Lmax
			## ----- create dictionaries to store momenta, positions, id ----- ##
			self.Qlist = {'4-momentum': [], '3-position': [], 'id': 5}
			self.Qbarlist = {'4-momentum': [], '3-position': [], 'id': -5}
			self.T1Slist = {'4-momentum': [], '3-position': [], 'id': 533, 'last_form_time': []}
			
			## --------- sample initial momenta and positions -------- ##
			if thermal_dist == True:
				for i in range(N_Q):
					self.Qlist['4-momentum'].append( thermal_sample(self.T, M) )
					self.Qlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_Qbar):
					self.Qbarlist['4-momentum'].append( thermal_sample(self.T, M) )
					self.Qbarlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_T1S):
					self.T1Slist['4-momentum'].append( thermal_sample(self.T, M_1S) )
					self.T1Slist['3-position'].append( np.random.rand(3)*Lmax )
					self.T1Slist['last_form_time'].append( self.t )
			
			if fonll_dist == True:
				p_generator = Static_Initial_Sample(Fonll_path, rapidity = 0.)
				
				for i in range(N_Q):
					self.Qlist['4-momentum'].append( p_generator.p_HQ_sample(M) )
					self.Qlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_Qbar):
					self.Qbarlist['4-momentum'].append( p_generator.p_HQ_sample(M) )
					self.Qbarlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_T1S):
					self.T1Slist['4-momentum'].append( p_generator.p_T1S_sample(M, M_1S) )
					self.T1Slist['3-position'].append( np.random.rand(3)*Lmax )
					self.T1Slist['last_form_time'].append( self.t )
			
			if uniform_dist == True:
				for i in range(N_Q):
					self.Qlist['4-momentum'].append( uniform_sample(Pmax, M) )
					self.Qlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_Qbar):
					self.Qbarlist['4-momentum'].append( uniform_sample(Pmax, M) )
					self.Qbarlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_T1S):
					self.T1Slist['4-momentum'].append( uniform_sample(Pmax, M_1S) )
					self.T1Slist['3-position'].append( np.random.rand(3)*Lmax )
					self.T1Slist['last_form_time'].append( self.t )
			
			if decaytestmode == True:
				E_decaytest = np.sqrt(M_1S**2 + P_decaytest[0]**2 + P_decaytest[1]**2 + P_decaytest[2]**2)
				for i in range(N_Q):
					self.Qlist['4-momentum'].append( [M, 0.0, 0.0, 0.0] )
					self.Qlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_Qbar):
					self.Qbarlist['4-momentum'].append( [M, 0.0, 0.0, 0.0] )
					self.Qbarlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_T1S):
					self.T1Slist['4-momentum'].append( np.append(E_decaytest, P_decaytest) )
					self.T1Slist['3-position'].append( np.random.rand(3)*Lmax )
					self.T1Slist['last_form_time'].append( self.t )
			                
			self.Qlist['4-momentum'] = np.array(self.Qlist['4-momentum'])
			self.Qlist['3-position'] = np.array(self.Qlist['3-position'])
			self.Qbarlist['4-momentum'] = np.array(self.Qbarlist['4-momentum'])
			self.Qbarlist['3-position'] = np.array(self.Qbarlist['3-position'])
			self.T1Slist['4-momentum'] = np.array(self.T1Slist['4-momentum'])
			self.T1Slist['3-position'] = np.array(self.T1Slist['3-position'])
			self.T1Slist['last_form_time'] = np.array(self.T1Slist['last_form_time'])
			

#### ---------------- event evolution function ------------------ ####
	def run(self, dt = 0.04, temp_run = -1.0):			# universal time to consider recombination
		len_Q = len(self.Qlist['4-momentum'])
		len_Qbar = len(self.Qbarlist['4-momentum'])
		len_T1S = len(self.T1Slist['4-momentum'])
		
		if temp_run != -1.0:
			self.T = temp_run
		
		### ------------- heavy quark diffusion --------------- ###
		if self.HQ_scat == True:
			for i in range(len_Q):
				self.Qlist['4-momentum'][i] = self.HQ_diff.update(HQ_Ep = self.Qlist['4-momentum'][i], Temp = self.T, time_step = dt)
			for i in range(len_Qbar):
				self.Qbarlist['4-momentum'][i] = self.HQ_diff.update(HQ_Ep = self.Qbarlist['4-momentum'][i], Temp = self.T, time_step = dt)
			for i in range(len_T1S):
				self.T1Slist['4-momentum'][i] = self.T_diff.update(HQ_Ep = self.T1Slist['4-momentum'][i], Temp = self.T, time_step = dt)
		### ----------- end of heavy quark diffusion ---------- ###




		### ----------- free stream these particles ------------###
		for i in range(len_Q):
			v3_Q = self.Qlist['4-momentum'][i][1:]/self.Qlist['4-momentum'][i][0]
			self.Qlist['3-position'][i] = (self.Qlist['3-position'][i] + dt * v3_Q)%self.Lmax
		for i in range(len_Qbar):
			v3_Qbar = self.Qbarlist['4-momentum'][i][1:]/self.Qbarlist['4-momentum'][i][0]
			self.Qbarlist['3-position'][i] = (self.Qbarlist['3-position'][i] + dt * v3_Qbar)%self.Lmax	
		for i in range(len_T1S):
			v3_T1S = self.T1Slist['4-momentum'][i][1:]/self.T1Slist['4-momentum'][i][0]
			self.T1Slist['3-position'][i] = (self.T1Slist['3-position'][i] + dt * v3_T1S)%self.Lmax
		### ----------- end of free stream particles -----------###




		### -------------------- decay ------------------------ ###
		delete_T1S = []
		add_pQ = []
		add_pQbar = []
		add_xQ = []
		#add_xQbar = [] the positions of Q and Qbar are the same
		

		for i in range(len_T1S):
			p4_in_box = self.T1Slist['4-momentum'][i]
			v3_in_box = p4_in_box[1:]/p4_in_box[0]
			v_in_box = np.sqrt(np.sum(v3_in_box**2))
			rate_decay = self.rates.get_R_1S_dis( v_in_box, self.T )		# GeV

			if rate_decay * dt/C1 >= np.random.rand(1):
				delete_T1S.append(i)
				# initial gluon sampling: incoming momentum
				q, costheta1, phi1 = self.rates.pydecay_sample_1S_init( v_in_box, self.T )
				sintheta1 = np.sqrt(1. - costheta1**2)
				# final QQbar sampling: relative momentum
				p_rel, costheta2, phi2 = self.rates.pydecay_sample_1S_final(q)
				sintheta2 = np.sqrt(1. - costheta2**2)
				
				# all the following three momentum components are in the quarkonium rest frame
				tempmomentum_g = np.array([q*sintheta1*np.cos(phi1), q*sintheta1*np.sin(phi1), q*costheta1])
				tempmomentum_Q = np.array([p_rel*sintheta2*np.cos(phi2), p_rel*sintheta2*np.sin(phi2), p_rel*costheta2])
				#tempmomentum_Qbar = -tempmomentum_Q	#(true in the quarkonium rest frame)
				
				# add the recoil momentum from the gluon
				recoil_p_Q = 0.5*tempmomentum_g + tempmomentum_Q
				recoil_p_Qbar = 0.5*tempmomentum_g - tempmomentum_Q
				
				# energy of Q and Qbar
				E_Q = np.sqrt(  np.sum(recoil_p_Q**2) + M**2  )
				E_Qbar = np.sqrt(  np.sum(recoil_p_Qbar**2) + M**2  )
				
				# Q, Qbar momenta need to be rotated from the v = z axis to what it is in the box frame
				# first get the rotation matrix angles
				theta_rot, phi_rot = LorRot.angle( v3_in_box )
				# then do the rotation
				rotmomentum_Q = LorRot.rotation(recoil_p_Q, theta_rot, phi_rot)
				rotmomentum_Qbar = LorRot.rotation(recoil_p_Qbar, theta_rot, phi_rot)
				
				# we now transform them back to the box frame
				momentum_Q = LorRot.lorentz(np.append(E_Q, rotmomentum_Q), -v3_in_box)			# final momentum of Q
				momentum_Qbar = LorRot.lorentz(np.append(E_Qbar, rotmomentum_Qbar), -v3_in_box)	# final momentum of Qbar
				
				# positions of Q and Qbar
				position_Q = self.T1Slist['3-position'][i]
				#position_Qbar = position_Q
		
				# add x and p for the QQbar to the temporary list
				add_pQ.append(momentum_Q)
				add_pQbar.append(momentum_Qbar)
				add_xQ.append(position_Q)
				#add_xQbar.append(position_Qbar)
		
		### ------------------ end of decay ------------------- ###
		
		
		
		### ------------------ recombination ------------------ ###
		if self.recombine == True and len_Q != 0 and len_Qbar !=0:
			delete_Q = []
			delete_Qbar = []
			
			# make the periodic box 26 times bigger!
			Qbar_x_list = np.concatenate((self.Qbarlist['3-position'], 
			self.Qbarlist['3-position']+[0.0, 0.0, self.Lmax], self.Qbarlist['3-position']+[0.0, 0.0, -self.Lmax],
			self.Qbarlist['3-position']+[0.0, self.Lmax, 0.0], self.Qbarlist['3-position']+[0.0, -self.Lmax, 0.0],
			self.Qbarlist['3-position']+[self.Lmax, 0.0, 0.0], self.Qbarlist['3-position']+[-self.Lmax, 0.0, 0.0],
			self.Qbarlist['3-position']+[0.0, self.Lmax, self.Lmax], self.Qbarlist['3-position']+[0.0, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[0.0, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[0.0, -self.Lmax, -self.Lmax],
			self.Qbarlist['3-position']+[self.Lmax, 0.0, self.Lmax], self.Qbarlist['3-position']+[self.Lmax, 0.0, -self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, 0.0, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, 0.0, -self.Lmax],
			self.Qbarlist['3-position']+[self.Lmax, self.Lmax, 0.0], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, 0.0], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, 0.0], self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, 0.0],
			self.Qbarlist['3-position']+[self.Lmax, self.Lmax, self.Lmax], self.Qbarlist['3-position']+[self.Lmax, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, self.Lmax],
			self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, -self.Lmax]),
			axis = 0 )
			
			pair_search = cKDTree(Qbar_x_list)
			# for each Q, obtain the Qbar indexes within R_search
			pair_list = pair_search.query_ball_point(self.Qlist['3-position'], r = R_search)
			
			for i in range(len_Q):		# loop over Q
				len_recoQbar = len(pair_list[i])
				reco_rate = []
				for j in range(len_recoQbar):		# loop over Qbar within R_search
					# positions in lab frame
					xQ = self.Qlist['3-position'][i]
					xQbar = Qbar_x_list[pair_list[i][j]]
					x_rel = xQ - xQbar
					i_Qbar_mod = pair_list[i][j]%len_Qbar	# use for momentum and delete_index
					rdotp = np.sum( x_rel* (self.Qlist['4-momentum'][i][1:] - self.Qbarlist['4-momentum'][i_Qbar_mod][1:]) )
					
					if  rdotp < 0.0 and i_Qbar_mod not in delete_Qbar:
						r_rel = np.sqrt(np.sum(x_rel**2))
						x_CM = 0.5*( xQ + xQbar )
					
						# momenta in hydro cell frame
						pQ = self.Qlist['4-momentum'][i]
						pQbar = self.Qbarlist['4-momentum'][i_Qbar_mod]
						
						# CM momentum and velocity
						p_CM = pQ[1:] + pQbar[1:]		# M_tot = 2M
						p_CM_sqd = np.sum(p_CM**2)
						E_CM = np.sqrt(p_CM_sqd + (2.*M)**2)
						v_CM_abs = np.sqrt(p_CM_sqd)/E_CM
						v_CM = p_CM/E_CM
						
						# viewed in the CM frame
						pQ_CM = LorRot.lorentz(pQ, v_CM)
						pQbar_CM = LorRot.lorentz(pQbar, v_CM)
						
						# relative momentum inside CM frame
						p_rel = 0.5*(pQ_CM - pQbar_CM)
						p_rel_abs = np.sqrt(np.sum(p_rel**2))
												
						reco_rate.append(self.rates.get_R_1S_reco(v_CM_abs, self.T, p_rel_abs, r_rel))
						
					else:	# rdotp >= 0 or the Qbar has been taken by other Q's
						reco_rate.append(0.)
				
				# get the recombine probability
				# the factor of 2 is for the theta function normalization
				reco_prob = 2.*8./9.*np.array(reco_rate)*dt/C1
				total_reco_prob = np.sum(reco_prob)
				reject_prob = np.random.rand(1)
				if total_reco_prob > reject_prob:
					delete_Q.append(i)		# remove this Q later
					# find the Qbar we need to remove
					a = 0.0
					for j in range(len_recoQbar):
						if a <= reject_prob <= a + reco_prob[j]:
							k = j
							break
						a += reco_prob[j]
					delete_Qbar.append(pair_list[i][k]%len_Qbar)
					
					# re-construct the reco event and sample initial and final states
					# positions and local temperature
					xQ = self.Qlist['3-position'][i]
					xQbar = Qbar_x_list[pair_list[i][k]]
					x_CM = 0.5*( xQ + xQbar )
					
					# momenta
					pQ = self.Qlist['4-momentum'][i]
					pQbar = self.Qbarlist['4-momentum'][pair_list[i][k]%len_Qbar]
					
					# CM momentum and velocity
					p_CM = pQ[1:] + pQbar[1:]		# M_tot = 2M
					p_CM_sqd = np.sum(p_CM**2)
					E_CM = np.sqrt(p_CM_sqd + (2.*M)**2)
					v_CM_abs = np.sqrt(p_CM_sqd)/E_CM
					v_CM = p_CM/E_CM
						
					# viewed in the CM frame
					pQ_CM = LorRot.lorentz(pQ, v_CM)
					pQbar_CM = LorRot.lorentz(pQbar, v_CM)
						
					# relative momentum inside CM frame
					p_rel = 0.5*(pQ_CM - pQbar_CM)
					p_rel_abs = np.sqrt(np.sum(p_rel**2))
				
					# calculate the final quarkonium momenta in the CM frame of QQbar
					q_T1S, costhetaU, phiU = self.rates.pyreco_sample_1S_final(v_CM_abs, self.T, p_rel_abs)
					sinthetaU = np.array(1.0-costhetaU**2)
					# get the 3-component of T1S momentum, where v_CM = z axis
					tempmomentum_U = np.array([q_T1S*sinthetaU*np.cos(phiU), q_T1S*sinthetaU*np.sin(phiU), q_T1S*costhetaU])
					E_T1S = np.sqrt( np.sum(tempmomentum_U**2)+M_1S**2 )
					
					# need to rotate the vector, v is not the z axis in box frame
					theta_rot, phi_rot = LorRot.angle(v_CM)
					rotmomentum_U = LorRot.rotation(tempmomentum_U, theta_rot, phi_rot)

					# lorentz back to the box frame
					momentum_T1S = LorRot.lorentz( np.append(E_T1S, rotmomentum_U), -v_CM )
					
					# positions of the quarkonium
					position_T1S = x_CM%self.Lmax
					
					# update the quarkonium list
					if len_T1S == 0:
						self.T1Slist['4-momentum'] = np.array([momentum_T1S])
						self.T1Slist['3-position'] = np.array([position_T1S])
						self.T1Slist['last_form_time'] = np.array(self.t)
					else:
						self.T1Slist['4-momentum'] = np.append(self.T1Slist['4-momentum'], [momentum_T1S], axis=0)
						self.T1Slist['3-position'] = np.append(self.T1Slist['3-position'], [position_T1S], axis=0)
						self.T1Slist['last_form_time'] = np.append(self.T1Slist['last_form_time'], self.t)
						
			## now update Q and Qbar lists
			self.Qlist['4-momentum'] = np.delete(self.Qlist['4-momentum'], delete_Q, axis=0)
			self.Qlist['3-position'] = np.delete(self.Qlist['3-position'], delete_Q, axis=0)
			self.Qbarlist['4-momentum'] = np.delete(self.Qbarlist['4-momentum'], delete_Qbar, axis=0)
			self.Qbarlist['3-position'] = np.delete(self.Qbarlist['3-position'], delete_Qbar, axis=0)
		
		### -------------- end of recombination --------------- ###
		
		
		### -------------- update lists due to decay ---------- ###
		add_pQ = np.array(add_pQ)
		add_pQbar = np.array(add_pQbar)
		add_xQ = np.array(add_xQ)
		#add_xQbar = np.array(add_xQbar)
		if len(add_pQ):	
			# if there is at least quarkonium decays, we need to update all the three lists
			self.T1Slist['3-position'] = np.delete(self.T1Slist['3-position'], delete_T1S, axis=0) # delete along the axis = 0
			self.T1Slist['4-momentum'] = np.delete(self.T1Slist['4-momentum'], delete_T1S, axis=0)
			self.T1Slist['last_form_time'] = np.delete(self.T1Slist['last_form_time'], delete_T1S)
			
			if len(self.Qlist['4-momentum']) == 0:
				self.Qlist['3-position'] = np.array(add_xQ)
				self.Qlist['4-momentum'] = np.array(add_pQ)
			else:
				self.Qlist['3-position'] = np.append(self.Qlist['3-position'], add_xQ, axis=0)
				self.Qlist['4-momentum'] = np.append(self.Qlist['4-momentum'], add_pQ, axis=0)
			if len(self.Qbarlist['4-momentum']) == 0:
				self.Qbarlist['3-position'] = np.array(add_xQ)
				self.Qbarlist['4-momentum'] = np.array(add_pQbar)
			else:
				self.Qbarlist['3-position'] = np.append(self.Qbarlist['3-position'], add_xQ, axis=0)
				self.Qbarlist['4-momentum'] = np.append(self.Qbarlist['4-momentum'], add_pQbar, axis=0)
		
		### ---------- end of update lists due to decay ------- ###


		self.t += dt
#### ----------------- end of evolution function ----------------- ####





#### -------------------- test recombination -------------------- ####
	def testrun(self):
		len_Q = len(self.Qlist['4-momentum'])
		len_Qbar = len(self.Qbarlist['4-momentum'])
		
		total_reco_rate = 0.0
		
		# make the periodic box 26 times bigger!
		Qbar_x_list = np.concatenate((self.Qbarlist['3-position'], 
		self.Qbarlist['3-position']+[0.0, 0.0, self.Lmax], self.Qbarlist['3-position']+[0.0, 0.0, -self.Lmax],
		self.Qbarlist['3-position']+[0.0, self.Lmax, 0.0], self.Qbarlist['3-position']+[0.0, -self.Lmax, 0.0],
		self.Qbarlist['3-position']+[self.Lmax, 0.0, 0.0], self.Qbarlist['3-position']+[-self.Lmax, 0.0, 0.0],
		self.Qbarlist['3-position']+[0.0, self.Lmax, self.Lmax], self.Qbarlist['3-position']+[0.0, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[0.0, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[0.0, -self.Lmax, -self.Lmax],
		self.Qbarlist['3-position']+[self.Lmax, 0.0, self.Lmax], self.Qbarlist['3-position']+[self.Lmax, 0.0, -self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, 0.0, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, 0.0, -self.Lmax],
		self.Qbarlist['3-position']+[self.Lmax, self.Lmax, 0.0], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, 0.0], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, 0.0], self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, 0.0],
		self.Qbarlist['3-position']+[self.Lmax, self.Lmax, self.Lmax], self.Qbarlist['3-position']+[self.Lmax, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, self.Lmax],
		self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, -self.Lmax]),
		axis = 0 )
			
		pair_search = cKDTree(Qbar_x_list)
		# for each Q, obtain the Qbar indexes within R_search
		pair_list = pair_search.query_ball_point(self.Qlist['3-position'], r = R_search)
		
		for i in range(len_Q):
			len_recoQbar = len(pair_list[i])
			reco_rate = []
			for j in range(len_recoQbar):		# loop over Qbar within R_search
				xQ = self.Qlist['3-position'][i]
				xQbar = Qbar_x_list[pair_list[i][j]]
				x_rel = xQ - xQbar
				i_Qbar_mod = pair_list[i][j]%len_Qbar	# use for momentum and delete_index
				rdotp = np.sum( x_rel* (self.Qlist['4-momentum'][i][1:] - self.Qbarlist['4-momentum'][i_Qbar_mod][1:]) )
					
				if  rdotp < 0.0:
					r_rel = np.sqrt(np.sum(x_rel**2))
					x_CM = 0.5*( xQ + xQbar )
					
					# momenta in the static medium frame
					pQ = self.Qlist['4-momentum'][i]
					pQbar = self.Qbarlist['4-momentum'][i_Qbar_mod]
						
					# CM momentum and velocity
					p_CM = pQ[1:] + pQbar[1:]		# M_tot = 2M
					p_CM_sqd = np.sum(p_CM**2)
					E_CM = np.sqrt(p_CM_sqd + (2.*M)**2)
					v_CM_abs = np.sqrt(p_CM_sqd)/E_CM
					v_CM = p_CM/E_CM
						
					# viewed in the CM frame
					pQ_CM = LorRot.lorentz(pQ, v_CM)
					pQbar_CM = LorRot.lorentz(pQbar, v_CM)
						
					# relative momentum inside CM frame
					p_rel = 0.5*(pQ_CM - pQbar_CM)
					p_rel_abs = np.sqrt(np.sum(p_rel**2))
					
					# the factor of 2 is to account the theta function renormalization					
					reco_rate.append(2.0*self.rates.get_R_1S_reco(v_CM_abs, self.T, p_rel_abs, r_rel))
			
			reco_rate = np.array(reco_rate)
			total_reco_rate += np.sum(reco_rate)
		
		return total_reco_rate/len_Q
			
			
			
			
			
			
			
		