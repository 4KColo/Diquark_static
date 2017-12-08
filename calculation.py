#!/usr/bin/env python
import numpy as np
import scipy.integrate as si
from Dynam_diquark_evolution import QQ_evol
import h5py


#### ------------ multiple runs averaged and compare ---------------- ####
N_ave = 3000		# #of parallel runnings
N_step = 400
dt = 0.04
tmax = N_step*dt
t = np.linspace(0.0, tmax, N_step+1)
N1s_t = []
Nc_t = []
T1s_p4 = []		# to store T1S momenta at the end of each event

# define the event generator
event_gen = QQ_evol(HQ_scat = True)

for i in range(N_ave):
	event_gen.initialize()
	N1s_t.append([])
	Nc_t.append([])
	for j in range(N_step+1):
		N1s_t[i].append( len(event_gen.T1Slist['4-momentum']) )	# store N(t) for each event
		Nc_t[i].append( len(event_gen.Qlist['4-momentum']) )
		event_gen.run()
	# store the final T1S dictionary list
	len_T1s = len(event_gen.T1Slist['4-momentum'])
	for k in range(len_T1s):
		T1s_p4.append(event_gen.T1Slist['4-momentum'][k])
	# finally clear the lists for next event
	event_gen.Qlist.clear()
	event_gen.T1Slist.clear()


N1s_t = np.array(N1s_t)
N1s_t_ave = np.sum(N1s_t, axis = 0)/(N_ave + 0.0)	# averaged number of 1S state (time-sequenced)
Nc_t = np.array(Nc_t)
Nc_t_ave = np.sum(Nc_t, axis = 0)/(N_ave + 0.0)	# averaged number of 1S state (time-sequenced)
T1s_p4 = np.array(T1s_p4)

#### ------------ end of multiple runs averaged and compare ---------- ####



#### ------------ save the data in a h5py file ------------- ####

file1 = h5py.File('2760-0-10.hdf5', 'w')
file1.create_dataset('N1s', data = N1s_t_ave)
file1.create_dataset('Nc', data = Nc_t_ave)
file1.create_dataset('T1s_p4', data = T1s_p4)
file1.create_dataset('time', data = t)
file1.close()

#### ------------ end of saving the file ------------- ####


















