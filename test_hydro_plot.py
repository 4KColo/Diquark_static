#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si
import h5py
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif'})
rc('text', usetex=True)

import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
from mpl_toolkits.mplot3d import Axes3D

#### -------- open file and read the time and percentage ------ ####

f1 = h5py.File('HydroData0-10.h5', 'r')
step_keys = list(f1['Event'].keys())	# at every tau step, tau^2=t^2-z^2
Nstep = len(step_keys)
Nx = f1['Event'].attrs['XH'][0] - f1['Event'].attrs['XL'][0] + 1
Ny = f1['Event'].attrs['YH'][0] - f1['Event'].attrs['YL'][0] + 1
dx = f1['Event'].attrs['DX'][0]
dy = f1['Event'].attrs['DY'][0]
dtau = f1['Event'].attrs['dTau'][0]
tau0 = f1['Event'].attrs['Tau0'][0]
tauf = tau0 + (Nstep-1.) * dtau
xmin = f1['Event'].attrs['XL'][0]*dx
xmax = f1['Event'].attrs['XH'][0]*dx
ymin = f1['Event'].attrs['YL'][0]*dy
ymax = f1['Event'].attrs['YH'][0]*dy

T_xy = f1['Event'][step_keys[50]]['Temp'].value
f1.close()
tau = np.linspace(tau0, tauf, Nstep)
print tau
x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)
x_grid, y_grid = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, T_xy, color='b')
plt.show()	
