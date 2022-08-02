# ctypes_test.py
import ctypes
import numpy as np
import pandas as pd
from math import ceil
import os

package_directory = os.path.dirname(os.path.abspath(__file__))

solver_lib = package_directory + "/time_integrator.dll"
solver = ctypes.CDLL(solver_lib)

def evaluate(particle_df,G=1,eps=0,steps=0,dt=1/64,precision="f4",n_params=10):

    n_particles = len(particle_df.index)

    pointer = ctypes.POINTER(ctypes.c_ulonglong)

    save_time = ctypes.c_ulonglong(0)

    saveTimePtr = pointer(save_time)

    if precision == "f4":
        
        part_pos = particle_df.loc[:,["x","y","z"]].to_numpy().flatten().astype(np.float32)
        part_vel = particle_df.loc[:,["vx","vy","vz"]].to_numpy().flatten().astype(np.float32)
        part_mass = particle_df.loc[:,"mass"].to_numpy().flatten().astype(np.float32)

        part_pos_ptr = part_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        part_vel_ptr = part_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        part_mass_ptr = part_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        solver.single_precision(part_pos_ptr,part_vel_ptr,part_mass_ptr,ctypes.c_int(n_particles),ctypes.c_int(steps),ctypes.c_float(G),ctypes.c_float(eps**2),ctypes.c_float(dt),saveTimePtr)

        raw_data = np.fromfile("out.dat",dtype=np.float32,sep="")
    
    step_labels = np.repeat(np.arange(steps+1),n_particles)
    ids = np.repeat(np.reshape(np.arange(n_particles),(n_particles,1)),(steps+1),axis=1).flatten(order="F")

    step_labels = pd.DataFrame(step_labels,columns=["step"],dtype=int)
    ids = pd.DataFrame(ids,columns=["id"],dtype=int)
    data = pd.DataFrame(raw_data.reshape((steps+1)*n_particles,n_params),columns=["x","y","z","vx","vy","vz","ax","ay","az","gpe"])

    out_df = pd.concat([step_labels,ids,data],axis=1)

    os.remove("out.dat")

    return out_df,save_time.value
