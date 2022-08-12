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

    pad = ((n_particles + 255)//256) * 256 - n_particles
    print("PAD",pad)
    
    vel_pad = np.zeros((pad * 3)).astype(np.float32)
    pos_pad = np.repeat(np.inf,pad*3).astype(np.float16).astype(np.float32)

    pointer = ctypes.POINTER(ctypes.c_ulonglong)

    save_time = ctypes.c_ulonglong(0)

    saveTimePtr = pointer(save_time)

    if precision == "f2":
        
        part_pos = np.concatenate((particle_df.loc[:,["x","y","z"]].to_numpy().flatten().astype(np.float32),pos_pad))
        part_vel = np.concatenate((particle_df.loc[:,["vx","vy","vz"]].to_numpy().flatten().astype(np.float32),vel_pad))
        part_mass = particle_df.iloc[0]["mass"]

        eps_array = np.repeat(eps**2,2).astype(np.float16)

        eps_pointer = eps_array.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

        part_pos_ptr = part_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        part_vel_ptr = part_vel.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        solver.half_precision(part_pos_ptr,part_vel_ptr,ctypes.c_float(part_mass),ctypes.c_int(n_particles + pad),ctypes.c_int(steps),ctypes.c_float(G),eps_pointer,ctypes.c_float(dt),saveTimePtr)

        raw_data = np.fromfile("out.dat",dtype=np.float32,sep="")
        #print(raw_data)
        raw_data = raw_data[:n_particles*n_params]

    step_labels = np.repeat(np.arange(steps+1),n_particles)
    ids = np.repeat(np.reshape(np.arange(n_particles),(n_particles,1)),(steps+1),axis=1).flatten(order="F")

    step_labels = pd.DataFrame(step_labels,columns=["step"],dtype=int)
    ids = pd.DataFrame(ids,columns=["id"],dtype=int)
    data = pd.DataFrame(raw_data.reshape((steps+1)*n_particles,n_params),columns=["x","y","z","vx","vy","vz","ax","ay","az","gpe"])

    out_df = pd.concat([step_labels,ids,data],axis=1)

    os.remove("out.dat")

    return out_df,save_time.value
