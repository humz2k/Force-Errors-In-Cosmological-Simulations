# ctypes_test.py
import ctypes
import pathlib
import numpy as np
import pandas as pd
import time
from math import ceil
import os

package_directory = os.path.dirname(os.path.abspath(__file__))

solver_lib = package_directory + "/static_solver.dll"
solver = ctypes.CDLL(solver_lib)

def evaluate(eval_df,particle_df,G=1,eps=0,precision="f4"):

    n_particles = len(particle_df.index)
    n_evals = len(eval_df.index)

    pointer = ctypes.POINTER(ctypes.c_ulonglong)

    save_time = ctypes.c_ulonglong(0)

    saveTimePtr = pointer(save_time)

    if precision == "f2":
        eval_pos = eval_df.loc[:,["x","y","z"]].to_numpy().flatten().astype(np.float16)
        part_pos = particle_df.loc[:,["x","y","z"]].to_numpy().flatten().astype(np.float16)
        part_mass = particle_df.loc[:,"mass"].to_numpy().flatten().astype(np.float16)
        output = np.zeros((n_evals),dtype=np.float32)

        eval_pos_ptr = eval_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        part_pos_ptr = part_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        part_mass_ptr = part_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        solver.half_precision(eval_pos_ptr,part_pos_ptr,part_mass_ptr,output_ptr,ctypes.c_int(n_evals),ctypes.c_int(n_particles),ctypes.c_float(G),ctypes.c_float(eps**2),saveTimePtr)

        return output,save_time.value

    elif precision == "f4":
        
        eval_pos = eval_df.loc[:,["x","y","z"]].to_numpy().flatten().astype(np.float32)
        part_pos = particle_df.loc[:,["x","y","z"]].to_numpy().flatten().astype(np.float32)
        part_mass = particle_df.loc[:,"mass"].to_numpy().flatten().astype(np.float32)
        output = np.zeros((n_evals),dtype=np.float32)

        eval_pos_ptr = eval_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        part_pos_ptr = part_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        part_mass_ptr = part_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        solver.single_precision(eval_pos_ptr,part_pos_ptr,part_mass_ptr,output_ptr,ctypes.c_int(n_evals),ctypes.c_int(n_particles),ctypes.c_float(G),ctypes.c_float(eps**2),saveTimePtr)

        return output,save_time.value
    
    elif precision == "f8":
        
        eval_pos = eval_df.loc[:,["x","y","z"]].to_numpy().flatten().astype(np.float64)
        part_pos = particle_df.loc[:,["x","y","z"]].to_numpy().flatten().astype(np.float64)
        part_mass = particle_df.loc[:,"mass"].to_numpy().flatten().astype(np.float64)
        output = np.zeros((n_evals),dtype=np.float64)

        eval_pos_ptr = eval_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        part_pos_ptr = part_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        part_mass_ptr = part_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        solver.double_precision(eval_pos_ptr,part_pos_ptr,part_mass_ptr,output_ptr,ctypes.c_int(n_evals),ctypes.c_int(n_particles),ctypes.c_double(G),ctypes.c_double(eps**2),saveTimePtr)

        return output,save_time.value
