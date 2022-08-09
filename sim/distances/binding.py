# ctypes_test.py
import ctypes
import pathlib
import numpy as np
import pandas as pd
import time
from math import ceil
import os

package_directory = os.path.dirname(os.path.abspath(__file__))

distances_lib = package_directory + "/distances.dll"
distances = ctypes.CDLL(distances_lib)

def evaluate(eval_df,particle_df,precision="f8"):

    n_particles = len(particle_df.index)
    n_evals = len(eval_df.index)
    
    if precision == "f8":
        
        eval_pos = eval_df.loc[:,["x","y","z"]].to_numpy().flatten().astype(np.float64)
        part_pos = particle_df.loc[:,["x","y","z"]].to_numpy().flatten().astype(np.float64)

        output = np.zeros((n_evals,n_particles),dtype=np.float64)

        eval_pos_ptr = eval_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        part_pos_ptr = part_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        distances.double_precision(eval_pos_ptr,part_pos_ptr,output_ptr,ctypes.c_int(n_evals),ctypes.c_int(n_particles))

        return output
    
    elif precision == "f4":
        
        eval_pos = eval_df.loc[:,["x","y","z"]].to_numpy().flatten().astype(np.float32)
        part_pos = particle_df.loc[:,["x","y","z"]].to_numpy().flatten().astype(np.float32)

        output = np.zeros((n_evals,n_particles),dtype=np.float32)

        eval_pos_ptr = eval_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        part_pos_ptr = part_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        distances.single_precision(eval_pos_ptr,part_pos_ptr,output_ptr,ctypes.c_int(n_evals),ctypes.c_int(n_particles))

        return output
    
    elif precision == "f2":
        
        eval_pos = eval_df.loc[:,["x","y","z"]].to_numpy()
        eval_pos = eval_pos.T.reshape((eval_pos.shape[1], eval_pos.shape[0]//2) + (2,)).transpose((1,0,2)).flatten().astype(np.float16)

        part_pos = particle_df.loc[:,["x","y","z"]].to_numpy()
        part_pos = part_pos.T.reshape((part_pos.shape[1], part_pos.shape[0]//2) + (2,)).transpose((1,0,2)).flatten().astype(np.float16)

        output = np.zeros((n_evals * n_particles),dtype=np.float16)

        eval_pos_ptr = eval_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        part_pos_ptr = part_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

        distances.half_precision(eval_pos_ptr,part_pos_ptr,output_ptr,ctypes.c_int(n_evals),ctypes.c_int(n_particles))

        return output
