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

def cudaDist(eval_df,particle_df,precision="f8"):

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
        if n_evals % 2 != 0:
            eval_pos = np.concatenate((eval_pos,np.array([[0,0,0]])))
            n_evals += 1
        
        eval_pos = eval_pos.T.reshape((eval_pos.shape[1], eval_pos.shape[0]//2) + (2,)).transpose((1,0,2)).flatten().astype(np.float16)

        part_pos = particle_df.loc[:,["x","y","z"]].to_numpy()
        if n_particles % 2 != 0:
            part_pos = np.concatenate((part_pos,np.array([[0,0,0]])))
            n_particles += 1
        part_pos = part_pos.T.reshape((part_pos.shape[1], part_pos.shape[0]//2) + (2,)).transpose((1,0,2)).flatten().astype(np.float16)

        output = np.zeros((n_evals * n_particles),dtype=np.float16)

        eval_pos_ptr = eval_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        part_pos_ptr = part_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

        distances.half_precision(eval_pos_ptr,part_pos_ptr,output_ptr,ctypes.c_int(n_evals),ctypes.c_int(n_particles))

        evens = output[::2]
        odds = output[1::2]
        odds = np.flip(odds.reshape(odds.shape[0]//2,2),axis=1).flatten()

        odds = odds.reshape((n_evals//2,n_particles))
        evens = evens.reshape((n_evals//2,n_particles))
        out = np.concatenate([evens,odds],axis=1).flatten().reshape((n_evals,n_particles))

        return out

def listPhis(eval_df,particle_df,precision="f2",G=1,eps=0):

    n_particles = len(particle_df.index)
    n_evals = len(eval_df.index)
    
    if precision == "f2":
        
        eval_pos = eval_df.loc[:,["x","y","z"]].to_numpy()
        if n_evals % 2 != 0:
            eval_pos = np.concatenate((eval_pos,np.array([[0,0,0]])))
            n_evals += 1
        eval_pos = eval_pos.T.reshape((eval_pos.shape[1], eval_pos.shape[0]//2) + (2,)).transpose((1,0,2)).flatten().astype(np.float16)

        part_pos = particle_df.loc[:,["x","y","z"]].to_numpy()
        if n_particles % 2 != 0:
            part_pos = np.concatenate((part_pos,np.array([[0,0,0]])))
            n_particles += 1
        part_pos = part_pos.T.reshape((part_pos.shape[1], part_pos.shape[0]//2) + (2,)).transpose((1,0,2)).flatten().astype(np.float16)

        eval_mass = particle_df.loc[:,["mass"]].to_numpy()
        if eval_mass.flatten().shape[0] % 2 != 0:
            eval_mass = np.concatenate((eval_mass,np.array([[0]])))
        eval_mass = eval_mass.T.reshape((eval_mass.shape[1], eval_mass.shape[0]//2) + (2,)).transpose((1,0,2)).flatten().astype(np.float16)
        
        output = np.zeros((n_evals * n_particles),dtype=np.float16)

        G_array = np.repeat(-G,2).astype(np.float16)
        eps_array = np.repeat(eps**2,2).astype(np.float16)

        G_ptr = G_array.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        eps_ptr = eps_array.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

        eval_pos_ptr = eval_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        part_pos_ptr = part_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        eval_mass_ptr = eval_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

        distances.half_precision_phis(eval_pos_ptr,part_pos_ptr,eval_mass_ptr,G_ptr,eps_ptr,output_ptr,ctypes.c_int(n_evals),ctypes.c_int(n_particles))

        evens = output[::2]
        odds = output[1::2]
        odds = np.flip(odds.reshape(odds.shape[0]//2,2),axis=1).flatten()

        odds = odds.reshape((n_evals//2,n_particles))
        evens = evens.reshape((n_evals//2,n_particles))
        out = np.concatenate([evens,odds],axis=1).flatten().reshape((n_evals,n_particles))
        out[out == -np.inf] = 0
        return out
    
    elif precision == "f8":
        
        eval_pos = eval_df.loc[:,["x","y","z"]].to_numpy().flatten().astype(np.float64)
        part_pos = particle_df.loc[:,["x","y","z"]].to_numpy().flatten().astype(np.float64)
        part_masses = particle_df.loc[:,"mass"].to_numpy().flatten().astype(np.float64)

        output = np.zeros((n_evals,n_particles),dtype=np.float64)

        eval_pos_ptr = eval_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        part_pos_ptr = part_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        part_mass_ptr = part_masses.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        distances.double_precision_phis(eval_pos_ptr,part_pos_ptr,part_mass_ptr,ctypes.c_double(-G),ctypes.c_double(eps**2),output_ptr,ctypes.c_int(n_evals),ctypes.c_int(n_particles))

        return output
