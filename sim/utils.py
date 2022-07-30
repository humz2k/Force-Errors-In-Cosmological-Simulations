import numpy as np
import pandas as pd
from scipy import spatial

def angles2vectors(alphas,betas):
    """Converts 2 floats or arrays of angles to (a) normalized 3D vector(s).

    A function that takes in arrays of 2 angles and returns normalized 3D vectors.

    Parameters
    ----------
    alphas : array_like
        The alpha angles.
    betas : array_like
        The beta angles.
    
    Returns
    -------
    numpy.ndarray
        An array of normalized 3D vectors, with shape (alphas.shape[0],3).

    """
    x = np.cos(alphas) * np.cos(betas)
    z = np.sin(alphas) * np.cos(betas)
    y = np.sin(betas)
    return np.column_stack([x,y,z])

def randangles(size=10):
    """Generates 2 arrays of random angles.

    A function that takes in a size and returns 2 arrays of random angles.

    Parameters
    ----------
    size : int
        The number of angles to generate. If unspecified, size defaults to 10.
    
    Returns
    -------
    tuple
        Two arrays of shape (size,) of random angles from 0,2*pi.

    """
    return np.random.uniform(0,2*np.pi,size=size),np.random.uniform(0,2*np.pi,size=size)

def random_vectors(size=1):
    """Generates a random array of normalized 3D vectors.

    A function that takes in a size and returns 2 arrays of random angles.

    Parameters
    ----------
    size : int
        The number of angles to generate. If unspecified, the size defaults to 1.
    
    Returns
    -------
    numpy.ndarray
        An array of random normalized 3D vectors with shape (size,3).

    """
    return angles2vectors(*randangles(size))

def ray(vector,length,nsteps,file=None):
    """Generates a ray from a vector, length and number of steps.

    A function that takes in a vector, a length, a number of steps, and returns a DataFrame of points along the vector.

    Parameters
    ----------
    vector : numpy.ndarray
        The vector of the ray, with shape (3,).
    length : float
        The length of the ray.
    nsteps : int
        The number of steps for the returned points along the vector.
    file : str
        The filename to save the resulting DataFrame to (.csv). If unspecified, the DataFrame will not be saved.
    
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame of nsteps 3D points along a ray pointing in direction vector with length length.

    """
    vector = np.reshape(vector/np.linalg.norm(vector),(1,) + vector.shape)
    rs = np.reshape(np.linspace(0,length,nsteps),(1,nsteps)).T
    points = rs * vector
    df = pd.DataFrame(points,columns=["x","y","z"])
    if file != None:
        df.to_csv(file,index=False)
    return df

def ray_rs(length,nsteps):
    return np.linspace(0,length,nsteps)

def points2radius(points):
    """Converts 3D points to radiuses from [0,0,0]

    A function that takes in a DataFrame of 3D points, and converts them to radiuses from the point [0,0,0]

    Parameters
    ----------
    points : pandas.DataFrame
        The DataFrame of points (should have "x","y" and "z" columns).
    
    Returns
    -------
    numpy.ndarray
        An array of radiuses with shape equal to the number of points in the input DataFrame

    """
    points = points.loc[:,["x","y","z"]].to_numpy()
    return spatial.distance.cdist(np.array([[0,0,0]]),points).flatten()

def outdf2numpy(df):
    """Converts an output DF from an evaluation, and converts it to numpy arrays.

    A function that takes in a DataFrame with columns ["x","y","z","vx","vy","vz","ax","ay","az","gpe"], and converts each to numpy arrays with the shape (nsteps,nparticles,3 (or 1 for GPE)).

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame outputted from an evaluation.
    
    Returns
    -------
    dict
        Dictionary containing the arrays for "pos" (the positions from the output), "vel" (the velocities from the output), "acc" (the accelerations form the output) and "gpe" (the GPEs from the output)

    """
    steps = np.unique(df.loc[:,"step"].to_numpy())
    nsteps = steps.shape[0]
    ids = np.unique(df.loc[:,"id"].to_numpy())
    nparticles = ids.shape[0]
    pos = df.loc[:,["x","y","z"]].to_numpy()
    pos = pos.reshape(nsteps,nparticles,3)
    vel = df.loc[:,["vx","vy","vz"]].to_numpy()
    vel = vel.reshape(nsteps,nparticles,3)
    acc = df.loc[:,["ax","ay","az"]].to_numpy()
    acc = acc.reshape(nsteps,nparticles,3)
    gpe = df.loc[:,["gpe"]].to_numpy()
    gpe = gpe.reshape(nsteps,nparticles,1)
    return {"pos":pos,"vel":vel,"acc":acc,"gpe":gpe}