import sim
import scipy.spatial
import numpy as np
import time

df = sim.halos.Sample.Uniform(5000)

ray = sim.utils.ray(np.array([1,0,0]),1,2)
print(ray)

mass = df.loc[:,"mass"][0]

#d = sim.distances.cudaDist(ray,df,precision="f8")
#print(d)


d = sim.distances.listPhis(ray,df,precision="f8")
print(d)

#d = sim.distances.listPhis(ray,df,precision="f2")
#print(d)


particles = df.loc[:,["x","y","z"]].to_numpy()
ray_pos = ray.loc[:,["x","y","z"]].to_numpy()
t1 = time.perf_counter()
b = scipy.spatial.distance.cdist(ray_pos,particles)
phis = -mass/b

t2 = time.perf_counter()
phi = -1*mass/b

#print((phi - d)/phi)