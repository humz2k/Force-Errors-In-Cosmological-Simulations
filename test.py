import sim
import scipy.spatial

df = sim.halos.Sample.Uniform(6)

#a = df.loc[:,["x","y","z"]].to_numpy()
#print(a)
#a = a.T.reshape((a.shape[1], a.shape[0]//2) + (2,)).transpose((1,0,2)).flatten()
#print(a)

n_particles = 6

#a = sim.distances.evaluate(df,df,precision="f8")

#c = sim.distances.evaluate(df,df,precision="f4")

d = sim.distances.evaluate(df,df,precision="f2")
for i in d:
    print(i)

#print(d)

particles = df.loc[:,["x","y","z"]].to_numpy()

b = scipy.spatial.distance.cdist(particles,particles)

#print(d)
print(b)
#print(b)

#print(a)
#print(d)