import sim
import numpy as np

df = sim.halos.Sample.Plummer(10000)

out2,time2 = sim.static_solver.evaluate(df,df,precision="f2-smcuda")

print(out2,time2)

out,time = sim.static_solver.evaluate(df,df,precision="f2")

print(out,time)

print(np.mean((out-out2)))

