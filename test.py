import sim

df = sim.halos.Sample.Plummer(1000)
print(df)
print(sim.time_integrator.evaluate(df,steps=1))