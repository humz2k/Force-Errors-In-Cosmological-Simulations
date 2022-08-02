time_integrator:
	nvcc time_integrator.cu kernels/*.cu -lineinfo -Xptxas -v -Xcompiler="-fPIC" -arch=sm_53 -shared -o time_integrator.dll