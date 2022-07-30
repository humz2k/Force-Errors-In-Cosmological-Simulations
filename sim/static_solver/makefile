static:
	nvcc static_solver.cu kernels/*.cu -lineinfo -Xptxas -v -Xcompiler="-fPIC" -arch=sm_53 -shared -o static_solver.dll