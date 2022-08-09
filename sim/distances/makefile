static:
	nvcc distances.cu kernels/*.cu -lineinfo -Xptxas -v -Xcompiler="-fPIC" -arch=sm_53 -shared -o distances.dll