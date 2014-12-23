ARCH ?= 30
OPTIX ?= /opt/optix
NVCC ?= nvcc

.SUFFIXES: .ptx

all: ray_tracing_optix

ray_tracing_optix: ray_tracing_optix.o constantbg.ptx normal_shader.ptx pinhole_camera.ptx sphere.ptx
	$(NVCC) -arch=sm_$(ARCH) $< -o $@ -L$(OPTIX)/lib64 -loptix -Xlinker -rpath=$(OPTIX)/lib64 -L$(OPTIX)/SDK-precompiled-samples -lsutil -Xlinker -rpath=$(OPTIX)/SDK-precompiled-samples

constantbg.ptx: constantbg.cu
	$(NVCC) -I$(OPTIX)/include -O3 -arch=sm_$(ARCH) --use_fast_math -ptx -c $< -o $@

normal_shader.ptx: normal_shader.cu
	$(NVCC) -I$(OPTIX)/include -O3 -arch=sm_$(ARCH) --use_fast_math -ptx -c $< -o $@

pinhole_camera.ptx: pinhole_camera.cu
	$(NVCC) -I$(OPTIX)/include -O3 -arch=sm_$(ARCH) --use_fast_math -ptx -c $< -o $@

sphere.ptx: sphere.cu
	$(NVCC) -I$(OPTIX)/include -O3 -arch=sm_$(ARCH) --use_fast_math -ptx -c $< -o $@

ray_tracing_optix.o: ray_tracing_optix.cpp
	$(NVCC) -I$(OPTIX)/include -O3 -arch=sm_$(ARCH) -Xcompiler -ffast-math -Xcompiler -mtune=native -I$(OPTIX)/SDK/sutil -c $< -o $@

clean:
	rm -rf ray_tracing_optix *.o *.ptx

test: ray_tracing_optix
	./$< --file out.ppm
