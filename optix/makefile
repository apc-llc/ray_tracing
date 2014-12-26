include ../makefile.in

.SUFFIXES: .ptx

all: ray_tracing_optix

OPT = -g -O3

ray_tracing_optix: ray_tracing_optix.o constantbg.ptx normal_shader.ptx pinhole_camera.ptx sphere.ptx EasyBMP.o
	$(NVCC) -arch=sm_$(ARCH) $(filter %.o, $^) -o $@ -L$(OPTIX)/lib64 -loptix -Xlinker -rpath=$(OPTIX)/lib64 -L$(OPTIX)/SDK-precompiled-samples -lsutil -Xlinker -rpath=$(OPTIX)/SDK-precompiled-samples -lcurand

constantbg.ptx: constantbg.cu
	$(NVCC) -I$(OPTIX)/include $(OPT) -arch=sm_$(ARCH) --use_fast_math -ptx -c $< -o $@

normal_shader.ptx: normal_shader.cu
	$(NVCC) -I$(OPTIX)/include $(OPT) -arch=sm_$(ARCH) --use_fast_math -ptx -c $< -o $@

pinhole_camera.ptx: pinhole_camera.cu
	$(NVCC) -I$(OPTIX)/include $(OPT) -arch=sm_$(ARCH) --use_fast_math -ptx -c $< -o $@

sphere.ptx: sphere.cu
	$(NVCC) -I$(OPTIX)/include $(OPT) -arch=sm_$(ARCH) --use_fast_math -ptx -c $< -o $@

ray_tracing_optix.o: ray_tracing_optix.cpp
	$(NVCC) -I.. -I../bmp -I$(OPTIX)/include $(OPT) -arch=sm_$(ARCH) -Xcompiler -ffast-math -Xcompiler -mtune=native -I$(OPTIX)/SDK/sutil -c $< -o $@

EasyBMP.o: ../bmp/EasyBMP.cpp
	g++ $(OPT) $< -c $<

test: ray_tracing_optix
	./$< $(NUM_SPHERES) $(NUM_LIGHTS) $(WIDTH) $(HEIGHT) $(FILE_NAME)

clean:
	rm -rf ray_tracing_optix *.o *.ptx $(FILE_NAME)

