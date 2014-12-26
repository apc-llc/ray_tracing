include makefile.in

.PHONY: cuda optix

all: cuda optix

cuda:
	cd $@ && $(MAKE)

optix:
	cd $@ && $(MAKE)

clean:
	rm -rf ray_tracing_compare $(FILE_NAME) && cd cuda && $(MAKE) clean && cd ../optix && $(MAKE) clean

random:
	cd cuda && $(MAKE) random && cd ../optix && $(MAKE) random

ray_tracing_compare: ray_tracing_compare.cu
	$(NVCC) -O3 -arch=sm_$(ARCH) $< -o $@ -lcurand

compare: ray_tracing_compare cuda optix
	./$< $(NUM_SPHERES) $(NUM_LIGHTS) $(WIDTH) $(HEIGHT) $(FILE_NAME)
