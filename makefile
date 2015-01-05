include makefile.in

.PHONY: cuda optix

all: cuda optix

cuda:
	cd $@ && $(MAKE)

optix:
	cd $@ && $(MAKE)

clean:
	rm -rf *.o ray_tracing_compare $(FILE_NAME) && cd cuda && $(MAKE) clean && cd ../optix && $(MAKE) clean

random:
	cd cuda && $(MAKE) random && cd ../optix && $(MAKE) random

ray_tracing_compare: ray_tracing_compare.o EasyBMP.o
	$(NVCC) -arch=sm_$(ARCH) $^ -o $@ -lcurand

ray_tracing_compare.o: ray_tracing_compare.cu
	$(NVCC) -I./bmp -O3 -arch=sm_$(ARCH) -c $< -o $@

EasyBMP.o: ./bmp/EasyBMP.cpp
	g++ -O3 -c $< -o $@

compare: ray_tracing_compare cuda optix
	./$< $(NUM_SPHERES) $(NUM_LIGHTS) $(WIDTH) $(HEIGHT) $(FILE_NAME)
