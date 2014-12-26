include ../makefile.in

NVCC := $(NVCC) -I.. -I../bmp -O3 --use_fast_math -Xcompiler -ffast-math -Xcompiler -mtune=native -arch=sm_$(ARCH) -DBLOCK_SIZE_X=$(BLOCK_SIZE_X) -DBLOCK_SIZE_Y=$(BLOCK_SIZE_Y)

SOURCES = debug.cu ../bmp/EasyBMP.cpp ray_tracing_cuda.cu kernel.cu
OBJECTS = $(addsuffix .o, $(basename $(SOURCES)))

all: ray_tracing_cuda

ray_tracing_cuda: $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $@ -lcurand

%.o: %.cu
	$(NVCC) -c $< -o $@ 
	
test: ray_tracing_cuda
	./$< $(NUM_SPHERES) $(NUM_LIGHTS) $(WIDTH) $(HEIGHT) $(FILE_NAME)

clean:
	rm -rf ray_tracing_cuda $(OBJECTS) $(FILE_NAME)

