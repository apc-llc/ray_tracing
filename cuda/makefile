ARCH ?= 30
BLOCK_SIZE_X = 32
BLOCK_SIZE_Y = 16

NVCC = nvcc -O3 --use_fast_math -Xcompiler -ffast-math -Xcompiler -mtune=native -arch=sm_$(ARCH) -DBLOCK_SIZE_X=$(BLOCK_SIZE_X) -DBLOCK_SIZE_Y=$(BLOCK_SIZE_Y)

NUM_SPHERES = 10
NUM_LIGHTS = 2
WIDTH = 1920
HEIGHT = 1080
FILE_NAME = output.bmp

SOURCES = debug.cu lib/EasyBMP.cpp main.cu  ray_tracing.cu kernel.cu
OBJECTS = $(addsuffix .o, $(basename $(SOURCES)))

all: ray_tracing_cuda

ray_tracing_cuda: $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $@ -lcuda -lcurand

%.o: %.cu
	$(NVCC) -c $< -o $@ 
	
test: ray_tracing_cuda
	./$< $(NUM_SPHERES) $(NUM_LIGHTS) $(WIDTH) $(HEIGHT) $(FILE_NAME)

clean:
	rm -rf ray_tracing_cuda $(OBJECTS) $(FILE_NAME)

