#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <curand_kernel.h>
#include <unistd.h>

#include "debug.h"
#include "types.h"
#include "EasyBMP.h"

#define ENABLE_CHECK

// Round a / b to nearest higher integer value
inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b)
{
    return (a % b != 0) ? (a - a % b + b) : a;
}

static void generate_scene(t_sphere * spheres, int n_spheres, t_light * lights, int n_lights)
{
	int n_random_coord = n_spheres * 3  + n_lights * 3;
	int n_random_rad = n_spheres;
	int n_random_colors = n_spheres * 3;

	size_t n = n_random_coord + n_random_rad + n_random_colors;

	curandGenerator_t gen;
	float *devData, *hostData;
	hostData = (float *)calloc(n, sizeof(float));

	if (!hostData)
	{
		fprintf(stderr, "Malloc error, exiting\n");
		exit(-1);
	}

	CUDA_CALL( cudaMalloc((void **)&devData, n*sizeof(float)) );

	CURAND_CALL( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
	CURAND_CALL( curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)time(NULL)) ); 

	CURAND_CALL( curandGenerateUniform(gen, devData, n) );
	CUDA_CALL( cudaMemcpy(hostData, devData, n * sizeof(float), cudaMemcpyDeviceToHost) );
	
	int j = 0;
	for (int i = 0; i < n_spheres; i++)
	{
		spheres[i].center.x = hostData[j++] * BOX_SIZE ;
		spheres[i].center.y = hostData[j++] * BOX_SIZE ;
		spheres[i].center.z = hostData[j++] * BOX_SIZE + DISTANCE ;
		spheres[i].radius = hostData[j++] * RADIUS_MAX + RADIUS_MIN;
		spheres[i].red   = hostData[j++] / (DEPTH_MAX - 3);
		spheres[i].green = hostData[j++] / (DEPTH_MAX - 3);
		spheres[i].blue  = hostData[j++] / (DEPTH_MAX - 3);
	}

	for (int i = 0; i < n_lights; i++)
	{
		lights[i].x = hostData[j++] * BOX_SIZE; 
		lights[i].y = hostData[j++] * BOX_SIZE; 
		lights[i].z = hostData[j++] * DISTANCE + BOX_SIZE / 2.0; 
	}

	CURAND_CALL( curandDestroyGenerator(gen) );
	CUDA_CALL( cudaFree(devData) );
	free(hostData);    
}

__global__ void kernel(
	unsigned char * dev_image_red, unsigned char * dev_image_blue,
	unsigned char * dev_image_green, int height, int width,
	t_sphere * spheres, int n_spheres, t_light * lights, int n_lights);

static void ray_trace(
	unsigned char * pR, unsigned char * pG, unsigned char * pB, 
	int height, int width, int n_spheres, int n_lights)
{
//#define STACK_INCREASE
#ifdef STACK_INCREASE 
	size_t stack=0;
	CUDA_CALL( cudaDeviceGetLimit(&stack, cudaLimitStackSize) ); 
	printf ("Cuda stack size %ld \n", stack);
	stack = 1536;
	printf ("Setting cuda stack size to %ld \n", stack);
	CUDA_CALL( cudaDeviceSetLimit(cudaLimitStackSize, stack) );
#endif

	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	t_sphere * spheres = (t_sphere *) malloc (sizeof(t_sphere) * n_spheres);
	t_light * lights = (t_light *) malloc (sizeof(t_light) * n_lights);

	if (lights == NULL || spheres == NULL)
	{
		fprintf(stderr, "Malloc error, exiting\n");
		exit(-1);
	}

	generate_scene(spheres, n_spheres, lights, n_lights);

#ifdef DEBUG
	print_spheres(spheres, n_spheres);
	print_lights(lights, n_lights);
#endif

	t_sphere * dev_spheres;
	t_light * dev_lights;

	cudaEvent_t start = 0, stop = 0;
	CUDA_CALL (cudaEventCreate (&start) );
	CUDA_CALL (cudaEventCreate (&stop) );
	CUDA_CALL( cudaEventRecord (start, 0) );

	CUDA_CALL( cudaMalloc((void **)&dev_spheres,  sizeof(t_sphere) * n_spheres ) );
	CUDA_CALL( cudaMalloc((void **)&dev_lights,  sizeof(t_light) * n_lights ) );

	CUDA_CALL( cudaMemcpy(dev_spheres, spheres, sizeof(t_sphere) * n_spheres, cudaMemcpyHostToDevice) );
	CUDA_CALL( cudaMemcpy(dev_lights, lights, sizeof(t_light) * n_lights, cudaMemcpyHostToDevice) );

	unsigned char * dev_image_red;
	unsigned char * dev_image_green;
	unsigned char * dev_image_blue;

	CUDA_CALL( cudaMalloc((void **)&dev_image_red,   height * width *sizeof(unsigned char)) );
	CUDA_CALL( cudaMalloc((void **)&dev_image_green, height * width *sizeof(unsigned char)) );
	CUDA_CALL( cudaMalloc((void **)&dev_image_blue,  height * width *sizeof(unsigned char)) );

	CUDA_CALL( cudaMemset(dev_image_red,   0, height * width *sizeof(unsigned char)) );
	CUDA_CALL( cudaMemset(dev_image_green, 0, height * width *sizeof(unsigned char)) );
	CUDA_CALL( cudaMemset(dev_image_blue,  0, height * width *sizeof(unsigned char)) );

	dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
	dim3 grid(iDivUp(width, block.x), iDivUp(height, block.y), 1);

#ifdef DEBUG
	printf ("Running kernel with block.x = %d block.y = %d \n", block.x, block.y);
	printf ("Running kernel with grid.x = %d grid.y = %d \n", grid.x, grid.y);
#endif

	kernel<<<grid,block>>>(dev_image_red, dev_image_blue, dev_image_green, 
			height, width, dev_spheres, n_spheres, dev_lights, n_lights); 
	CUDA_CALL( cudaGetLastError() );

	CUDA_CALL( cudaMemcpy(pR, dev_image_red,  height * width *sizeof(unsigned char), cudaMemcpyDeviceToHost) );
	CUDA_CALL( cudaMemcpy(pB, dev_image_blue, height * width *sizeof(unsigned char), cudaMemcpyDeviceToHost) );
	CUDA_CALL( cudaMemcpy(pG, dev_image_green,height * width *sizeof(unsigned char), cudaMemcpyDeviceToHost) );

	CUDA_CALL( cudaFree(dev_image_red) );
	CUDA_CALL( cudaFree(dev_image_green) );
	CUDA_CALL( cudaFree(dev_image_blue) );

	CUDA_CALL( cudaFree(dev_spheres) );
	CUDA_CALL( cudaFree(dev_lights) );

	CUDA_CALL( cudaEventRecord (stop, 0) );
	CUDA_CALL( cudaEventSynchronize(stop) );

	float gpuTime = 0.0f;
	CUDA_CALL( cudaEventElapsedTime (&gpuTime, start, stop) );

	printf("CUDA ray tracing time: %.2f milliseconds\n", gpuTime);

	CUDA_CALL( cudaEventDestroy (start) );
	CUDA_CALL( cudaEventDestroy (stop) );

	free (spheres);
	free (lights);
}

int main( int argc, char* argv[] )
{
	if (argc != 6)
	{
		printf("Usage: %s <n_spheres> <n_lights> <width> <height> <bmp_filename> \n", argv[0]);
		return -1; 
	}

	int n_spheres = atoi( argv[1]);
#ifdef ENABLE_CHECK
	if (n_spheres < 5 || n_spheres > 10)
	{
		printf ("n_spheres is out of range [5:10]\n");
		return -1;
	}
#endif

	int n_lights = atoi( argv[2]);
#ifdef ENABLE_CHECK
	if (n_lights < 1 || n_lights > 2)
	{
		printf ("n_lights is out of range [1:2]\n");
		return -1;
	}
#endif

	int width = atoi( argv[3]);
#ifdef ENABLE_CHECK
	if (width < 800 || width > 1920)
	{
		printf ("width is out of range [800:1920]\n");
		return -1;
	}
#endif

	int height = atoi( argv[4]);
#ifdef ENABLE_CHECK
	if (height < 600 || height > 1080)
	{
		printf ("height is out of range [600:1080]\n");
		return -1;
	}
#endif

#ifdef DEBUG
	printf ("Picture size is width = %d  height = %d \n", width, height);
#endif

	unsigned char * pR = (unsigned char *) malloc( height*width );
	unsigned char * pG = (unsigned char *) malloc( height*width );
	unsigned char * pB = (unsigned char *) malloc( height*width );

	if ( pR == NULL || pG == NULL || pB == NULL)
	{
		fprintf(stderr, "Malloc error, exiting\n");
		return -1;
	}

	ray_trace(pR, pG, pB, height, width, n_spheres, n_lights);

	BMP AnImage;
	AnImage.SetSize(width, height);
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			RGBApixel pixel;
			pixel.Red = pR [ j * width + i ];
			pixel.Green = pG [ j * width + i ];
			pixel.Blue = pB [ j * width + i ];
			AnImage.SetPixel( i , j , pixel );
		}
	}
	AnImage.WriteToFile(argv[5]);

	free(pR);
	free(pG);
	free(pB);
	
	return 0;
}

