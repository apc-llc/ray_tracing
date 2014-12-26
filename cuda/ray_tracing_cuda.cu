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

	float x_pos = 0.9f;
	float y_pos = BOX_SIZE / 5.0;

	for (int i = 0; i < n_spheres; i++)
	{
		spheres[i].center.x = x_pos;
		spheres[i].center.y = y_pos;

		x_pos += BOX_SIZE / (n_spheres / 2.0);

		if  ( x_pos > BOX_SIZE - 0.9)
		{
			x_pos = 0.9f;
			y_pos = BOX_SIZE / 2.5 ;
		}
	}
	
	int j = 0;
	for (int i = 0; i < n_spheres; i++)
	{
		spheres[i].center.x += 2.0 * (hostData[j++] - 0.5);
		spheres[i].center.y += 2.0 * (hostData[j++] - 0.5);
		spheres[i].center.z = hostData[j++] * BOX_SIZE_Z + DISTANCE;
		spheres[i].radius = hostData[j++] * RADIUS_MAX + RADIUS_MIN;
		spheres[i].red   = hostData[j++] / (DEPTH_MAX - 3);
		spheres[i].green = hostData[j++] / (DEPTH_MAX - 3);
		spheres[i].blue  = hostData[j++] / (DEPTH_MAX - 3);
	}

	for (int i = 0; i < n_lights; i++)
	{
		lights[i].x = (hostData[j++] - 0.5) * BOX_SIZE * 6;
		lights[i].y = (hostData[j++] - 0.5) * BOX_SIZE * 6;
		lights[i].z = hostData[j++] * DISTANCE/2.0;
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
	int height, int width, int n_spheres, int n_lights, char** values)
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

	if (!values)
		generate_scene(spheres, n_spheres, lights, n_lights);
	else
	{
		// Parse scene from the command line.
		char** value = values;
		value++; // skip n_spheres
		for (int i = 0; i < n_spheres; i++)
		{
			spheres[i].center.x = atof(*(value++));
			spheres[i].center.y = atof(*(value++));
			spheres[i].center.z = atof(*(value++));
			spheres[i].radius = atof(*(value++));
			spheres[i].red = atof(*(value++));
			spheres[i].green = atof(*(value++));
			spheres[i].blue = atof(*(value++));
		}
		value++; // skip n_lights
		for (int i = 0; i < n_lights; i++)
		{
			lights[i].x = atof(*(value++));
			lights[i].y = atof(*(value++));
			lights[i].z = atof(*(value++));
		}
	}

#ifdef DEBUG
	print_spheres(spheres, n_spheres);
	print_lights(lights, n_lights);
#endif

	t_sphere * dev_spheres;
	t_light * dev_lights;

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

	free (spheres);
	free (lights);
}

int main( int argc, char* argv[] )
{
	bool randomScene = false;
	if (argc > 1)
		if ((std::string)argv[1] == "random")
			randomScene = true;

	if ((randomScene && (argc != 7)) || (argc == 1))
	{
		printf("Usage: %s random <n_spheres> <n_lights> <width> <height> <bmp_filename> \n", argv[0]);
		return -1;
	}

	int n_spheres, n_lights, width, height;
	char* filename;
	if (randomScene)
	{
		n_spheres = atoi(argv[2]);
		n_lights = atoi(argv[3]);
		width = atoi(argv[4]);
		height = atoi(argv[5]);
		filename = argv[6];
	}
	else
	{
		char** arg = &argv[1];
		bool failed = true;
		do
		{
			if ((size_t)arg - (size_t)argv >= sizeof(char*) * argc) break;
			n_spheres = atoi(*(arg++));
			arg += n_spheres * 7;
			if ((size_t)arg - (size_t)argv >= sizeof(char*) * argc) break;
			n_lights = atoi(*(arg++));
			arg += n_lights * 3;
			if ((size_t)arg - (size_t)argv >= sizeof(char*) * argc) break;
			width = atoi(*(arg++));
			if ((size_t)arg - (size_t)argv >= sizeof(char*) * argc) break;
			height = atoi(*(arg++));
			if ((size_t)arg - (size_t)argv >= sizeof(char*) * argc) break;
			filename = *arg;
			failed = false;
		}
		while (0);
		if (failed)
		{
			fprintf(stderr, "Parsing failed: not enough arguments\n");
			exit(1);
		}
	}

#ifdef ENABLE_CHECK
	if (n_spheres < 5 || n_spheres > 10)
	{
		printf ("n_spheres is out of range [5:10]\n");
		return -1;
	}
#endif

#ifdef ENABLE_CHECK
	if (n_lights < 1 || n_lights > 2)
	{
		printf ("n_lights is out of range [1:2]\n");
		return -1;
	}
#endif

#ifdef ENABLE_CHECK
	if (width < 800 || width > 1920)
	{
		printf ("width is out of range [800:1920]\n");
		return -1;
	}
#endif

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

	cudaEvent_t start = 0, stop = 0;
	CUDA_CALL (cudaEventCreate (&start) );
	CUDA_CALL (cudaEventCreate (&stop) );
	CUDA_CALL( cudaEventRecord (start, 0) );

	unsigned char * pR = (unsigned char *) malloc( height*width );
	unsigned char * pG = (unsigned char *) malloc( height*width );
	unsigned char * pB = (unsigned char *) malloc( height*width );

	if ( pR == NULL || pG == NULL || pB == NULL)
	{
		fprintf(stderr, "Malloc error, exiting\n");
		return -1;
	}

	// Pass down spheres/lights, if specific scene is given in command line.
	char** values = NULL;
	if (!randomScene) values = &argv[1];
	ray_trace(pR, pG, pB, height, width, n_spheres, n_lights, values);

	CUDA_CALL( cudaEventRecord (stop, 0) );
	CUDA_CALL( cudaEventSynchronize(stop) );

	float gpuTime = 0.0f;
	CUDA_CALL( cudaEventElapsedTime (&gpuTime, start, stop) );

	printf("CUDA ray tracing time: %.2f milliseconds\n", gpuTime);

	CUDA_CALL( cudaEventDestroy (start) );
	CUDA_CALL( cudaEventDestroy (stop) );

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
	AnImage.WriteToFile(filename);

	free(pR);
	free(pG);
	free(pB);
	
	return 0;
}

