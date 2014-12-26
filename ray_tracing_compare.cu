#include <cstdio>
#include <cstdlib>
#include <curand_kernel.h>
#include <iostream>
#include <string>
#include <sstream>

#include "types.h"

using namespace std;

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

int main(int argc, char** argv)
{
	if (argc != 6)
	{
		printf("Usage: %s <n_spheres> <n_lights> <width> <height> <bmp_filename> \n", argv[0]);
		return -1;
	}

	int n_spheres = atoi(argv[1]);
	int n_lights = atoi(argv[2]);
	int width = atoi(argv[3]);
	int height = atoi(argv[4]);
	char* filename = argv[5];

	// Generate random scene config and store it in the command line args
	t_sphere * spheres = (t_sphere *) malloc (sizeof(t_sphere) * n_spheres);
	t_light * lights = (t_light *) malloc (sizeof(t_light) * n_lights);

	if (lights == NULL || spheres == NULL)
	{
		fprintf(stderr, "Malloc error, exiting\n");
		exit(-1);
	}

	generate_scene(spheres, n_spheres, lights, n_lights);
	
	string args;
	{
		stringstream s;
		s << n_spheres << " ";
		for (int i = 0; i < n_spheres; i++)
		{
			s << spheres[i].center.x << " ";
			s << spheres[i].center.y << " ";
			s << spheres[i].center.z << " ";
			s << spheres[i].radius << " ";
			s << spheres[i].red << " ";
			s << spheres[i].green << " ";
			s << spheres[i].blue << " ";
		}
		s << n_lights << " ";
		for (int i = 0; i < n_lights; i++)
		{
			s << lights[i].x << " "; 
			s << lights[i].y << " "; 
			s << lights[i].z << " ";; 
		}
		s << width << " " << height << " " << filename;
		args = s.str();
	}

	// Launch CUDA ray tracer
	{
		stringstream s;
		s << "cd cuda && ./ray_tracing_cuda " << args;
		string cmd = s.str();
		//cout << cmd << endl;
		int result = system(cmd.c_str());
		if (result) return result;
	}

	// Launch OptiX ray tracer
	{
		stringstream s;
		s << "cd optix && ./ray_tracing_optix " << args;
		string cmd = s.str();
		//cout << cmd << endl;
		int result = system(cmd.c_str());
		if (result) return result;
	}	

	// Compare BMP outputs

	return 0;
}

