#include <cstdio>
#include "debug.h"
#include "kernel.h"

void print_array(unsigned char* p, int h, int w)
{
	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < h; j++)
			printf("%02X ", p[j + i * w]);
		printf("\n");
	}
}

void print_array_float(float* p, int h, int w)
{
	int start = 0;
	int finish = start + 15;

	if (w > finish)
		w = finish;
	if (h > finish)
		h = finish;

	for (int i = start; i < w; i++)
	{
		for (int j = start; j < h; j++)
			printf("%08.3f ", p[j + i * w]);
		printf("\n");
	}
	printf("\n");
}

void print_spheres(t_sphere* spheres, int n_spheres)
{
	for (int i = 0; i < n_spheres; i++)
	{
		printf("sphere %d\n", i);
		printf("\t x=%2.3f y=%2.3f z=%2.3f r=%2.3f \n", 
			spheres[i].center.x, spheres[i].center.y, 
			spheres[i].center.z, spheres[i].radius);
		printf("\t red=%2.3f green=%2.3f blue=%2.3f  \n", 
			spheres[i].red, spheres[i].green, spheres[i].blue);
	}
	printf ("\n");
}

void print_lights(t_light* lights, int n_lights)
{
	for (int i = 0; i < n_lights; i++)
	{
		printf("light %d\n", i);
		printf("\t x=%2.3f y=%2.3f z=%2.3f \n", 
			lights[i].x, lights[i].y, lights[i].z);
	}
	printf("\n");
}

