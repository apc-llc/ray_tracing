#ifndef KERNEL_GPU_H
#define KERNEL_GPU_H

#define DISTANCE 25.0

#define BOX_SIZE 20.0

#define DEPTH_MAX 5
#define SPHERES_MAX 10
#define LIGHTS_MAX 2

#define RGB_MAX  255
#define RADIUS_MAX 2.0
#define RADIUS_MIN 2.0

#define FORCE_ALIGNING

typedef struct
{
	float x;
	float y;
	float z;
#ifdef FORCE_ALIGNING
	float dummy;
#endif
}
t_vector;

typedef struct
{
	int i;
	int j;
}
t_pixel;

typedef struct
{
	t_vector normal;
	t_vector point;
	float  lambda_in;
#ifdef FORCE_ALIGNING
	float dummy1;
	float dummy2;
	float dummy3;
#endif
}
t_sphere_intersection;

typedef struct
{
	t_vector center;
	float radius;
	float red;
	float green;
	float blue;
}
t_sphere;

typedef struct
{
	t_vector origin;
	t_vector direction;
}
t_ray;

typedef struct
{
	float red;
	float green;
	float blue;
#ifdef FORCE_ALIGNING
	float dummy;
#endif
}
t_color;

#define t_light t_vector

__global__ void kernel( unsigned char * dev_image_red, 
			unsigned char * dev_image_blue, 
			unsigned char * dev_image_green, 
			int  height, int width, 
			t_sphere * spheres, int n_spheres, 
			t_light * lights, int n_lights);

#endif // KERNEL_GPU_H

