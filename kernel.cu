#include "kernel.h"
#include <float.h>
#include <stdio.h>


#define X_DIM BOX_SIZE

__device__ __shared__ t_vector view_point;
__device__ __shared__ float y_dim;

__device__ __shared__ int n_spheres; 
__device__ __shared__ int n_lights;
__device__ __shared__ t_sphere spheres[ SPHERES_MAX ];
__device__ __shared__ t_light  lights[ LIGHTS_MAX  ];


#define EPSILON 1e-2
__device__  bool fequal(float a, float b)
{
	return fabs( __fadd_rn(a, -b) ) < EPSILON;
}

__device__ void vec_sub (t_vector *v1, t_vector *v2, t_vector *v3) {

	v1->x = __fadd_rn( v2->x, -v3->x);
	v1->y = __fadd_rn( v2->y, -v3->y);
	v1->z = __fadd_rn( v2->z, -v3->z);
}

__device__ void vec_add (t_vector *v1, t_vector *v2, t_vector *v3) {
	
	v1->x = __fadd_rn (v2->x, v3->x);
	v1->y = __fadd_rn (v2->y, v3->y);
	v1->z = __fadd_rn (v2->z, v3->z);
}

__device__ void vec_scale (float scale, t_vector *v1, t_vector *v2) {
	
	v1->x = __fmul_rn (scale, v2->x); // multiplying
	v1->y = __fmul_rn (scale, v2->y);
	v1->z = __fmul_rn (scale, v2->z);
}

__device__ float dotproduct (t_vector *v1, t_vector *v2) {
	
	return 
		__fadd_rn(
		 	__fmul_rn (v1->x, v2->x), 
			__fadd_rn ( __fmul_rn (v1->y, v2->y),  __fmul_rn (v1->z, v2->z))
		 );
}


__device__ void normalize_vector (t_vector *v) {
	
	float magnitude;
	
	magnitude = __fsqrt_rn ( dotproduct(v, v) );
	v->x = __fdiv_rn (v->x, magnitude);
	v->y = __fdiv_rn (v->y, magnitude);
	v->z = __fdiv_rn (v->z, magnitude);
}

__device__ void compute_ray(t_ray* ray, t_vector* view_point,
			 t_pixel* pixel) 
{
	ray->origin = *view_point;

	ray->direction.x = 
		__fdiv_rn (__fmul_rn (X_DIM, pixel->i), 
			__mul24(blockDim.x, gridDim.x)) - __fdiv_rn (X_DIM, 2.0) ;

	ray->direction.y = 
		__fdiv_rn (__fmul_rn (y_dim, pixel->j), 
			__mul24(blockDim.y, gridDim.y)) - __fdiv_rn (y_dim, 2.0) ;

	ray->direction.z = (float) DISTANCE;

	normalize_vector(&ray->direction);
}


__device__ void compute_reflected_ray(t_ray* reflected_ray, t_ray* incidence_ray, 
		t_sphere_intersection* intersection) 
{
	
	float dp1;
	t_vector scaled_normal;
	reflected_ray->origin=intersection->point;

	dp1 = dotproduct(&intersection->normal, &incidence_ray->direction);
	dp1 = __fmul_rn (2, dp1);

	vec_scale(dp1, &scaled_normal, &intersection->normal);
	
	vec_sub(&reflected_ray->direction, &incidence_ray->direction, &scaled_normal);
}


__device__ void compute_ray_to_light(t_ray* ray, 
		t_sphere_intersection* intersection, t_vector* light)
{
	ray->origin = intersection->point;
        vec_sub(&ray->direction, light, &intersection->point);
        normalize_vector(&ray->direction);
}


__device__ bool sphere_intersection (t_ray *ray, t_sphere *sphere, 
		t_sphere_intersection* intersection) 
{

	float discriminant;
	float A, B, C;
	float lambda1, lambda2;
	t_vector temp;
	
	A = dotproduct(&ray->direction, &ray->direction);
	
	vec_sub(&temp, &ray->origin, &sphere->center);
	B = __fmul_rn (2.0, dotproduct(&temp, &ray->direction));
	
	C = __fadd_rn( dotproduct(&temp, &temp), 
		-__fmul_rn( sphere->radius, sphere->radius ));
	
	discriminant = __fadd_rn( __fmul_rn(B, B), 
		-__fmul_rn(4.0, __fmul_rn(A, C)));
	
	if (discriminant >= 0) {
		lambda1 = __fdiv_rn (__fadd_rn(-B,  __fsqrt_rn(discriminant)), 
				__fmul_rn(2.0, A));
		lambda2 = __fdiv_rn (__fadd_rn(-B, -__fsqrt_rn(discriminant)), 
				__fmul_rn(2.0, A));

		intersection->lambda_in = fminf(lambda1, lambda2);

		// is the object visible from the eye (lambda1,2>0)
		if (fequal( intersection->lambda_in, 0.0) || (lambda1>0 && lambda2>0) ){
			return true;
		}
	}
	return false;
}


// Calculate normal vector in the point of intersection:
__device__ void intersection_normal(t_sphere *sphere, 
		t_sphere_intersection* intersection, t_ray* ray) 
{
	float  scale;
	t_vector tmp_vec;
	
	//calculating coordinates of intersection point
	vec_scale(intersection->lambda_in, &tmp_vec, &ray->direction);
	vec_add(&intersection->point, &tmp_vec, &ray->origin);

	//calculating direction of normal in the point of intersection 
	vec_sub(&tmp_vec, &intersection->point, &sphere->center);
	
	//scaling normal vector
	scale = __frcp_rn(sphere->radius);
	vec_scale(scale, &intersection->normal, &tmp_vec);
	normalize_vector(&intersection->normal);
}





 __device__ t_color TraceRay(t_ray ray, int depth )
{
	t_ray ray_tmp;
	t_color illumination={0.0, 0.0, 0.0};
	t_color tmp;

	if( depth > DEPTH_MAX )
	{
		return illumination ;
	}

	t_sphere_intersection intersection, current_intersection;
	int intersection_object = -1; // none
	int k,i;

	float visible = 1.0;
	float current_lambda = FLT_MAX; // maximum positive float
	int count=0;

	//find closest ray object / intersection ;
	for (k=0; k<n_spheres; k++)
	{
		if (sphere_intersection(&ray, &spheres[k], &intersection))
		{
			if (intersection.lambda_in<current_lambda)
			{
				current_lambda=intersection.lambda_in;
				intersection_object=k;
				current_intersection=intersection;
			}
		}
	}
	//if( intersection exists )
        if (intersection_object > -1)
        {
		intersection_normal(&spheres[intersection_object], &current_intersection, &ray);
		//for each light source in the scene
		for (i=0; i<n_lights; i++)
		{
			compute_ray_to_light(&ray_tmp, &current_intersection, &lights[i]);

			for (k=0; k<n_spheres; k++)
			{
				if (sphere_intersection
					(&ray_tmp, &spheres[k], &intersection)
				   )
				{
					if (count++ == 0)
					{   
						visible = 0.2; 
					}else
					{   
						visible = 0.0;
					}   
					break;
				}
			}


			illumination.red   = __fadd_rn (
				illumination.red, __fmul_rn(visible, spheres[intersection_object].red));
			illumination.green = __fadd_rn (
				illumination.green, __fmul_rn(visible, spheres[intersection_object].green));
			illumination.blue  = __fadd_rn (
				illumination.blue, __fmul_rn(visible, spheres[intersection_object].blue));


		}
		compute_reflected_ray(&ray_tmp, &ray, &current_intersection);

		tmp = TraceRay(ray_tmp, depth+1 );

		illumination.red   = __fadd_rn (illumination.red,  tmp.red);
		illumination.blue  = __fadd_rn (illumination.blue, tmp.blue);
		illumination.green = __fadd_rn (illumination.green,tmp.green);

	}
	
	return illumination;
}





__global__ void kernel(unsigned char * dev_image_red, 
			unsigned char * dev_image_blue, 
			unsigned char * dev_image_green, 
			int  height, int width, 
			t_sphere * dev_spheres, int dev_n_spheres, 
			t_light * dev_lights, int dev_n_lights)
{

	t_color illumination;
	t_ray ray;
	t_pixel pixel;

	pixel.i = blockIdx.x * blockDim.x + threadIdx.x; // x coordinate inside whole picture
	pixel.j = blockIdx.y * blockDim.y + threadIdx.y; // y coordinate inside whole picture

	if (pixel.i>= width || pixel.j>=height)
	{
		return;
	}
	 
	int idx = threadIdx.x + threadIdx.y * blockDim.x; //linear index inside a block

	// is there a way to overcome warp divergence?
	if (threadIdx.x ==0 && threadIdx.y==0)
	{
		n_spheres = dev_n_spheres;
		n_lights = dev_n_lights;

		y_dim = __fdiv_rn (BOX_SIZE, __fdiv_rn ( (float) width, (float) height ));

		view_point.x = __fdiv_rn (X_DIM, 2.0);
		view_point.y = __fdiv_rn (y_dim, 2.0); 
		view_point.z = 0; 
	}

//	if (threadIdx.x < n_spheres && threadIdx.y==0 )
//	{
//		spheres[threadIdx.x].center = dev_spheres[threadIdx.x].center;
//		spheres[threadIdx.x].radius = dev_spheres[threadIdx.x].radius;
//		spheres[threadIdx.x].red    = dev_spheres[threadIdx.x].red;
//		spheres[threadIdx.x].green  = dev_spheres[threadIdx.x].green;
//		spheres[threadIdx.x].blue   = dev_spheres[threadIdx.x].blue;
//	}
//	if (threadIdx.x <n_lights && threadIdx.y==0)
//	{
//		lights[threadIdx.x] = dev_lights[threadIdx.x];
//	}

	if (idx < n_spheres * int(sizeof(t_sphere)/sizeof(float)) )
	{
		( (float * )spheres )[idx] = ((float *)dev_spheres)[idx];
	}
	__syncthreads();

	if (idx <n_lights * int(sizeof(t_light)/sizeof(float)) )
	{
		( (float * )lights )[idx] = ((float *) dev_lights)[idx];
	}
	__syncthreads();
    

	//compute ray starting point and direction ;
	compute_ray(&ray, &view_point, &pixel);
	illumination = TraceRay(ray, 0) ;
	//pixel color = illumination tone mapped to displayable range ;

	if (illumination.red>1.0)
		illumination.red=1.0;
	if (illumination.green>1.0)
		illumination.green=1.0;
	if (illumination.blue>1.0)
		illumination.blue=1.0;


	idx = pixel.i + __mul24(width, pixel.j);

	dev_image_red  [idx ]  = 
		(unsigned char) round (__fmul_rn (RGB_MAX, illumination.red));

	dev_image_green[ idx ]  = 
		(unsigned char) round (__fmul_rn (RGB_MAX, illumination.green));

	dev_image_blue [ idx ]  = 
		(unsigned char) round (__fmul_rn (RGB_MAX, illumination.blue));

}



