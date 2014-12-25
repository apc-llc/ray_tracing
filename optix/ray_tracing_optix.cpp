#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <optix.h>
#include <sutil.h>
#include <optix_world.h>

#include "EasyBMP.h"

#define CUDA_CALL(x) do { cudaError_t err = x; if (( err ) != cudaSuccess ) { \
	printf ("Error \"%s\" at %s :%d \n" , cudaGetErrorString(err), \
			__FILE__ , __LINE__ ) ; exit(-1);\
}} while (0)

typedef struct struct_BoxExtent
{
	float min[3];
	float max[3];
}
BoxExtent;

struct BasicLight
{
	optix::float3 pos;
	optix::float3 color;
	int casts_shadow;
};

unsigned int width = 1920;
unsigned int height = 1080;

static void createContext( RTcontext* context, RTbuffer* output_buffer_obj )
{
	RTprogram  ray_gen_program;
	RTprogram  exception_program;
	RTprogram  miss_program;
	RTvariable output_buffer;
	RTvariable radiance_ray_type, shadow_ray_type;
	RTvariable epsilon;

	// Variables for ray gen program
	RTvariable eye;
	RTvariable U;
	RTvariable V;
	RTvariable W;
	RTvariable badcolor;

	// Viewing params
	float cam_eye[3], lookat[3], up[3];
	float hfov, aspect_ratio; 
	float camera_u[3], camera_v[3], camera_w[3];

	// Variables for miss program
	RTvariable bg_color;

	// Setup context
	RT_CHECK_ERROR2( rtContextCreate( context ) );
	RT_CHECK_ERROR2( rtContextSetRayTypeCount( *context, 2 ) );
	RT_CHECK_ERROR2( rtContextSetEntryPointCount( *context, 1 ) );

	RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "output_buffer" , &output_buffer) );
	RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "radiance_ray_type" , &radiance_ray_type) );
	RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "shadow_ray_type" , &shadow_ray_type) );  
	RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "scene_epsilon" , &epsilon) );

	RT_CHECK_ERROR2( rtVariableSet1ui( radiance_ray_type, 0u ) );
	RT_CHECK_ERROR2( rtVariableSet1ui( shadow_ray_type, 1u ) );
  
	RT_CHECK_ERROR2( rtVariableSet1f( epsilon, 1.e-3f ) );

	// Render result buffer
	RT_CHECK_ERROR2( rtBufferCreate( *context, RT_BUFFER_OUTPUT, output_buffer_obj ) );
	RT_CHECK_ERROR2( rtBufferSetFormat( *output_buffer_obj, RT_FORMAT_UNSIGNED_BYTE4 ) );
	RT_CHECK_ERROR2( rtBufferSetSize2D( *output_buffer_obj, width, height ) );
	RT_CHECK_ERROR2( rtVariableSetObject( output_buffer, *output_buffer_obj ) );

	// Ray generation program
	RT_CHECK_ERROR2( rtProgramCreateFromPTXFile( *context, "pinhole_camera.ptx", "pinhole_camera", &ray_gen_program ) );
	RT_CHECK_ERROR2( rtContextSetRayGenerationProgram( *context, 0, ray_gen_program ) );
	RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "eye" , &eye) );
	RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "U" , &U) );
	RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "V" , &V) );
	RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "W" , &W) );

	cam_eye[0]= 0.0f;  cam_eye[1]= 0.0f;  cam_eye[2]= 5.0f;
	lookat[0] = 0.0f;  lookat[1] = 0.0f;  lookat[2] = 0.0f;
	up[0]     = 0.0f;  up[1]     = 1.0f;  up[2]     = 0.0f;
	hfov      = 60.0f;

	aspect_ratio = (float)width/(float)height;
	sutilCalculateCameraVariables( cam_eye, lookat, up, hfov, aspect_ratio,
		camera_u, camera_v, camera_w );

	RT_CHECK_ERROR2( rtVariableSet3fv( eye, cam_eye ) );
	RT_CHECK_ERROR2( rtVariableSet3fv( U, camera_u ) );
	RT_CHECK_ERROR2( rtVariableSet3fv( V, camera_v ) );
	RT_CHECK_ERROR2( rtVariableSet3fv( W, camera_w ) );

	// Exception program
	RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "bad_color" , &badcolor) );
	RT_CHECK_ERROR2( rtVariableSet3f( badcolor, .0f, .0f, 0.5f ) );
	RT_CHECK_ERROR2( rtProgramCreateFromPTXFile( *context, "pinhole_camera.ptx", "exception", &exception_program ) );
	RT_CHECK_ERROR2( rtContextSetExceptionProgram( *context, 0, exception_program ) );

	// Miss program
	RT_CHECK_ERROR2( rtProgramCreateFromPTXFile( *context, "constantbg.ptx", "miss", &miss_program ) );
	RT_CHECK_ERROR2( rtProgramDeclareVariable( miss_program, "bg_color" , &bg_color) );
	RT_CHECK_ERROR2( rtVariableSet3f( bg_color, .0f, .0f, .0f ) );
	RT_CHECK_ERROR2( rtContextSetMissProgram( *context, 0, miss_program ) );
}

static void createGeometry( RTcontext context, RTgeometry* sphere )
{
	int num_spheres=2;
	int num_lightSources = 2;
	RT_CHECK_ERROR( rtGeometryCreate( context, sphere ) );
	RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( *sphere, 1u ) );

	RTprogram  intersection_program;
	RTprogram  bounding_box_program;
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, "sphere.ptx", "bounds", &bounding_box_program) );
	RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( *sphere, bounding_box_program ) );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, "sphere.ptx", "intersect", &intersection_program) );
	RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( *sphere, intersection_program ) );

	RTvariable spheres,spheres_colors, lights;
	optix::float4 * coords;
	optix::float3 * colors;

	RTbuffer buff_coords,buff_colors;
	RTbuffer buff_lights;
	BasicLight * lightsPtr;

	RT_CHECK_ERROR(rtBufferCreate(context, RT_BUFFER_INPUT, &buff_lights));
	RT_CHECK_ERROR(rtBufferSetFormat(buff_lights, RT_FORMAT_USER));
	RT_CHECK_ERROR(rtBufferSetElementSize(buff_lights, sizeof(BasicLight)));
	RT_CHECK_ERROR(rtBufferSetSize1D(buff_lights, num_lightSources));
	RT_CHECK_ERROR(rtContextDeclareVariable( context, "lights", &lights ) );
	RT_CHECK_ERROR(rtVariableSetObject( lights, buff_lights ) );
	RT_CHECK_ERROR(rtBufferMap(buff_lights, (void**)&lightsPtr));

	RT_CHECK_ERROR(rtBufferCreate(context, RT_BUFFER_INPUT, &buff_coords));
	RT_CHECK_ERROR(rtBufferSetFormat(buff_coords, RT_FORMAT_FLOAT4));
	RT_CHECK_ERROR(rtBufferSetSize1D(buff_coords, num_spheres));
	RT_CHECK_ERROR( rtContextDeclareVariable( context, "spheres", &spheres ) );
	RT_CHECK_ERROR( rtVariableSetObject( spheres, buff_coords ) );
	RT_CHECK_ERROR(rtBufferMap(buff_coords, (void**)&coords));

	RT_CHECK_ERROR(rtBufferCreate(context, RT_BUFFER_INPUT, &buff_colors));
	RT_CHECK_ERROR(rtBufferSetFormat(buff_colors, RT_FORMAT_FLOAT3));
	RT_CHECK_ERROR(rtBufferSetSize1D(buff_colors, num_spheres));
	RT_CHECK_ERROR( rtContextDeclareVariable( context, "spheres_colors", &spheres_colors ) );
	RT_CHECK_ERROR( rtVariableSetObject( spheres_colors, buff_colors ) );
	RT_CHECK_ERROR(rtBufferMap(buff_colors, (void**)&colors));
	for (int i = 0; i < num_spheres; i++)
	{
		coords[i].x = i * 0.6;
		coords[i].y = i * 0.6;
		coords[i].z = 0;
		coords[i].w = 0.25f;
		colors[i].x = i * 0.5f;
		colors[i].y = 1 - i * 0.5f;
		colors[i].z = 0;	
	}
	lightsPtr[0].pos.x = -10.0f;
	lightsPtr[0].pos.y = -10.0f;
	lightsPtr[0].pos.z = 0.0f;

	lightsPtr[1].pos.x = 10.0f;
	lightsPtr[1].pos.y = -10.0f;
	lightsPtr[1].pos.z = 5.0f;

	RT_CHECK_ERROR(rtBufferUnmap(buff_coords));
	RT_CHECK_ERROR(rtBufferUnmap(buff_colors));
	RT_CHECK_ERROR(rtBufferUnmap(buff_lights));

	RTvariable Ka,Kd,Ks,phong_exp;
	RT_CHECK_ERROR(rtContextDeclareVariable( context, "Ka", &Ka ) );
	RT_CHECK_ERROR(rtContextDeclareVariable( context, "Kd", &Kd ) );
	RT_CHECK_ERROR(rtContextDeclareVariable( context, "Ks", &Ks ) );
	RT_CHECK_ERROR(rtContextDeclareVariable( context, "phong_exp", &phong_exp ) );
	RT_CHECK_ERROR(rtVariableSet3f(Ka, 1.0f,1.0f,1.0f));
	RT_CHECK_ERROR(rtVariableSet3f(Kd, 0.6f,0.6f,0.6f));
	RT_CHECK_ERROR(rtVariableSet3f(Ks, 0.4f,0.4f,0.4f));
	RT_CHECK_ERROR(rtVariableSet3f(Ka, 1.0f,1.0f,1.0f));
	RT_CHECK_ERROR(rtVariableSet1f(phong_exp, 88));
}

static void createMaterial( RTcontext context, RTmaterial* material )
{
	RTprogram chp, ahp;

	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, "normal_shader.ptx", "closest_hit_radiance", &chp ) );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, "normal_shader.ptx", "any_hit_shadow", &ahp ) );

	RT_CHECK_ERROR( rtMaterialCreate( context, material ) );
	RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( *material, 0, chp) );
	RT_CHECK_ERROR( rtMaterialSetAnyHitProgram( *material, 1, ahp) );
}

static void createInstance( RTcontext context, RTgeometry sphere, RTmaterial material )
{
	RTgeometrygroup geometrygroup;
	RTvariable top_object, top_shadower;
	RTacceleration acceleration;
	RTgeometryinstance instance;

	// Create geometry instance
	RT_CHECK_ERROR( rtGeometryInstanceCreate( context, &instance ) );
	RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( instance, sphere ) );
	RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( instance, 1 ) );
	RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( instance, 0, material ) );

	// Create geometry group
	RT_CHECK_ERROR( rtAccelerationCreate( context, &acceleration ) );
	RT_CHECK_ERROR( rtAccelerationSetBuilder( acceleration, "NoAccel" ) );
	RT_CHECK_ERROR( rtAccelerationSetTraverser( acceleration, "NoAccel" ) );
	RT_CHECK_ERROR( rtGeometryGroupCreate( context, &geometrygroup ) );
	RT_CHECK_ERROR( rtGeometryGroupSetChildCount( geometrygroup, 1 ) );
	RT_CHECK_ERROR( rtGeometryGroupSetChild( geometrygroup, 0, instance ) );
	RT_CHECK_ERROR( rtGeometryGroupSetAcceleration( geometrygroup, acceleration ) );

	RT_CHECK_ERROR( rtContextDeclareVariable( context, "top_object", &top_object ) );
	RT_CHECK_ERROR( rtContextDeclareVariable( context, "top_shadower", &top_shadower ) );

	RT_CHECK_ERROR( rtVariableSetObject( top_object, geometrygroup ) );
	RT_CHECK_ERROR( rtVariableSetObject( top_shadower, geometrygroup ) );  
}

static void writeBMP( RTcontext context, const char* filename, RTbuffer buffer)
{
	// Check to see if the buffer is two dimensional
	unsigned int ndims;
	RT_CHECK_ERROR( rtBufferGetDimensionality(buffer, &ndims) );
	if (ndims != 2)
	{
		fprintf(stderr, "Dimensionality of the buffer is %u instead of 2\n", ndims);
		exit(-1);
	}

	// Check to see if the buffer is of type float{1,3,4} or uchar4
	RTformat format;
	RT_CHECK_ERROR( rtBufferGetFormat(buffer, &format) );
	if (RT_FORMAT_UNSIGNED_BYTE4 != format)
	{
		fprintf(stderr, "Buffer's format isn't uchar4\n");
		exit(-1);
	}

	void* imageData;
	RT_CHECK_ERROR( rtBufferMap(buffer, &imageData) );
	if (!imageData)
	{
		fprintf(stderr, "Data in buffer is NULL\n");
		exit(-1);
	}

	// Data is BGRA and upside down, so we need to swizzle to RGB
	BMP AnImage;
	AnImage.SetSize(width, height);
	for (int j = height - 1; j >= 0; --j)
	{
		unsigned char *src = ((unsigned char*)imageData) + (4 * width * j);
		for (int i = 0; i < width; i++)
		{
			RGBApixel pixel;
			pixel.Red = *(src + 2);
			pixel.Green = *(src + 1);
			pixel.Blue = *(src + 0);
			src += 4;
			AnImage.SetPixel( i , j , pixel ) ;
		}
	}
	AnImage.WriteToFile(filename);

	RT_CHECK_ERROR( rtBufferUnmap(buffer) );
}

int main(int argc, char* argv[])
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
	printf ("Picture size is width=%d  height=%d \n", width, height);
#endif

	cudaEvent_t start = 0, stop = 0;
	CUDA_CALL( cudaEventCreate (&start) );
	CUDA_CALL( cudaEventCreate (&stop) );
	CUDA_CALL( cudaEventRecord (start, 0) );

    // Create our objects and set state
    RTcontext context;
    RT_CHECK_ERROR( rtContextCreate( &context ) );
    RT_CHECK_ERROR( rtContextSetRayTypeCount( context, 1 ) );
    RT_CHECK_ERROR( rtContextSetEntryPointCount( context, 1 ) );

	// Setup state
	RTbuffer output_buffer_obj;
	createContext( &context, &output_buffer_obj );

	RTgeometry sphere;
	createGeometry( context, &sphere );

	RTmaterial material;
	createMaterial( context, &material);
	createInstance( context, sphere, material );

	// Run
	RT_CHECK_ERROR( rtContextValidate( context ) );
	RT_CHECK_ERROR( rtContextCompile( context ) );
	RT_CHECK_ERROR( rtContextLaunch2D( context, 0, width, height ) );

	CUDA_CALL( cudaEventRecord (stop, 0) );
	CUDA_CALL( cudaEventSynchronize(stop) );

	float gpuTime = 0.0f;
	CUDA_CALL( cudaEventElapsedTime (&gpuTime, start, stop) );

	printf("OptiX ray tracing time: %.2f milliseconds\n", gpuTime);

    // Display image
	writeBMP( context, argv[5], output_buffer_obj );

	// Clean up
	RT_CHECK_ERROR( rtContextDestroy( context ) );
	
	return 0;
}

