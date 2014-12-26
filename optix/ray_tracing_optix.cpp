#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <optix.h>
#include <sutil.h>
#include <optix_world.h>

#include "EasyBMP.h"
#include "types.h"

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

	cam_eye[0]= 10.0f;  cam_eye[1]= 5.625f;  cam_eye[2]=  0.0f;
	lookat[0] = 5.0f;  lookat[1] = 5.625f;  lookat[2] = 25.f;
	up[0]     = 0.0f;  up[1]     = -1.0f;  up[2]     = 0.0f;
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

static void generateScene(optix::float4 * coords, optix::float3 * colors, int n_spheres,
	BasicLight * lights, int n_lights)
{
	int n_random_coord = n_spheres * 3  + n_lights * 3;
	int n_random_rad = n_spheres;
	int n_random_colors = n_spheres * 3;

	size_t n = n_random_coord + n_random_rad + n_random_colors;

	curandGenerator_t gen;
	float *hostData;
	hostData = (float *)calloc(n, sizeof(float));

	if (!hostData)
	{
		fprintf(stderr, "Malloc error, exiting\n");
		exit(-1);
	}

	float *devData;
	CUDA_CALL( cudaMalloc((void **)&devData, n*sizeof(float)) );

	CURAND_CALL( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
	CURAND_CALL( curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)time(NULL)) ); 

	CURAND_CALL( curandGenerateUniform(gen, devData, n) );
	CUDA_CALL( cudaMemcpy(hostData, devData, n * sizeof(float), cudaMemcpyDeviceToHost) );

	float x_pos = 0.9f;
	float y_pos = BOX_SIZE / 5.0;

	for (int i = 0; i < n_spheres; i++)
	{
		coords[i].x = x_pos;
		coords[i].y = y_pos;

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
		coords[i].x += 2.0 * (hostData[j++] - 0.5);
		coords[i].y += 2.0 * (hostData[j++] - 0.5);
		coords[i].z = hostData[j++] * BOX_SIZE_Z + DISTANCE;
		coords[i].w = hostData[j++] * RADIUS_MAX + RADIUS_MIN;
		colors[i].x = hostData[j++] / (DEPTH_MAX - 3);
		colors[i].y = hostData[j++] / (DEPTH_MAX - 3);
		colors[i].z = hostData[j++] / (DEPTH_MAX - 3);
	}

	for (int i = 0; i < n_lights; i++)
	{
		lights[i].pos.x = (hostData[j++] - 0.5) * BOX_SIZE * 6;
		lights[i].pos.y = (hostData[j++] - 0.5) * BOX_SIZE * 6;
		lights[i].pos.z = hostData[j++] * DISTANCE/2.0;
	}

	CURAND_CALL( curandDestroyGenerator(gen) );
	CUDA_CALL( cudaFree(devData) );
	free(hostData);
}

static void createGeometry( RTcontext context, RTgeometry* geometry, int n_spheres, int n_lights,
	char** values )
{
	RT_CHECK_ERROR( rtGeometryCreate( context, geometry ) );
	RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( *geometry, 1u ) );

	RTprogram  intersection_program;
	RTprogram  bounding_box_program;
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, "sphere.ptx", "bounds", &bounding_box_program) );
	RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( *geometry, bounding_box_program ) );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, "sphere.ptx", "intersect", &intersection_program) );
	RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( *geometry, intersection_program ) );

	RTvariable spheres, spheres_colors, lights;
	optix::float4 * spheresCoords;
	optix::float3 * spheresColors;

	RTbuffer buff_coords, buff_colors;
	RTbuffer buff_lights;
	BasicLight * lightsCoords;

	RT_CHECK_ERROR(rtBufferCreate(context, RT_BUFFER_INPUT, &buff_lights));
	RT_CHECK_ERROR(rtBufferSetFormat(buff_lights, RT_FORMAT_USER));
	RT_CHECK_ERROR(rtBufferSetElementSize(buff_lights, sizeof(BasicLight)));
	RT_CHECK_ERROR(rtBufferSetSize1D(buff_lights, n_lights));
	RT_CHECK_ERROR(rtContextDeclareVariable( context, "lights", &lights ) );
	RT_CHECK_ERROR(rtVariableSetObject( lights, buff_lights ) );
	RT_CHECK_ERROR(rtBufferMap(buff_lights, (void**)&lightsCoords));

	RT_CHECK_ERROR(rtBufferCreate(context, RT_BUFFER_INPUT, &buff_coords));
	RT_CHECK_ERROR(rtBufferSetFormat(buff_coords, RT_FORMAT_FLOAT4));
	RT_CHECK_ERROR(rtBufferSetSize1D(buff_coords, n_spheres));
	RT_CHECK_ERROR( rtContextDeclareVariable( context, "spheres", &spheres ) );
	RT_CHECK_ERROR( rtVariableSetObject( spheres, buff_coords ) );
	RT_CHECK_ERROR(rtBufferMap(buff_coords, (void**)&spheresCoords));

	RT_CHECK_ERROR(rtBufferCreate(context, RT_BUFFER_INPUT, &buff_colors));
	RT_CHECK_ERROR(rtBufferSetFormat(buff_colors, RT_FORMAT_FLOAT3));
	RT_CHECK_ERROR(rtBufferSetSize1D(buff_colors, n_spheres));
	RT_CHECK_ERROR( rtContextDeclareVariable( context, "spheres_colors", &spheres_colors ) );
	RT_CHECK_ERROR( rtVariableSetObject( spheres_colors, buff_colors ) );
	RT_CHECK_ERROR(rtBufferMap(buff_colors, (void**)&spheresColors));

	if (!values)
		generateScene(spheresCoords, spheresColors, n_spheres, lightsCoords, n_lights);
	else
	{
		// Parse scene from the command line.
		char** value = values;
		value++; // skip n_spheres
		for (int i = 0; i < n_spheres; i++)
		{
			spheresCoords[i].x = atof(*(value++));
			spheresCoords[i].y = atof(*(value++));
			spheresCoords[i].z = atof(*(value++));
			spheresCoords[i].w = atof(*(value++));
			spheresColors[i].x = atof(*(value++));
			spheresColors[i].y = atof(*(value++));
			spheresColors[i].z = atof(*(value++));
		}
		value++; // skip n_lights
		for (int i = 0; i < n_lights; i++)
		{
			lightsCoords[i].pos.x = atof(*(value++));
			lightsCoords[i].pos.y = atof(*(value++));
			lightsCoords[i].pos.z = atof(*(value++));
		}
	}

	RT_CHECK_ERROR(rtBufferUnmap(buff_coords));
	RT_CHECK_ERROR(rtBufferUnmap(buff_colors));
	RT_CHECK_ERROR(rtBufferUnmap(buff_lights));

	RTvariable Ka, Kd, Ks, phong_exp;
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

static void createInstance( RTcontext context, RTgeometry geometry, RTmaterial material )
{
	RTgeometrygroup geometrygroup;
	RTvariable top_object, top_shadower;
	RTacceleration acceleration;
	RTgeometryinstance instance;

	// Create geometry instance
	RT_CHECK_ERROR( rtGeometryInstanceCreate( context, &instance ) );
	RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( instance, geometry ) );
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
	bool randomScene = false;
	if (argc > 1)
		if ((std::string)argv[1] == "random")
			randomScene = true;

	if (randomScene && (argc != 7))
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

	RTgeometry geometry;
	// Pass down spheres/lights, if specific scene is given in command line.
	char** values = NULL;
	if (!randomScene) values = &argv[1];
	createGeometry( context, &geometry, n_spheres, n_lights, values );

	RTmaterial material;
	createMaterial( context, &material);
	createInstance( context, geometry, material );

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
	writeBMP( context, filename, output_buffer_obj );

	// Clean up
	RT_CHECK_ERROR( rtContextDestroy( context ) );
	
	return 0;
}

