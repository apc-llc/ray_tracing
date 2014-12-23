#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

#include "lib/EasyBMP.h"
#include "ray_tracing.h"

#include "debug.h"

#define ENABLE_CHECK

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
	printf ("Picture size is width=%d  height=%d \n", width, height);
#endif

	unsigned char * pR = (unsigned char *) malloc( height*width );
	unsigned char * pG = (unsigned char *) malloc( height*width );
	unsigned char * pB = (unsigned char *) malloc( height*width );

	if ( pR == NULL || pG == NULL || pB == NULL)
	{
		printf ("Malloc error. Exiting. \n");
		return -1;
	}

	ray_trace(pR, pG, pB, height, width, n_spheres, n_lights);

	BMP AnImage;
	AnImage.SetSize(width, height);
	for (int i = 0 ; i < width ; i++)
	{
		for (int j = 0 ; j < height ; j++)
		{
			RGBApixel pixel ;
			pixel.Red = pR [ j * width + i ] ;
			pixel.Green = pG [ j * width + i ] ;
			pixel.Blue = pB [ j * width + i ] ;
			AnImage.SetPixel( i , j , pixel ) ;
		}
	}
	AnImage.WriteToFile(argv[5]);

	free(pR);
	free(pG);
	free(pB);
	
	return 0;
}

