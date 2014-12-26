#ifndef MYDEBUG_H
#define MYDEBUG_H

#include "types.h"

void print_array(unsigned char* p, int h, int w);
void print_array_float(float* p, int h, int w);
void print_spheres(t_sphere* spheres, int n_spheres);
void print_lights(t_light* lights, int n_lights);

#endif // MYDEBUG_H
