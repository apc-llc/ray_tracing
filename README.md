## Simple Ray Tracing engine in CUDA and OptiX

### Prerequisites

* Each version of OptiX works only starting from specific version of CUDA. We've tested CUDA `6.5` with OptiX `3.7.0`.

### Building

Use central `makefile` to build both *cuda* and *optix* versions:

```
$ git clone https://github.com/apc-llc/ray_tracing.git
$ cd ray_tracing
$ make
```
