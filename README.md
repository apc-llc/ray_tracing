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

### Comparing CUDA and OptiX

Frontend tool `ray_tracing_compare` invokes CUDA and OptiX versions on the same randomly generated scene:

```
$ make compare
./ray_tracing_compare 10 2 1920 1080 output.bmp
CUDA ray tracing time: 12.52 milliseconds
OptiX ray tracing time: 232.49 milliseconds
```

Output scene will be rendered to `output.bmp` in `cuda` and `optix` subfolders, respectively.

