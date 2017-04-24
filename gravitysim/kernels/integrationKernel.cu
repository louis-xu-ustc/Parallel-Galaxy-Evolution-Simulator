#include <cuda.h>

#define THREADS6 512
#define FACTOR6 3
#define WARPSIZE 32
#define MAXDEPTH 32


__global__
__launch_bounds__(THREADS6, FACTOR6)
void IntegrationKernel()
{
  register int i, inc;
  register float dvelx, dvely, dvelz;
  register float velhx, velhy, velhz;

  // iterate over all bodies assigned to thread
  inc = blockDim.x * gridDim.x;
  for (i = threadIdx.x + blockIdx.x * blockDim.x; i < nbodiesd; i += inc) {
    // integrate
    dvelx = accxd[i] * dthfd;
    dvely = accyd[i] * dthfd;

    velhx = velxd[i] + dvelx;
    velhy = velyd[i] + dvely;

    posxd[i] += velhx * dtimed;
    posyd[i] += velhy * dtimed;

    velxd[i] = velhx + dvelx;
    velyd[i] = velhy + dvely;
  }
}
