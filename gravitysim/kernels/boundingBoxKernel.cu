#include <cuda.h>

// thread count for a block
#define THREADS1 512  /* must be a power of 2 */
#define THREADS2 1024
#define THREADS3 1024
#define THREADS4 256
#define THREADS5 256
#define THREADS6 512

// block count = factor * #SMs
#define FACTOR1 3
#define FACTOR2 1
#define FACTOR3 1  /* must all be resident at the same time */
#define FACTOR4 1  /* must all be resident at the same time */
#define FACTOR5 5
#define FACTOR6 3

#define WARPSIZE 32
#define MAXDEPTH 32

__global__
void BoundingBoxKernel()
{
  register int i, j, k, inc;
  register float val, minx, maxx, miny, maxy, minz, maxz;
  __shared__ volatile float sminx[THREADS1], smaxx[THREADS1], sminy[THREADS1], smaxy[THREADS1], sminz[THREADS1], smaxz[THREADS1];

  // initialize with valid data (in case #bodies < #threads)
  minx = maxx = posxd[0];
  miny = maxy = posyd[0];
  minz = maxz = poszd[0];

  // scan all bodies
  i = threadIdx.x;
  inc = THREADS1 * gridDim.x;
  for (j = i + blockIdx.x * THREADS1; j < nbodiesd; j += inc) {
    val = posxd[j];
    minx = min(minx, val);
    maxx = max(maxx, val);
    val = posyd[j];
    miny = min(miny, val);
    maxy = max(maxy, val);
    val = poszd[j];
    minz = min(minz, val);
    maxz = max(maxz, val);
  }

  // reduction in shared memory
  sminx[i] = minx;
  smaxx[i] = maxx;
  sminy[i] = miny;
  smaxy[i] = maxy;
  sminz[i] = minz;
  smaxz[i] = maxz;

  for (j = THREADS1 / 2; j > 0; j /= 2) {
    __syncthreads();
    if (i < j) {
      k = i + j;
      sminx[i] = minx = min(minx, sminx[k]);
      smaxx[i] = maxx = max(maxx, smaxx[k]);
      sminy[i] = miny = min(miny, sminy[k]);
      smaxy[i] = maxy = max(maxy, smaxy[k]);
      sminz[i] = minz = min(minz, sminz[k]);
      smaxz[i] = maxz = max(maxz, smaxz[k]);
    }
  }

  // write block result to global memory
  if (i == 0) {
    k = blockIdx.x;
    minxd[k] = minx;
    maxxd[k] = maxx;
    minyd[k] = miny;
    maxyd[k] = maxy;
    minzd[k] = minz;
    maxzd[k] = maxz;

    inc = gridDim.x - 1;
    if (inc == atomicInc((unsigned int *)&blkcntd, inc)) {
      // I'm the last block, so combine all block results
      for (j = 0; j <= inc; j++) {
        minx = min(minx, minxd[j]);
        maxx = max(maxx, maxxd[j]);
        miny = min(miny, minyd[j]);
        maxy = max(maxy, maxyd[j]);
        minz = min(minz, minzd[j]);
        maxz = max(maxz, maxzd[j]);
      }

      // compute 'radius'
      val = max(maxx - minx, maxy - miny);
      radiusd = max(val, maxz - minz) * 0.5f;

      // create root node
      k = nnodesd;
      bottomd = k;

      massd[k] = -1.0f;
      startd[k] = 0;
      posxd[k] = (minx + maxx) * 0.5f;
      posyd[k] = (miny + maxy) * 0.5f;
      poszd[k] = (minz + maxz) * 0.5f;
      k *= 8;
      for (i = 0; i < 8; i++) childd[k + i] = -1;

      stepd++;
    }
  }
}

