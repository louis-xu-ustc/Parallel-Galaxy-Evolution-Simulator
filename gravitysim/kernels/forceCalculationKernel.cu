#include <cuda.h>

#define THREADS5 256
#define FACTOR5 5
#define WARPSIZE 32
#define MAXDEPTH 32


__global__
__launch_bounds__(THREADS5, FACTOR5)
void ForceCalculationKernel()
{
  register int i, j, k, n, depth, base, sbase, diff;
  register float px, py, pz, ax, ay, az, dx, dy, dz, tmp;
  __shared__ volatile int pos[MAXDEPTH * THREADS5/WARPSIZE], node[MAXDEPTH * THREADS5/WARPSIZE];
  __shared__ volatile float dq[MAXDEPTH * THREADS5/WARPSIZE];
  __shared__ volatile int step, maxdepth;

  if (0 == threadIdx.x) {
    step = stepd;
    maxdepth = maxdepthd;
    tmp = radiusd;
    // precompute values that depend only on tree level
    dq[0] = tmp * tmp * itolsqd;
    for (i = 1; i < maxdepth; i++) {
      dq[i] = dq[i - 1] * 0.25f;
    }

    if (maxdepth > MAXDEPTH) {
      *errd = maxdepth;
    }
  }
  __syncthreads();

  if (maxdepth <= MAXDEPTH) {
    // figure out first thread in each warp (lane 0)
    base = threadIdx.x / WARPSIZE;
    sbase = base * WARPSIZE;
    j = base * MAXDEPTH;

    diff = threadIdx.x - sbase;
    // make multiple copies to avoid index calculations later
    if (diff < MAXDEPTH) {
      dq[diff+j] = dq[diff];
    }
    __syncthreads();

    // iterate over all bodies assigned to thread
    for (k = threadIdx.x + blockIdx.x * blockDim.x; k < nbodiesd; k += blockDim.x * gridDim.x) {
      i = sortd[k];  // get permuted/sorted index
      // cache position info
      px = posxd[i];
      py = posyd[i];

      ax = 0.0f;
      ay = 0.0f;

      // initialize iteration stack, i.e., push root node onto stack
      depth = j;
      if (sbase == threadIdx.x) {
        node[j] = nnodesd;
        pos[j] = 0;
      }
      __threadfence();  // make sure it's visible

      while (depth >= j) {
        // stack is not empty
        while (pos[depth] < 4) {
          // node on top of stack has more children to process
          n = childd[node[depth]*4+pos[depth]];  // load child pointer
          if (sbase == threadIdx.x) {
            // I'm the first thread in the warp
            pos[depth]++;
          }
          __threadfence();  // make sure it's visible
          if (n >= 0) {
            dx = posxd[n] - px;
            dy = posyd[n] - py;
            tmp = dx*dx + (dy*dy + epssqd);  // compute distance squared (plus softening)
            if ((n < nbodiesd) || __all(tmp >= dq[depth])) {  // check if all threads agree that cell is far enough away (or is a body)
              tmp = rsqrtf(tmp);  // compute distance
              tmp = massd[n] * tmp * tmp * tmp;
              ax += dx * tmp;
              ay += dy * tmp;
            } else {
              // push cell onto stack
              depth++;
              if (sbase == threadIdx.x) {
                node[depth] = n;
                pos[depth] = 0;
              }
              __threadfence();  // make sure it's visible
            }
          } else {
            depth = max(j, depth - 1);  // early out because all remaining children are also zero
          }
        }
        depth--;  // done with this level
      }

      if (step > 0) {
        // update velocity
        velxd[i] += (ax - accxd[i]) * dthfd;
        velyd[i] += (ay - accyd[i]) * dthfd;
      }

      // save computed acceleration
      accxd[i] = ax;
      accyd[i] = ay;
    }
  }
}