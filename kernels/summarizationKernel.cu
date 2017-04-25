#include <cuda.h>

#define THREADS3 1024
#define FACTOR3 1  /* must all be resident at the same time */
#define WARPSIZE 32
#define MAXDEPTH 32


__global__
__launch_bounds__(THREADS3, FACTOR3)
void SummarizationKernel()
{
  register int i, j, k, ch, inc, missing, cnt, bottom;
  register float m, cm, px, py, pz;
  __shared__ volatile int child[THREADS3 * 4];

  // comment: traverse from bottom to nnodesd
  // comment: bottom-up searching
  bottom = bottomd;
  // comment: inc is the stride width of the cuda thread
  inc = blockDim.x * gridDim.x;
  k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom)
    k += inc;

  // comment: at the start, no children is missing
  missing = 0;
  // comment: notice that actions are conducted on cells
  // iterate over all cells assigned to thread
  while (k <= nnodesd) {
    if (missing == 0) {
      // new cell, so initialize
      // comment: cm is short for cumulative mass
      cm = 0.0f;
      // comment: the cumulative position x and y
      px = 0.0f;
      py = 0.0f;
      // comment: cnt is for storing the number of all sub-node of this node
      cnt = 0;
      // comment: j refers to the number of non-null-pointer children
      j = 0;
      // comment: traverse its four children
      for (i = 0; i < 4; i++) {
        ch = childd[k*4+i];
        // comment: if this child is not null pointer (may be a cell or a body)
        if (ch >= 0) {
          // comment: this happens when some children is found to be null pointer
          // commnet: because j is only incremented when a non-null-pointer children is found
          // comment: when they are not equal, j should always be smaller than i
          if (i != j) {
            // comment: 
            // move children to front (needed later for speed)
            childd[k*4+i] = -1;
            childd[k*4+j] = ch;
          }
          child[missing*THREADS3+threadIdx.x] = ch;  // cache missing children
          m = massd[ch];
          // comment: assume the mass of the child is not ready yet -> missing++
          missing++;
          if (m >= 0.0f) {
            // comment: if child is ready -> missing--
            missing--;
            // comment: if the computed child is a cell
            if (ch >= nbodiesd) {  // count bodies (needed later)
              // comment: countd is for storing the number of sub-nodes of a node
              // comment: the storing can only be done only when all sub-nodes are computed
              cnt += countd[ch] - 1;
            }
            // add child's contribution
            cm += m;
            px += posxd[ch] * m;
            py += posyd[ch] * m;
          }
          j++;
        }
      }
      cnt += j;
    }

    // comment: some children are still not computed
    if (missing != 0) {
      do {
        // poll missing child
        ch = child[(missing-1)*THREADS3+threadIdx.x];
        m = massd[ch];
        if (m >= 0.0f) {
          // child is now ready
          missing--;
          if (ch >= nbodiesd) {
            // count bodies (needed later)
            cnt += countd[ch] - 1;
          }
          // add child's contribution
          cm += m;
          px += posxd[ch] * m;
          py += posyd[ch] * m;
          pz += poszd[ch] * m;
        }
        // repeat until we are done or child is not ready
      } while ((m >= 0.0f) && (missing != 0));
    }

    if (missing == 0) {
      // all children are ready, so store computed information
      countd[k] = cnt;
      m = 1.0f / cm;
      posxd[k] = px * m;
      posyd[k] = py * m;
      poszd[k] = pz * m;
      __threadfence();  // make sure data are visible before setting mass
      massd[k] = cm;
      k += inc;  // move on to next cell
    }
  }
}
