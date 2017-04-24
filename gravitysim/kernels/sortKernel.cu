#include <cuda.h>

#define THREADS4 256
#define FACTOR4 1  /* must all be resident at the same time */


__global__
__launch_bounds__(THREADS4, FACTOR4)
void SortKernel()
{
  register int i, k, ch, dec, start, bottom;

  bottom = bottomd;
  // comment: stride, just like inc 
  dec = blockDim.x * gridDim.x;
  k = nnodesd + 1 - dec + threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all cells assigned to thread
  while (k >= bottom) {
    // comment: the startd is used to signify the boundary in the sortd array
    // comment: it concurrently places the bodies into an array such that the bodies appear in the same order in the array as they would during an in-order traversal of the octree
    start = startd[k];
    // comment: this is quite like kernel 3, if the start is still -1, it keeps polling until the start is ready
    // comment: at the start, only root is able to compute because only its start is not -1 (it is 0)
    // comment: start serves both purpose, one is for signify whether it can start, another is for signify the area it puts its elements
    if (start >= 0) {
      // comment: traverse from left child to right child
      for (i = 0; i < 4; i++) {
        // comment: iterate through the children of the cell
        ch = childd[k*4+i];
        if (ch >= nbodiesd) {
          // child is a cell
          startd[ch] = start;  // set start ID of child
          start += countd[ch];  // add #bodies in subtree
        } else if (ch >= 0) {
          // child is a body
          sortd[start] = ch;  // record body in 'sorted' array
          start++;
        }
      }
      k -= dec;  // move on to next cell
    }
    __syncthreads();  // throttle
  }
}
