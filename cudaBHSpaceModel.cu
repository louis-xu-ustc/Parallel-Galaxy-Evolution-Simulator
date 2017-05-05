#include <stdio.h>
#include "cuda.h"
#include "cudaBHSpaceModel.h"


#define NULL_BODY (-1)
#define LOCK (-2)

#define printf(...) 

// thread count for a block
#define THREADS2 1024
#define THREADS3 1024
#define THREADS4 256
#define THREADS5 256
#define THREADS6 512

// block count = factor * #SMs
#define FACTOR2 1
#define FACTOR3 1  /* must all be resident at the same time */
#define FACTOR4 1  /* must all be resident at the same time */
#define FACTOR5 5
#define FACTOR6 3

#define WARPSIZE 32
#define MAXDEPTH 32

// nbodiesd -> number of leaf nodes
__constant__ int nbodiesd;
// dtimed -> update interval
__constant__ float dtimed, dthfd, epssqd, itolsqd;
// leaf node information, transfer from CPU to GPU
__constant__ volatile float *leaf_node_mass, *leaf_node_posx, *leaf_node_posy, *leaf_node_velx, *leaf_node_vely, *leaf_node_accx, *leaf_node_accy;
// internal node information, generated in GPU, should be cudaMalloc-ed
// TODO how do cudaMalloc do initialization
__device__ volatile int internal_node_num;    // <= nbodiesd
__constant__ volatile int *internal_node_child;   // 4 x nbodiesd, initialized to -1
__constant__ volatile int *internal_node_mass;    // nbodiesd
__constant__ volatile int *internal_node_posx;    // nbodiesd
__constant__ volatile int *internal_node_posy;    // nbodiesd

// maxdepthd is for accelerating computation, can be ignored
// __device__ volatile int maxdepthd;

// TODO placed in __constant__
__device__ volatile float radiusd;


__global__ void InitializationKernel() {
    printf("initialization kernel\n");
    // the below steps are originally done in another kernel
    radiusd = 700.f;
    int k = 0; // root node placed at first place
    for (int i = 0; i < 4; i++)
      internal_node_child[4 * k + i] = -1;
    internal_node_mass[k] = -1.0f;
    internal_node_posx[k] = 500.f;
    internal_node_posy[k] = 300.f;
    internal_node_num = 1;  // root
}


__global__
void TreeBuildingKernel()
{
  // register int i, j, k, depth, localmaxdepth, skip, inc;
  // register float x, y, r;
  // register float px, py;
  // register int ch, n, cell, locked, patch;
  // register float radius, rootx, rooty;

  // cache root data
  float radius = radiusd;
  // nnodesd is the number of 
  float rootx = internal_node_posx[0];
  float rooty = internal_node_posy[0];

  // comments: why need recording depth at all, not necessary, seem not to save computation
  // localmaxdepth = 1;
  // skip = 1;

  // After each thread finishes its work at current index, increment each of them by the total number of threads running in the grid, which is blockDim.x*gridDim.x
  int inc = blockDim.x * gridDim.x;
  // current thread id
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all bodies assigned to thread
  // comment: a simple comparison of the array index with the number of bodies determines whether the index points to a cell or a body
  // comment: since i is the unique thread id, if not properly scheduled, many thread resources may be wasted
  // comment: this one looks strange
  while (i < nbodiesd) {
    // if (skip != 0) {
    // comment: do only once when the thread try this body for the first time
    // comment: when retrying, we don't want to start ahead again
    // comment: instead, we want to start from where we left off
    // new body, so start traversing at root
    // skip = 0;

    // its original coordinate
    int px = leaf_node_posx[i];
    int py = leaf_node_posy[i];
    // n = nnodesd;
    int n = 0;  // root
    // comment: the depth 
    r = radius;
    // comment: j is the index of children (of a node)
    j = 0;
    // determine which child to follow
    // comment: left up - 2; right up - 3; left down - 0; right down - 1
    if (rootx < px) j = 1;
    if (rooty < py) j += 2;
    // }

    // follow path to leaf cell
    // comment: childd is a specialized array for storing the children nodes
    // comment: since every node can have at most 4 children, we allocate the most space for them
    // comment: this is for convenience of traversal
    // comment: ch is the value of that child node, possibly pointing to another node, possibly just -1 (null pointer)
    // comment: if didin't skip, this should point to children of root node
    // comment: if did skip, it probably point to something else

    // start with the children of root
    int child = internal_node_child[j];
    // comment: in some code sections, we need to find out whether an index refers to a body or to null. Because −1 is also smaller than the number of bodies, a single integer comparison suffices to test both conditions.
    // comment: if ch >= nbodiesd, the child already points to a cell, then go find some place in its children
    while (child >= nbodiesd) {
      // the - nbodiesd is for differentiating internal node and leaf node
      n = child - nbodiesd;
      // find it in the next level
      r *= 0.5f;
      // comment: reset j
      j = 0;
      // determine which child to follow
      if (internal_node_posx[n] < px) j = 1;
      if (internal_node_posy[n] < py) j += 2;
      child = internal_node_child[n*4+j];
    }
    // comment: finally found a null pointer (-1 <= nbodiesd) or a body (since the id of bodies is <= nbodiesd)
    // comment: this two situations are reflected below

    // skip if child pointer is locked and try again later
    if (child != -2) {
      // comment: if it is not locked, try to lock it
      int locked = n*4+j;
      // try to lock. However, it is still possible to fail because of contention
      // comments: if fail, also try again later
      if (child == atomicCAS((int *)&internal_node_child[locked], child, -2)) {
        // locked, child has the original value
        if (child == -1) { // if it is a null pointer, just insert the new body
          internal_node_child[locked] = i;
        } else {  // there already is a body in this position
          // there are no more than nbodiesd internal nodes
          patch = nbodiesd + 1;
          // create new cell(s) and insert the old and new body
          // comment: you can't do summarization at the same time as you insert new ones (really?)
          // comment: because, if you do so, you have traverse backward to the root (still doable?)

          // create new internal node

          // compute the center of this internal node

          // place the old node into the children of this new internal node
            // computer j
            // insert

          // computer the child position the new node should be in
            // see if it collides with the old node
            // if so, continue newing internal node and repeat the process
            // if not, insert and end this
          do {
            // depth++;

            // comment: the new cell is "allocated" from the bottom, according to the thesis
            // comment: remember that this is not actual allocation. it is like assignment 2/3
            // comment: cell is the index of the new cell

            // this is not like GPU CAS; it must will succeed
            // the retry mechanism is hidden from programmers
            // only return the old value, so we need to + 1
            int internal_node_idx = atomicAdd((int *)&internal_node_num, 1) + 1;
            // if (cell <= nbodiesd) {
            //   // comment: cells + bodies should be less than the size of the array
            //   // comment: if it is more than, they collide and error
            //   // *errd = 1;
            //   bottomd = nnodesd;
            // }

            // comment: why do we need patch? because we need to unlock the uppest level
            // comment: when the insertion completes
            // take the value of internal_node_idx only once (in the uppest level)
            // and that's why we have to internal_node_child[locked] = patch; after insertion
            
            // in the original solution, internal_node_idx descend from nnodesd
            // however, in the modified version, internal_node_idx ascend
            // so we use min instead of max; and the initial value of patch is changed accordingly

            patch = min(patch, internal_node_idx);

            int x = (j & 1) * r;
            int y = ((j >> 1) & 1) * r;
            r *= 0.5f;

            // comment: mass will be calculated in the summarization kernel later on?
            // comment: here we insert the new cell to the end of asd
            // comment: Initially, all cells have negative masses, indicating that their true masses still need to be computed.
            internal_node_mass[internal_node_idx] = -1.0f;
            // x and y is the temporary coordinate of the new internal node

            // internal_node_posy[n] is the x position of the parent node of the new internal node
            // internal_node_posx[n] - r + x; is adjusting the position
            x = internal_node_posx[internal_node_idx] = internal_node_posx[n] - r + x;
            y = internal_node_posy[internal_node_idx] = internal_node_posy[n] - r + y;
            // comment: first set all children to null pointer
            for (k = 0; k < 4; k++) 
              internal_node_child[internal_node_idx*4+k] = -1;

            // comment: if it is not the uppest level, we can add this immediately
            // comment: since it doesn't serve as a lock
            if (patch != internal_node_idx) {
              // the + nbodiesd is for differentiating leaf nodes and internal nodes
              internal_node_child[n*4+j] = internal_node_idx + nbodiesd;
            }

            // comment: put the old body in place
            j = 0;
            if (x < leaf_node_posx[child]) j = 1;
            if (y < leaf_node_posy[child]) j += 2;
            internal_node_child[internal_node_idx*4+j] = child;

            // comment: the child >= 0 condition is in case the old body and the new body collide in the newly allocated children
            n = internal_node_idx;
            j = 0;
            if (x < px) j = 1;
            if (y < py) j += 2;
            child = internal_node_child[n*4+j];
            // repeat until the two bodies are different children
          } while (child >= 0);
          // comment: when old body and new body are no colliding anymore, put the new body in place
          internal_node_child[n*4+j] = i;
          __threadfence();  // push out subtree
          // the + nbodiesd is for differentiating leaf nodes and internal nodes
          internal_node_child[locked] = patch + nbodiesd;
        }

        // localmaxdepth = max(depth, localmaxdepth);
        i += inc;  // move on to next body
        // comment: for another body, will have to start all over again
        // skip = 1;
      }
    }
    __syncthreads();  // throttle
  }
  // record maximum tree depth
  // atomicMax((int *)&maxdepthd, localmaxdepth);
}


__global__
void SummarizationKernel()
{
  // register int i, j, k, ch, inc, missing, cnt, bottom;
  // register float m, cm, px, py;
  // __shared__ volatile int child[THREADS3 * 4];

  // comment: traverse from bottom to nnodesd
  // comment: bottom-up searching
  int last_internal_node = internal_node_num;
  // comment: inc is the stride width of the cuda thread
  int inc = blockDim.x * gridDim.x;

  // k is the actual index of internal node it is responsible for (instead of `unique thread index`)
  // TODO this is still ambiguous
  int k = (last_internal_node & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k >= last_internal_node)
    k -= inc;

  // comment: at the start, no children is missing
  int missing = 0;
  // comment: notice that actions are conducted on cells
  // iterate over all cells assigned to thread
  while (k >= 0) {
    if (missing == 0) {
      // new cell, so initialize
      // comment: cm is short for cumulative mass
      float cm = 0.0f;
      // comment: the cumulative position x and y
      float px = 0.0f;
      float py = 0.0f;
      // // comment: cnt is for storing the number of all sub-node of this node
      // cnt = 0;
      // comment: j refers to the number of non-null-pointer children
      // j = 0;
      // comment: traverse its four children
      int mask = 0;

      for (int i = 0; i < 4; i++) {
        int child = internal_node_child[k*4+i];
        // comment: if this child is not null pointer (may be internal or leaf)
        if (child >= 0) {
          // comment: this happens when some children is found to be null pointer
          // commnet: because j is only incremented when a non-null-pointer children is found
          // comment: when they are not equal, j should always be smaller than i
          // if (i != j) {
          //   // comment: 
          //   // move children to front (needed later for speed)
          //   childd[k*4+i] = -1;
          //   childd[k*4+j] = ch;
          // }
          // child[missing*THREADS3+threadIdx.x] = ch;  // cache missing children
          float m = 0.f;
          if (child >= nbodiesd) {
            // it is a internal node
            m = internal_node_mass[child - nbodiesd];
          } else {
            // it is a leaf
            m = leaf_node_mass[child];
          }
          // float m = massd[ch];
          // comment: assume the mass of the child is not ready yet -> missing++
          missing++;
          if (m >= 0.0f) {
            // comment: if child is ready -> missing--
            missing--;
            // comment: if the computed child is a cell
            // if (ch >= nbodiesd) {  // count bodies (needed later)
            //   // comment: countd is for storing the number of sub-nodes of a node
            //   // comment: the storing can only be done only when all sub-nodes are computed
            //   cnt += countd[ch] - 1;
            // }
            // add child's contribution
            cm += m;
            if (child >= nbodiesd) {
              // it is a internal node
              px += internal_node_posx[child - nbodiesd] * m;
              py += internal_node_posy[child - nbodiesd] * m;
            } else {
              // it is a leaf node
              px += leaf_node_posx[child] * m;
              py += leaf_node_posy[child] * m;
            }
          } else {
            // add it the mask for next processing stage
            mask |= (1 << i);
          }
          // j++;
        }
      }
      // cnt += j;
    }

    // comment: some children are still not computed
    if (missing != 0) {
      // do {
      //   // poll missing child
      //   ch = child[(missing-1)*THREADS3+threadIdx.x];
      //   // comment: poll the mass
      //   m = massd[ch];
      //   if (m >= 0.0f) {
      //     // child is now ready
      //     missing--;
      //     if (ch >= nbodiesd) {
      //       // count bodies (needed later)
      //       cnt += countd[ch] - 1;
      //     }
      //     // add child's contribution
      //     cm += m;
      //     px += posxd[ch] * m;
      //     py += posyd[ch] * m;
      //   }
      //   // repeat until we are done or child is not ready
      // } while ((m >= 0.0f) && (missing != 0));

      for (i = 0; i < 4; i++) {
        // poll missing children
        if ((mask & (1 << i)) == 0) {
          // it was not missing
          continue;
        }

        int child = internal_node_child[k*4+i];
        if (child < 0) {
          // when this children is null pointer, skip it
          continue;
        } else {
          // if it is leaf node, ignore it because it was computed already
          if (child >= nbodiesd) {
            // solve the children one by one
            while (1) {
              // it is a internal node
              m = internal_node_mass[child];
              if (m < 0.0f) {
                // if the internal node is not ready; keep retrying
                continue;
              } else {
                // the internal is ready
                missing--;
                cm += m;
                px += internal_node_posx[child - nbodiesd] * m;
                py += internal_node_posy[child - nbodiesd] * m;
                break;
              }
            }
          }
        }
      }
    }

    if (missing == 0) {
      // all children are ready, so store computed information
      // countd[k] = cnt;
      m = 1.0f / cm;
      internal_node_posx[k] = px * m;
      internal_node_posy[k] = py * m;
      __threadfence();  // make sure data are visible before setting mass
      internal_node_mass[k] = cm;
      k -= inc;  // move on to next cell
    }
  }
}


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


__global__
__launch_bounds__(THREADS5, FACTOR5)
void ForceCalculationKernel()
{
  register int i, j, k, n, depth, base, sbase, diff;
  register float px, py, ax, ay, dx, dy, tmp;
  __shared__ volatile int pos[MAXDEPTH * THREADS5/WARPSIZE], node[MAXDEPTH * THREADS5/WARPSIZE];
  __shared__ volatile float dq[MAXDEPTH * THREADS5/WARPSIZE];
  __shared__ volatile int /*step, */maxdepth;

  if (0 == threadIdx.x) {
    // step = stepd;
    maxdepth = MAXDEPTH;
    tmp = radiusd;
    // precompute values that depend only on tree level
    dq[0] = tmp * tmp * itolsqd;
    for (i = 1; i < maxdepth; i++) {
      // 
      dq[i] = dq[i - 1] * 0.25f;
    }

    if (maxdepth > MAXDEPTH) {
      // *errd = maxdepth;
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
              tmp = massd[n] * tmp * tmp;
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

      // save computed acceleration
      accxd[i] = ax;
      // printf("accxd[i] = %f\n", accxd[i]);
      accyd[i] = ay;
      // printf("accyd[i] = %f\n", accyd[i]);
    }
  }
}


__global__
__launch_bounds__(THREADS6, FACTOR6)
void IntegrationKernel()
{
  register int i, inc;
  register float dvelx, dvely;
  register float velhx, velhy;

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


cudaBHSpaceModel::cudaBHSpaceModel(RectangleD bounds, std::vector<Object> &objects, Screen *screen)
    : SpaceModel(bounds, objects, screen) {
    this->tree = new QuadTree(this->bounds);
    if (this->tree == NULL) {
        printf("unable to initialize QuadTree in SpaceModel\n");
        return;
    }
}


void
cudaBHSpaceModel::update(GS_FLOAT dt) {
    size_t i;
#ifdef CONST_TIME
    dt = CONST_TIME;
#endif
    int blocks = 15; // number of multiprocessor, specific to K40m

    int nbodies = this->objects.size();
    int nnodes = nbodies * 2;

    if (nnodes < 1024*blocks)
      nnodes = 1024*blocks;
    while ((nnodes & (WARPSIZE-1)) != 0)
      nnodes++;
    nnodes--;

    float dtime, dthf, epssq, itolsq;
    dtime = dt;
    dthf = dtime * 0.5f;
    // EPSILON; for soothing
    epssq = 0.05 * 0.05;
    itolsq = 1.0f / (0.5 * 0.5);

    float *mass = (float *)malloc(sizeof(float) * nbodies);
    float *posx = (float *)malloc(sizeof(float) * nbodies);
    float *posy = (float *)malloc(sizeof(float) * nbodies);
    float *velx = (float *)malloc(sizeof(float) * nbodies);
    float *vely = (float *)malloc(sizeof(float) * nbodies);
    int *sortl, *childl, *countl, *startl;
    float *massl;
    float *posxl, *posyl;
    float *velxl, *velyl;
    float *accxl, *accyl;
    
    cudaMalloc((void **)&childl, sizeof(int) * (nnodes + 1) * 4);
    cudaMalloc((void **)&massl, sizeof(float) * (nnodes + 1));
    cudaMalloc((void **)&posxl, sizeof(float) * (nnodes + 1));
    cudaMalloc((void **)&posyl, sizeof(float) * (nnodes + 1));
    cudaMalloc((void **)&countl, sizeof(int) * (nnodes + 1));
    cudaMalloc((void **)&startl, sizeof(int) * (nnodes + 1));
    cudaMalloc((void **)&velxl, sizeof(int) * (nnodes + 1));
    cudaMalloc((void **)&velyl, sizeof(int) * (nnodes + 1));
    cudaMalloc((void **)&accxl, sizeof(int) * (nnodes + 1));
    cudaMalloc((void **)&accyl, sizeof(int) * (nnodes + 1));
    cudaMalloc((void **)&sortl, sizeof(int) * (nnodes + 1));

    // copy the address of the array to constant memory
    cudaMemcpyToSymbol(nnodesd, &nnodes, sizeof(int));
    cudaMemcpyToSymbol(nbodiesd, &nbodies, sizeof(int));
    cudaMemcpyToSymbol(dtimed, &dtime, sizeof(float));
    cudaMemcpyToSymbol(dthfd, &dthf, sizeof(float));
    cudaMemcpyToSymbol(epssqd, &epssq, sizeof(float));
    cudaMemcpyToSymbol(itolsqd, &itolsq, sizeof(float));
    cudaMemcpyToSymbol(sortd, &sortl, sizeof(void *));
    cudaMemcpyToSymbol(countd, &countl, sizeof(void *));
    cudaMemcpyToSymbol(startd, &startl, sizeof(void *));
    cudaMemcpyToSymbol(childd, &childl, sizeof(void *));

    for (i = 0; i < nbodies; i++) {
        posx[i] = this->objects[i].position.x;
        posy[i] = this->objects[i].position.y;
        velx[i] = this->objects[i].speed.x;
        vely[i] = this->objects[i].speed.y;
        mass[i] = this->objects[i].mass;
    }

    cudaMemcpy(massl, mass, sizeof(float) * nbodies, cudaMemcpyHostToDevice);
    cudaMemcpy(posxl, posx, sizeof(float) * nbodies, cudaMemcpyHostToDevice);
    cudaMemcpy(posyl, posy, sizeof(float) * nbodies, cudaMemcpyHostToDevice);
    cudaMemcpy(velxl, velx, sizeof(float) * nbodies, cudaMemcpyHostToDevice);
    cudaMemcpy(velyl, vely, sizeof(float) * nbodies, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(massd, &massl, sizeof(void *));
    cudaMemcpyToSymbol(posxd, &posxl, sizeof(void *));
    cudaMemcpyToSymbol(posyd, &posyl, sizeof(void *));
    cudaMemcpyToSymbol(velxd, &velxl, sizeof(void *));
    cudaMemcpyToSymbol(velyd, &velyl, sizeof(void *));
    cudaMemcpyToSymbol(accxd, &accxl, sizeof(void *));
    cudaMemcpyToSymbol(accyd, &accyl, sizeof(void *));

    InitializationKernel <<< 1, 1>>>();
    TreeBuildingKernel <<< blocks *FACTOR2, THREADS2>>>();
    SummarizationKernel <<< blocks *FACTOR3, THREADS3>>>();
    SortKernel <<< blocks *FACTOR4, THREADS4>>>();
    ForceCalculationKernel <<< blocks *FACTOR5, THREADS5>>>();
    IntegrationKernel <<< blocks *FACTOR6, THREADS6>>>();

    cudaMemcpy(posx, posxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost);
    cudaMemcpy(posy, posyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost);
    cudaMemcpy(velx, velxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost);
    cudaMemcpy(vely, velyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost);

    // update posx, posy, velx, vely to this->objects array
    for (i = 0; i < nbodies; i++) {
        this->objects[i].position.x = posx[i];
        this->objects[i].position.y = posy[i];
        this->objects[i].speed.x = velx[i];
        this->objects[i].speed.y = vely[i];
        this->objects[i].mass = mass[i];
    }

    remove_objects_outside_bounds();

    // delete this->tree;
    // this->tree = new QuadTree(this->bounds);
    // this->tree->add_objects(this->objects);

    free(mass);
    free(posx);
    free(posy);
    free(velx);
    free(vely);

    cudaFree(childl);
    cudaFree(massl);
    cudaFree(posxl);
    cudaFree(posyl);
    cudaFree(countl);
    cudaFree(startl);
    cudaFree(velxl);
    cudaFree(velyl);
    cudaFree(accxl);
    cudaFree(accyl);
    cudaFree(sortl);
}

cudaBHSpaceModel::~cudaBHSpaceModel() {
    delete this->tree;
}
