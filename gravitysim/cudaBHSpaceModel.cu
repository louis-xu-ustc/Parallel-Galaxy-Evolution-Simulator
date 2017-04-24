#include <stdio.h>
#include "cuda.h"
#include "cudaBHSpaceModel.h"


#define NULL_BODY (-1)
#define LOCK (-2)

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

// data on the device
__constant__ int nnodesd, nbodiesd;
__constant__ float dtimed, dthfd, epssqd, itolsqd;
__constant__ volatile float *massd, *posxd, *posyd, *velxd, *velyd, *accxd, *accyd;
__constant__ volatile float *maxxd, *maxyd, *minxd, *minyd;
__constant__ volatile int *errd, *sortd, *childd, *countd, *startd;

__device__ volatile int stepd, bottomd, maxdepthd, blkcntd;
__device__ volatile float radiusd;


__global__ void InitializationKernel() {
    *errd = 0;
    stepd = -1;
    maxdepthd = 1;
    blkcntd = 0;
}

__global__
__launch_bounds__(THREADS2, FACTOR2)
void TreeBuildingKernel()
{
  register int i, j, k, depth, localmaxdepth, skip, inc;
  register float x, y, r;
  register float px, py;
  register int ch, n, cell, locked, patch;
  register float radius, rootx, rooty;

  // cache root data
  radius = radiusd;
  // nnodesd is the number of 
  rootx = posxd[nnodesd];
  rooty = posyd[nnodesd];

  // comments: why need recording depth at all??
  localmaxdepth = 1;
  skip = 1;
  inc = blockDim.x * gridDim.x;
  i = threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all bodies assigned to thread
  // comment: a simple comparison of the array index with the number of bodies determines whether the index points to a cell or a body
  // comment: since i is the unique thread id, if not properly scheduled, many thread resources may be wasted
  // comment: this one looks strange
  while (i < nbodiesd) {
    if (skip != 0) {
      // comment: do only once when the thread try this body for the first time
      // comment: when retrying, we don't want to start ahead again
      // comment: instead, we want to start from where we left off
      // new body, so start traversing at root
      skip = 0;
      px = posxd[i];
      py = posyd[i];
      n = nnodesd;
      // comment: the depth 
      depth = 1;
      r = radius;
      // comment: j is the index of children (of a node)
      j = 0;
      // determine which child to follow
      // comment: left up - 2; right up - 3; left down - 0; right down - 1
      if (rootx < px) j = 1;
      if (rooty < py) j += 2;
    }

    // follow path to leaf cell
    // comment: childd is a specialized array for storing the children nodes
    // comment: since every node can have at most 4 children, we allocate the most space for them
    // comment: this is for convenience of traversal
    // comment: ch is the value of that child node, possibly pointing to another node, possibly just -1 (null pointer)
    // comment: if didin't skip, this should point to children of root node
    // comment: if did skip, it probably point to something else
    ch = childd[n*4+j];
    // comment: in some code sections, we need to find out whether an index refers to a body or to null. Because −1 is also smaller than the number of bodies, a single integer comparison suffices to test both conditions.
    // comment: if ch >= nbodiesd, the child already points to a cell, then go find some place in its children
    while (ch >= nbodiesd) {
      n = ch;
      // find it in the next level
      depth++;
      r *= 0.5f;
      // comment: reset j
      j = 0;
      // determine which child to follow
      if (posxd[n] < px) j = 1;
      if (posyd[n] < py) j += 2;
      ch = childd[n*4+j];
    }
    // comment: finally found a null pointer (-1 <= nbodiesd) or a body (since the id of bodies is <= nbodiesd)
    // comment: this two situations are reflected below

    // skip if child pointer is locked and try again later
    if (ch != -2) {
      // comment: if it is not locked, try to lock it
      locked = n*4+j;
      // try to lock. However, it is still possible to fail because of contention
      // comments: if fail, also try again later
      if (ch == atomicCAS((int *)&childd[locked], ch, -2)) {
        if (ch == -1) { // if it is a null pointer, just insert the new body
          childd[locked] = i;
        } else {  // there already is a body in this position
          patch = -1;
          // create new cell(s) and insert the old and new body
          // comment: you can't do summarization at the same time as you insert new ones (really?)
          // comment: because, if you do so, you have traverse backward to the root (still doable?)
          do {
            depth++;

            // comment: the new cell is "allocated" from the bottom, according to the thesis
            // comment: remember that this is not actual allocation. it is like assignment 2/3
            // comment: cell is the index of the new cell
            cell = atomicSub((int *)&bottomd, 1) - 1;
            if (cell <= nbodiesd) {
              // comment: cells + bodies should be less than the size of the array
              // comment: if it is more than, they collide and error
              *errd = 1;
              bottomd = nnodesd;
            }
            // comment: why do we need patch?????????????
            patch = max(patch, cell);

            x = (j & 1) * r;
            y = ((j >> 1) & 1) * r;
            r *= 0.5f;

            // comment: mass will be calculated in the summarization kernel later on?
            // comment: here we insert the new cell to the end of asd
            // comment: Initially, all cells have negative masses, indicating that their true masses still need to be computed.
            massd[cell] = -1.0f;
            startd[cell] = -1;
            x = posxd[cell] = posxd[n] - r + x;
            y = posyd[cell] = posyd[n] - r + y;
            // comment: first set all children to null pointer
            for (k = 0; k < 4; k++) 
              childd[cell*4+k] = -1;

            // comment: ???????????????
            if (patch != cell) {
              childd[n*4+j] = cell;
            }

            // comment: put the old body in place
            j = 0;
            if (x < posxd[ch]) j = 1;
            if (y < posyd[ch]) j += 2;
            childd[cell*4+j] = ch;

            // comment: the ch >= 0 condition is in case the old body and the new body collide in the newly allocated children
            n = cell;
            j = 0;
            if (x < px) j = 1;
            if (y < py) j += 2;
            ch = childd[n*4+j];
            // repeat until the two bodies are different children
          } while (ch >= 0);
          // comment: when old body and new body are no colliding anymore, put the new body in place
          childd[n*4+j] = i;
          __threadfence();  // push out subtree
          // comment: ???????????????
          childd[locked] = patch;
        }

        localmaxdepth = max(depth, localmaxdepth);
        i += inc;  // move on to next body
        // comment: for another body, will have to start all over again
        skip = 1;
      }
    }
    __syncthreads();  // throttle
  }
  // record maximum tree depth
  atomicMax((int *)&maxdepthd, localmaxdepth);
}


__global__
void BoundingBoxKernel()
{
  register int i, j, k, inc;
  register float val, minx, maxx, miny, maxy;
  __shared__ volatile float sminx[THREADS1], smaxx[THREADS1], sminy[THREADS1], smaxy[THREADS1];

  // initialize with valid data (in case #bodies < #threads)
  minx = maxx = posxd[0];
  miny = maxy = posyd[0];

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
  }

  // reduction in shared memory
  sminx[i] = minx;
  smaxx[i] = maxx;
  sminy[i] = miny;
  smaxy[i] = maxy;

  for (j = THREADS1 / 2; j > 0; j /= 2) {
    __syncthreads();
    if (i < j) {
      k = i + j;
      sminx[i] = minx = min(minx, sminx[k]);
      smaxx[i] = maxx = max(maxx, smaxx[k]);
      sminy[i] = miny = min(miny, sminy[k]);
      smaxy[i] = maxy = max(maxy, smaxy[k]);
    }
  }

  // write block result to global memory
  if (i == 0) {
    k = blockIdx.x;
    minxd[k] = minx;
    maxxd[k] = maxx;
    minyd[k] = miny;
    maxyd[k] = maxy;

    inc = gridDim.x - 1;
    if (inc == atomicInc((unsigned int *)&blkcntd, inc)) {
      // I'm the last block, so combine all block results
      for (j = 0; j <= inc; j++) {
        minx = min(minx, minxd[j]);
        maxx = max(maxx, maxxd[j]);
        miny = min(miny, minyd[j]);
        maxy = max(maxy, maxyd[j]);
      }

      // compute 'radius'
      radiusd = max(maxx - minx, maxy - miny) * 0.5;

      // create root node
      k = nnodesd;
      bottomd = k;

      massd[k] = -1.0f;
      startd[k] = 0;
      posxd[k] = (minx + maxx) * 0.5f;
      posyd[k] = (miny + maxy) * 0.5f;
      k *= 4;
      for (i = 0; i < 4; i++) 
        childd[k + i] = -1;

      stepd++;
    }
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
__launch_bounds__(THREADS3, FACTOR3)
void SummarizationKernel()
{
  register int i, j, k, ch, inc, missing, cnt, bottom;
  register float m, cm, px, py;
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
      __threadfence();  // make sure data are visible before setting mass
      massd[k] = cm;
      k += inc;  // move on to next cell
    }
  }
}


cudaBHSpaceModel::cudaBHSpaceModel(RectangleD bounds, std::vector<Object> &objects)
    : SpaceModel(bounds, objects) {
    printf("initializing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    this->tree = new QuadTree(this->bounds);
    if (this->tree == NULL) {
        printf("unable to initialize QuadTree in SpaceModel\n");
        return;
    }
}


void
cudaBHSpaceModel::update(GS_FLOAT dt) {
    printf("1111111111111111\n");
    size_t i;
#ifdef CONST_TIME
    dt = CONST_TIME;
#endif
    // this->tree->apply_to_objects(this->objects, dt);
    // for (i = 0; i < objects.size(); i++) {
    //     objects[i].update_position(dt);
    // }
    // remove_objects_outside_bounds();
    // delete this->tree;
    // this->tree = new QuadTree(this->bounds);
    // this->tree->add_objects(this->objects);

    cudaFuncSetCacheConfig(BoundingBoxKernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(TreeBuildingKernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(SummarizationKernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(SortKernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(ForceCalculationKernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(IntegrationKernel, cudaFuncCachePreferL1);


    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, 0);
    // int blocks = deviceProp.multiProcessorCount;
    int blocks = 15;

    int error; // for accepting error code from device
    int nbodies = this->objects.size();
    int nnodes = nbodies * 2;

    if (nnodes < 1024*blocks)
      nnodes = 1024*blocks;
    while ((nnodes & (WARPSIZE-1)) != 0) 
      nnodes++;
    nnodes--;
    printf("2222222222222\n");

    float dtime, dthf, epssq, itolsq;
    dtime = 0.025;
    dthf = dtime * 0.5f;
    epssq = 0.05 * 0.05;
    itolsq = 1.0f / (0.5 * 0.5);

    float *mass = (float *)malloc(sizeof(float) * nbodies);
    float *posx = (float *)malloc(sizeof(float) * nbodies);
    float *posy = (float *)malloc(sizeof(float) * nbodies);
    float *velx = (float *)malloc(sizeof(float) * nbodies);
    float *vely = (float *)malloc(sizeof(float) * nbodies);
    int *errl, *sortl, *childl, *countl, *startl;
    float *massl;
    float *posxl, *posyl;
    float *velxl, *velyl;
    float *accxl, *accyl;
    float *maxxl, *maxyl;
    float *minxl, *minyl;
    printf("333333333333333333\n");

    printf("333333 zhihou \n");
    cudaMalloc((void **)&errl, sizeof(int));
    cudaMalloc((void **)&childl, sizeof(int) * (nnodes + 1) * 4);
    cudaMalloc((void **)&massl, sizeof(float) * (nnodes + 1));
    cudaMalloc((void **)&posxl, sizeof(float) * (nnodes + 1));
    cudaMalloc((void **)&posyl, sizeof(float) * (nnodes + 1));
    cudaMalloc((void **)&countl, sizeof(int) * (nnodes + 1));
    cudaMalloc((void **)&startl, sizeof(int) * (nnodes + 1));
    printf("4444444444444444\n");

    // ?????
    int inc = (nbodies + WARPSIZE - 1) & (-WARPSIZE);
    velxl = (float *)&childl[0 * inc];
    velyl = (float *)&childl[1 * inc];
    accxl = (float *)&childl[2 * inc];
    accyl = (float *)&childl[3 * inc];
    sortl = (int *)&childl[4 * inc];

    cudaMalloc((void **)&maxxl, sizeof(float) * blocks);
    cudaMalloc((void **)&maxyl, sizeof(float) * blocks);
    cudaMalloc((void **)&minxl, sizeof(float) * blocks);
    cudaMalloc((void **)&minyl, sizeof(float) * blocks);
    printf("555555555555555555555\n");

    cudaMemcpyToSymbol(nnodesd, &nnodes, sizeof(int));
    cudaMemcpyToSymbol(nbodiesd, &nbodies, sizeof(int));
    cudaMemcpyToSymbol(errd, &errl, sizeof(void *));
    cudaMemcpyToSymbol(dtimed, &dtime, sizeof(float));
    cudaMemcpyToSymbol(dthfd, &dthf, sizeof(float));
    cudaMemcpyToSymbol(epssqd, &epssq, sizeof(float));
    cudaMemcpyToSymbol(itolsqd, &itolsq, sizeof(float));
    cudaMemcpyToSymbol(sortd, &sortl, sizeof(void *));
    cudaMemcpyToSymbol(countd, &countl, sizeof(void *));
    cudaMemcpyToSymbol(startd, &startl, sizeof(void *));
    cudaMemcpyToSymbol(childd, &childl, sizeof(void *));
    cudaMemcpyToSymbol(maxxd, &maxxl, sizeof(void*));
    cudaMemcpyToSymbol(maxyd, &maxyl, sizeof(void*));
    cudaMemcpyToSymbol(minxd, &minxl, sizeof(void*));
    cudaMemcpyToSymbol(minyd, &minyl, sizeof(void*));
    printf("66666666666666666666\n");

    for (i = 0; i < nbodies; i++) {
        posx[i] = this->objects[i].position.x;
        posy[i] = this->objects[i].position.y;
        velx[i] = this->objects[i].speed.x;
        vely[i] = this->objects[i].speed.y;
        mass[i] = this->objects[i].mass;
    }
    printf("7777777777777777777777\n");

    cudaMemcpy(massl, mass, sizeof(float) * nbodies, cudaMemcpyHostToDevice);
    cudaMemcpy(posxl, posx, sizeof(float) * nbodies, cudaMemcpyHostToDevice);
    cudaMemcpy(posyl, posy, sizeof(float) * nbodies, cudaMemcpyHostToDevice);
    cudaMemcpy(velxl, velx, sizeof(float) * nbodies, cudaMemcpyHostToDevice);
    cudaMemcpy(velyl, vely, sizeof(float) * nbodies, cudaMemcpyHostToDevice);
    printf("888888888888888888\n");

    cudaMemcpyToSymbol(massd, &massl, sizeof(void *));
    cudaMemcpyToSymbol(posxd, &posxl, sizeof(void *));
    cudaMemcpyToSymbol(posyd, &posyl, sizeof(void *));
    cudaMemcpyToSymbol(velxd, &velxl, sizeof(void *));
    cudaMemcpyToSymbol(velyd, &velyl, sizeof(void *));
    cudaMemcpyToSymbol(accxd, &accxl, sizeof(void *));
    cudaMemcpyToSymbol(accyd, &accyl, sizeof(void *));
    printf("999999999999999999\n");

    InitializationKernel <<< 1, 1>>>();
    BoundingBoxKernel<<<blocks * FACTOR1, THREADS1>>>();
    TreeBuildingKernel <<< blocks *FACTOR2, THREADS2>>>();
    SummarizationKernel <<< blocks *FACTOR3, THREADS3>>>();
    SortKernel <<< blocks *FACTOR4, THREADS4>>>();
    ForceCalculationKernel <<< blocks *FACTOR5, THREADS5>>>();
    IntegrationKernel <<< blocks *FACTOR6, THREADS6>>>();
    printf("10101010101010110\n");

    cudaMemcpy(&error, errl, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(posx, posxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost);
    cudaMemcpy(posy, posyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost);
    cudaMemcpy(velx, velxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost);
    cudaMemcpy(vely, velyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost);
    printf("111100001111000011110000\n");

    // update posx, posy, velx, vely to this->objects array
    for (i = 0; i < nbodies; i++) {
        this->objects[i].position.x = posx[i];
        this->objects[i].position.y = posy[i];
        this->objects[i].speed.x = velx[i];
        this->objects[i].speed.y = vely[i];
        this->objects[i].mass = mass[i];
    }
    printf("2333333333333333333\n");

    remove_objects_outside_bounds();

    delete this->tree;
    this->tree = new QuadTree(this->bounds);
    this->tree->add_objects(this->objects);

    free(mass);
    free(posx);
    free(posy);
    free(velx);
    free(vely);

    cudaFree(errl);
    cudaFree(childl);
    cudaFree(massl);
    cudaFree(posxl);
    cudaFree(posyl);
    cudaFree(countl);
    cudaFree(startl);

    cudaFree(maxxl);
    cudaFree(maxyl);
    cudaFree(minxl);
    cudaFree(minyl);
}

cudaBHSpaceModel::~cudaBHSpaceModel() {
    delete this->tree;
}
