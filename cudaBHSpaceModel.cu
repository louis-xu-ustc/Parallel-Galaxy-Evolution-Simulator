#include <stdio.h>
#include <cmath>
#include "cuda.h"
#include "cutil_math.h"
#include "cudaBHSpaceModel.h"
#include "build_config.h"


#define NULL_BODY (-1)
#define LOCK (-2)

//#define printf(...) 

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

// nbodiesd -> number of leaf nodes
__constant__ int nbodiesd;
// dtimed -> update interval
__constant__ float dtimed, dthfd, epssqd, itolsqd;
// leaf node information, transfer from CPU to GPU
__constant__ volatile float *leaf_node_mass, *leaf_node_posx, *leaf_node_posy, *leaf_node_velx, *leaf_node_vely, *leaf_node_accx, *leaf_node_accy;
// internal node information, generated in GPU, should be cudaMalloc-ed
__device__ volatile int32_t internal_node_num;    // <= nbodiesd
__constant__ volatile int *internal_node_child;   // 4 x nbodiesd
__constant__ volatile float *internal_node_mass;    // nbodiesd
__constant__ volatile float *internal_node_posx;    // nbodiesd
__constant__ volatile float *internal_node_posy;    // nbodiesd
__device__ volatile float radiusd;


__global__
void InitializationKernel() {
    // the below steps are originally done in another kernel
    radiusd = 700.f;
    int k = 0; // root node placed at first place
    for (int i = 0; i < 4; i++)
      internal_node_child[4 * k + i] = NULL_BODY;
    internal_node_mass[k] = -1.0f;
    internal_node_posx[k] = 500.f;
    internal_node_posy[k] = 300.f;
    internal_node_num = 0;  // internel_node_num refers to the index of the last internal node
}


__global__
void TreeBuildingKernel()
{
  float radius = radiusd;
  float rootx = internal_node_posx[0];
  float rooty = internal_node_posy[0];
  int inc = blockDim.x * gridDim.x;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < nbodiesd) {
    float px = leaf_node_posx[i];
    float py = leaf_node_posy[i];
    int n = 0;  // root
    float r = radius;
    int j = 0;
    if (rootx < px) j = 1;
    if (rooty < py) j += 2;
    int child = internal_node_child[n*4+j];
    while (child >= nbodiesd) {
      n = child - nbodiesd;
      r *= 0.5f;
      j = 0;
      if (internal_node_posx[n] < px) j = 1;
      if (internal_node_posy[n] < py) j += 2;
      child = internal_node_child[n*4+j];
    }

    if (child != LOCK) {
      int locked = n*4+j;
      if (child == atomicCAS((int *)&internal_node_child[locked], child, LOCK)) {
        if (child == NULL_BODY) { // if it is a null pointer, just insert the new body
          internal_node_child[locked] = i;
        } else {  // there already is a body in this position
          int patch = 999999999;
          do {
            int internal_node_idx = atomicAdd((int *)&internal_node_num, 1) + 1;
            patch = min(patch, internal_node_idx);

            float x = (j & 1) * r;
            float y = ((j >> 1) & 1) * r;
            r *= 0.5f;
            internal_node_mass[internal_node_idx] = -1.0f;
            x = internal_node_posx[internal_node_idx] = internal_node_posx[n] - r + x;
            y = internal_node_posy[internal_node_idx] = internal_node_posy[n] - r + y;
            for (int k = 0; k < 4; k++) {
              internal_node_child[internal_node_idx*4+k] = NULL_BODY;
            }
            if (patch != internal_node_idx) {
              internal_node_child[n*4+j] = internal_node_idx + nbodiesd;
            }

            j = 0;
            if (x < leaf_node_posx[child]) j = 1;
            if (y < leaf_node_posy[child]) j += 2;
            internal_node_child[internal_node_idx*4+j] = child;
            n = internal_node_idx;
            j = 0;
            if (x < px) j = 1;
            if (y < py) j += 2;
            child = internal_node_child[n*4+j];
          } while (child >= 0);
          internal_node_child[n*4+j] = i;
          __threadfence();  // push out subtree
          internal_node_child[locked] = patch + nbodiesd;
        }
        i += inc;  // move on to next body
      }
    }
    __syncthreads();  // throttle
  }
}


__global__
void SummarizationKernel()
{
  int last_internal_node = internal_node_num;
  int inc = blockDim.x * gridDim.x;

  int k = last_internal_node - (threadIdx.x + blockIdx.x * blockDim.x);
  int missing = 0;
  while (k >= 0) {
    printf("this is thread %d\n", k);
    int mask = 0;
    float cm = 0.0f;
    float px = 0.0f;
    float py = 0.0f;
    float m;

    if (missing == 0) {
      for (int i = 0; i < 4; i++) {
        int child = internal_node_child[k*4+i];
        // comment: if this child is not null pointer (may be internal or leaf)
        if (child >= 0) {
	  float m = 0.f;
          if (child >= nbodiesd) {
            // it is a internal node
            m = internal_node_mass[child - nbodiesd];
          } else {
            // it is a leaf
            m = leaf_node_mass[child];
          }
          missing++;
          if (m >= 0.0f) {
            missing--;
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
            mask |= (1 << i);
          }
        }
      }
    }

    if (missing != 0) {
        for (int i = 0; i < 4; i++) {
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
	      printf("I can't get out\n");
              // it is an internal node
              m = internal_node_mass[child - nbodiesd];
              if (m < 0.0f) {
                // if the internal node is not ready; keep retrying
                printf("waiting for it\n");
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

    printf("missing = %d\n", missing);

    if (missing == 0) {
      // all children are ready, so store computed information
      // countd[k] = cnt;
      float m = 1.0f / cm;
      internal_node_posx[k] = px * m;
      internal_node_posy[k] = py * m;
      __threadfence();  // make sure data are visible before setting mass
      internal_node_mass[k] = cm;
      k -= inc;  // move on to next cell
    } else {
      printf("it should be missing = 0, dude!\n");
    }
    __syncthreads();
  }
}


__device__
float2 CalculateForceOnLeafNode(int leaf_node, int depth, int target_node) {
  // notice: float2 overloads + (part of cuda runtime)
  // depth is for calculating region size (radiusd / (2^depth)); the depth of root is 0
  // target_node is the index in internel_node_child

  // when target_node < 0, this is a null children
  if (target_node < 0) {
    // it is null children
    return make_float2(0.f, 0.f);
  }

  float ax = 0.f, ay = 0.f;
  // if the target_node is a leaf node, simply calculate the force and return
  if (target_node >= 0 && target_node < nbodiesd) {
    float px = leaf_node_posx[leaf_node];
    float py = leaf_node_posy[leaf_node];

    float dx = leaf_node_posx[target_node] - px;
    float dy = leaf_node_posy[target_node] - py;

    float distance = dx*dx + dy*dy + epssqd;
    distance = rsqrtf(distance);
    distance = leaf_node_mass[target_node] * distance * distance;
    ax += dx * distance;
    ay += dy * distance;
    return make_float2(ax, ay);
  }

  // otherwise, calculate s/d, where s is the size of the region (of the target_node) 
  // and d is the actual distance
  target_node -= nbodiesd;   // conversion
  float s = radiusd / std::pow(2, depth);
  float distance = 0.f;
  float px = leaf_node_posx[leaf_node];
  float py = leaf_node_posy[leaf_node];
  float dx = internal_node_posx[target_node] - px;
  float dy = internal_node_posy[target_node] - py;
  distance = dx*dx + dy*dy + epssqd;  // d is the actual distance
  distance = rsqrtf(distance);
  distance = internal_node_mass[target_node] * distance * distance;

  // if s/d < ? (SD_TRESHOLD), see the internal node as an object, calculate the force and return
  if ((s/distance) < SD_TRESHOLD) {
    ax += dx * distance;
    ay += dy * distance;
    return make_float2(ax, ay);
  } else {
    // if s/d >= ?, do the above recursively on every child of the target_node
    return CalculateForceOnLeafNode(leaf_node, depth+1, internal_node_child[target_node*4]) + \
    CalculateForceOnLeafNode(leaf_node, depth+1, internal_node_child[target_node*4 + 1]) +    \
    CalculateForceOnLeafNode(leaf_node, depth+1, internal_node_child[target_node*4 + 2]) +    \
    CalculateForceOnLeafNode(leaf_node, depth+1, internal_node_child[target_node*4 + 3]);
  }

  printf("you shouldn't be here\n");
  return make_float2(0.f, 0.f);
}

__global__
void ForceCalculationKernel()
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nbodiesd; i += blockDim.x * gridDim.x) {
    // (target_node = nbodiesd) for signifying root node
    float2 acceleration = CalculateForceOnLeafNode(i, 0, nbodiesd);
    float acccx = acceleration.x;
    float acccy = acceleration.y;
    leaf_node_accx[i] = acccx;
    leaf_node_accy[i] = acccy;
  }
}


__global__
void IntegrationKernel()
{
  register int i, inc;
  register float dvelx, dvely;
  register float velhx, velhy;

  // iterate over all bodies assigned to thread
  inc = blockDim.x * gridDim.x;
  for (i = threadIdx.x + blockIdx.x * blockDim.x; i < nbodiesd; i += inc) {
    // integrate
    dvelx = leaf_node_accx[i] * dthfd;
    dvely = leaf_node_accy[i] * dthfd;

    velhx = leaf_node_velx[i] + dvelx;
    velhy = leaf_node_vely[i] + dvely;

    leaf_node_posx[i] += velhx * dtimed;
    leaf_node_posy[i] += velhy * dtimed;

    leaf_node_velx[i] = velhx + dvelx;
    leaf_node_vely[i] = velhy + dvely;
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

    // for segregate information of objects into different array
    // on the host
    float *mass = (float *)malloc(sizeof(float) * nbodies);
    float *posx = (float *)malloc(sizeof(float) * nbodies);
    float *posy = (float *)malloc(sizeof(float) * nbodies);
    float *velx = (float *)malloc(sizeof(float) * nbodies);
    float *vely = (float *)malloc(sizeof(float) * nbodies);
    for (i = 0; i < nbodies; i++) {
        posx[i] = this->objects[i].position.x;
        posy[i] = this->objects[i].position.y;
        velx[i] = this->objects[i].speed.x;
        vely[i] = this->objects[i].speed.y;
        mass[i] = this->objects[i].mass;
    }

    // on the device
    float *massl, *posxl, *posyl, *velxl, *velyl;
    cudaMalloc((void **)&massl, sizeof(float) * (nbodies + 1));
    cudaMalloc((void **)&posxl, sizeof(float) * (nbodies + 1));
    cudaMalloc((void **)&posyl, sizeof(float) * (nbodies + 1));
    cudaMalloc((void **)&velxl, sizeof(float) * (nbodies + 1));
    cudaMalloc((void **)&velyl, sizeof(float) * (nbodies + 1));

    // copy from host to device
    cudaMemcpy(massl, mass, sizeof(float) * nbodies, cudaMemcpyHostToDevice);
    cudaMemcpy(posxl, posx, sizeof(float) * nbodies, cudaMemcpyHostToDevice);
    cudaMemcpy(posyl, posy, sizeof(float) * nbodies, cudaMemcpyHostToDevice);
    cudaMemcpy(velxl, velx, sizeof(float) * nbodies, cudaMemcpyHostToDevice);
    cudaMemcpy(velyl, vely, sizeof(float) * nbodies, cudaMemcpyHostToDevice);

    // copy from __device__ to __constant__ for acceleration
    cudaMemcpyToSymbol(leaf_node_mass, &massl, sizeof(void *));
    cudaMemcpyToSymbol(leaf_node_posx, &posxl, sizeof(void *));
    cudaMemcpyToSymbol(leaf_node_posy, &posyl, sizeof(void *));
    cudaMemcpyToSymbol(leaf_node_velx, &velxl, sizeof(void *));
    cudaMemcpyToSymbol(leaf_node_vely, &velyl, sizeof(void *));

    // allocate leaf node acceleration information
    float *leaf_node_accxl, *leaf_node_accyl;
    cudaMalloc((void **)&leaf_node_accxl, sizeof(float) * (nbodies + 1));
    cudaMalloc((void **)&leaf_node_accyl, sizeof(float) * (nbodies + 1));

    // copy to constant memory
    cudaMemcpyToSymbol(leaf_node_accx, &leaf_node_accxl, sizeof(void *));
    cudaMemcpyToSymbol(leaf_node_accy, &leaf_node_accyl, sizeof(void *));

    printf("before allocate space for internal nodes\n");
    // allocate space for internal nodes
    // on the device
    int *internal_node_childl;
    float *internal_node_massl, *internal_node_posxl, *internal_node_posyl; 
    cudaMalloc((void**)&internal_node_childl, sizeof(int) * (nbodies + 1) * 4);
    cudaMalloc((void**)&internal_node_massl, sizeof(float) * (nbodies + 1));
    cudaMalloc((void**)&internal_node_posxl, sizeof(float) * (nbodies + 1));
    cudaMalloc((void**)&internal_node_posyl, sizeof(float) * (nbodies + 1));

    // copy to constant memory
    cudaMemcpyToSymbol(internal_node_child, &internal_node_childl, sizeof(void *));
    cudaMemcpyToSymbol(internal_node_mass, &internal_node_massl, sizeof(void *));
    cudaMemcpyToSymbol(internal_node_posx, &internal_node_posxl, sizeof(void *));
    cudaMemcpyToSymbol(internal_node_posy, &internal_node_posyl, sizeof(void *));

    // calculation factors
    float dtime, dthf, epssq, itolsq;
    dtime = dt;
    dthf = dtime * 0.5f;
    epssq = 0.05 * 0.05;  // EPSILON; for soothing
    itolsq = 1.0f / (0.5 * 0.5);

    printf("before copy the address of the array to constant memory\n");
    // copy the address of the array to constant memory
    cudaMemcpyToSymbol(nbodiesd, &nbodies, sizeof(int));
    cudaMemcpyToSymbol(dtimed, &dtime, sizeof(float));
    cudaMemcpyToSymbol(dthfd, &dthf, sizeof(float));
    cudaMemcpyToSymbol(epssqd, &epssq, sizeof(float));
    cudaMemcpyToSymbol(itolsqd, &itolsq, sizeof(float));

    printf("before InitializationKernel\n");
    InitializationKernel <<< 1, 1>>>();
    printf("before TreeBuildingKernel\n");
    TreeBuildingKernel <<< blocks * FACTOR3, THREADS3>>>();
    printf("before SummarizationKernel\n");
    SummarizationKernel <<< 1, THREADS4>>>();
    printf("before ForceCalculationKernel\n");
    ForceCalculationKernel <<< blocks * FACTOR5, THREADS5>>>();
    printf("before IntegrationKernel\n");
    IntegrationKernel <<< blocks * FACTOR6, THREADS6>>>();
    printf("after IntegrationKernel\n");

    // we only need to copy these four back into host
    cudaMemcpy(posx, posxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost);
    cudaMemcpy(posy, posyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost);
    cudaMemcpy(velx, velxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost);
    cudaMemcpy(vely, velyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost);

    // update leaf_node_posx, leaf_node_posy, leaf_node_velx, leaf_node_vely to objects array
    for (i = 0; i < nbodies; i++) {
    	float posxx = posx[i];
    	float posyy = posy[i];
    	float velxx = velx[i];
    	float velyy = vely[i];
        this->objects[i].position.x = posxx;
        this->objects[i].position.y = posyy;
        this->objects[i].speed.x = velxx;
        this->objects[i].speed.y = velyy;
    }

    remove_objects_outside_bounds();

    free(mass);
    free(posx);
    free(posy);
    free(velx);
    free(vely);

    cudaFree(massl);
    cudaFree(posxl);
    cudaFree(posyl);
    cudaFree(velxl);
    cudaFree(velyl);
    cudaFree(leaf_node_accxl);
    cudaFree(leaf_node_accyl);
    cudaFree(internal_node_childl);
    cudaFree(internal_node_massl);
    cudaFree(internal_node_posxl);
    cudaFree(internal_node_posyl);
}

cudaBHSpaceModel::~cudaBHSpaceModel() {
    delete this->tree;
}

