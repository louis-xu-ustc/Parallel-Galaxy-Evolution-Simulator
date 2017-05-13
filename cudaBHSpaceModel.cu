#include <stdio.h>
#include <iostream>
#include <cmath>
#include <assert.h>
#include "cuda.h"
#include "cudaBHSpaceModel.h"
#include "build_config.h"


#define NULL_BODY (-1)
#define LOCK (-2)

//#define printf(...)

// nbodiesd -> number of leaf nodes
__constant__ int nbodiesd;
// dtimed -> update interval
__constant__ float dtimed, dthfd, epssqd, itolsqd;
// leaf node information, transfer from CPU to GPU
__constant__ volatile float *leaf_node_mass, *leaf_node_posx, *leaf_node_posy;
__constant__ volatile float *leaf_node_velx, *leaf_node_vely, *leaf_node_accx, *leaf_node_accy;
// internal node information, generated in GPU, should be cudaMalloc-ed
__device__ volatile int internal_node_num;    // <= nbodiesd
__constant__ volatile int *internal_node_child;   // 4 x nbodiesd
__constant__ volatile float *internal_node_mass;    // nbodiesd
__constant__ volatile float *internal_node_posx;    // nbodiesd
__constant__ volatile float *internal_node_posy;    // nbodiesd
__device__ volatile float radiusd;
__constant__ volatile int *countd, *startd, *sortd;


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
    startd[k] = 0;
    internal_node_num = 0;  // internel_node_num refers to the index of the last internal node
}


__global__
void TreeBuildingKernel() {
    // totally 16 registers used
    register float radius = radiusd;
    register float rootx = internal_node_posx[0];
    register float rooty = internal_node_posy[0];
    register int inc = blockDim.x * gridDim.x;
    register int i = threadIdx.x + blockIdx.x * blockDim.x;
    register float px = 0.f;
    register float py = 0.f;
    register float r = 0.f;
    register int n = 0;  // root
    register int j = 0;
    register int locked = 0;
    register int patch = 999999999;
    register int internal_node_idx;
    register int k = 0;
    register float x;
    register float y;


    while (i < nbodiesd) {
        px = leaf_node_posx[i];
        py = leaf_node_posy[i];
        n = 0;  // root
        r = radius;
        j = 0;
        if (rootx < px) j = 1;
        if (rooty < py) j += 2;
        int child = internal_node_child[n * 4 + j];
        while (child >= nbodiesd) {
            n = child - nbodiesd;
            r *= 0.5f;
            j = 0;
            if (internal_node_posx[n] < px) j = 1;
            if (internal_node_posy[n] < py) j += 2;
            child = internal_node_child[n * 4 + j];
        }

        if (child != LOCK) {
            locked = n * 4 + j;
            if (child == atomicCAS((int *)&internal_node_child[locked], child, LOCK)) {
                if (child == NULL_BODY) { // if it is a null pointer, just insert the new body
                    internal_node_child[locked] = i;
                } else {  // there already is a body in this position
                    patch = 999999999;
                    do {
                        internal_node_idx = atomicAdd((int *)&internal_node_num, 1) + 1;
                        patch = min(patch, internal_node_idx);

                        x = (j & 1) * r;
                        y = ((j >> 1) & 1) * r;
                        r *= 0.5f;
                        internal_node_mass[internal_node_idx] = -1.0f;
                        startd[internal_node_idx] = -1;
                        x = internal_node_posx[internal_node_idx] = internal_node_posx[n] - r + x;
                        y = internal_node_posy[internal_node_idx] = internal_node_posy[n] - r + y;
                        for (k = 0; k < 4; k++) {
                            internal_node_child[internal_node_idx * 4 + k] = NULL_BODY;
                        }
                        if (patch != internal_node_idx) {
                            internal_node_child[n * 4 + j] = internal_node_idx + nbodiesd;
                        }

                        j = 0;
                        if (x < leaf_node_posx[child]) j = 1;
                        if (y < leaf_node_posy[child]) j += 2;
                        internal_node_child[internal_node_idx * 4 + j] = child;
                        n = internal_node_idx;
                        j = 0;
                        if (x < px) j = 1;
                        if (y < py) j += 2;
                        child = internal_node_child[n * 4 + j];
                    } while (child >= 0);
                    internal_node_child[n * 4 + j] = i;
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
void SummarizationKernel() {
    // totally 12 registers used
    register int last_internal_node = internal_node_num;
    register int inc = blockDim.x * gridDim.x;
    __shared__ volatile int cache[256 * 4];
    register int j;
    register int cnt;
    register float cm = 0.0f;
    register float px = 0.0f;
    register float py = 0.0f;
    register float m;
    register int i;
    register int child;

    // bottom-up
    register int k = last_internal_node - (threadIdx.x + blockIdx.x * blockDim.x);
    register int missing = 0;

    while (k >= 0) {
        cm = 0.0f;
        px = 0.0f;
        py = 0.0f;
        cnt = 0;
        j = 0;

        if (missing == 0) {
            for (i = 0; i < 4; i++) {
                child = internal_node_child[k * 4 + i];
                // comment: if this child is not null pointer (may be internal or leaf)
                if (child >= 0) {
                    if (i != j) {
                        // comment:
                        // move children to front (needed later for speed)
                        internal_node_child[k * 4 + i] = -1;
                        internal_node_child[k * 4 + j] = child;
                    }
                    cache[missing * 256 + threadIdx.x] = child;
                    m = 0.f;
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
                            cnt += (countd[child - nbodiesd]);
                            // assert(cnt >= 0);
                            // it is a internal node
                            px += internal_node_posx[child - nbodiesd] * m;
                            py += internal_node_posy[child - nbodiesd] * m;
                        } else {
                            // it is a leaf node
                            px += leaf_node_posx[child] * m;
                            py += leaf_node_posy[child] * m;
                        }
                    }
                    j++;
                }
            }
            cnt += j;
            // assert(cnt >= 0);
        }

        if (missing != 0) {
            do {
                // poll missing children
                child = cache[(missing - 1) * 256 + threadIdx.x];
                m = internal_node_mass[child - nbodiesd];
                if (m >= 0.f) {
                    missing--;
                    cm += m;
                    if (child >= nbodiesd) {
                        cnt += (countd[child - nbodiesd]);
                        assert(cnt >= 0);
                        // it is a internal node
                        px += internal_node_posx[child - nbodiesd] * m;
                        py += internal_node_posy[child - nbodiesd] * m;
                    } else {
                        // it is a leaf node
                        px += leaf_node_posx[child] * m;
                        py += leaf_node_posy[child] * m;
                    }
                }
            } while ((m >= 0.f) && (missing != 0));
        }

        if (missing == 0) {
            // all children are ready, so store computed information
            countd[k] = cnt;  // These counts make kernel 4 much faster
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
void SortKernel() {
    // totally 5 registers used
    register int i, k, child, inc, start;

    inc = blockDim.x * gridDim.x;
    k = threadIdx.x + blockIdx.x * blockDim.x;

    while (k <= internal_node_num) {
        // it concurrently places the bodies into an array such that the bodies appear in the same order in the array as they would during an in-order traversal of the octree
        start = startd[k];
        if (start >= 0) {
            for (i = 0; i < 4; i++) {
                child = internal_node_child[k * 4 + i];
                if (child >= nbodiesd) {
                    // child is a cell
                    startd[child - nbodiesd] = start;
                    start += countd[child - nbodiesd];
                } else if (child >= 0) {
                    // child is a body
                    sortd[start] = child;
                    __threadfence();
                    start++;
                }
            }
            k += inc;
        }
        __syncthreads();  // throttle
    }
    __syncthreads();
}

// The most obvious problem with our recursive implementation is high execution divergence
__device__
float2 CalculateForceOnLeafNode(int leaf_node) {
    // notice: float2 overloads + (part of cuda runtime)
    // depth is for calculating region size (radiusd / (2^depth)); the depth of root is 0
    // target_node is the index in internel_node_child

    // initialize a stack and push_back root
    // there is no null pointer in the stack (check before push_back)
    register int stack[1024];   // store the index of internal nodes and leaf nodes
    register int stack_idx = -1;
    // push_back
    stack[++stack_idx] = 0;  // depth; for calculating s
    stack[++stack_idx] = nbodiesd;

    // while (1) {
    //  if stack is not empty
    //    fetch one element
    //    check if (s/d < SD_TRESHOLD):
    //      yes -> add to ax and ay
    //      no -> push its four children onto stack
    // }

    register int node_idx = 0;;
    register int depth = 0;
    register float s = 0.f;
    register float distance = 0.f;
    register float dx = 0.f, dy = 0.f;
    register float px = 0.f, py = 0.f;
    register float ax = 0.f, ay = 0.f;
    volatile float *x_array, *y_array, *mass_array;
    register bool isleaf;
    register int k;

    while (stack_idx >= 0) {
        // pop
        node_idx = stack[stack_idx--];
        depth = stack[stack_idx--];
        isleaf = false;

        if (node_idx >= 0 && node_idx < nbodiesd) {
            x_array = leaf_node_posx;
            y_array = leaf_node_posy;
            mass_array = leaf_node_mass;
            isleaf = true;
        } else {
            node_idx -= nbodiesd;
            x_array = internal_node_posx;
            y_array = internal_node_posy;
            mass_array = internal_node_mass;
        }

        s = radiusd / (1 << depth);
        px = leaf_node_posx[leaf_node];
        py = leaf_node_posy[leaf_node];
        // if the target_node is a leaf node, simply calculate the force and return
        dx = x_array[node_idx] - px;
        dy = y_array[node_idx] - py;

        distance = dx * dx + dy * dy + epssqd;
        distance = rsqrtf(distance);
        distance = mass_array[node_idx] * distance * distance;

        if (((s / distance) < SD_TRESHOLD) || isleaf) {
            //add to ax and ay
            ax += dx * distance;
            ay += dy * distance;
        } else {
            for (k = 0; k < 4; k++) {
                if (internal_node_child[4 * node_idx + k] != NULL_BODY) {
                    stack[++stack_idx] = depth + 1;   // push_back depth first
                    stack[++stack_idx] = internal_node_child[4 * node_idx + k];
                }
            }
        }
    }
    __syncthreads();
    return make_float2(ax, ay);
}

__global__
void ForceCalculationKernel() {
    // totally 15 registers used
    register int inc = blockDim.x * gridDim.x;
    register int i = threadIdx.x + blockIdx.x * blockDim.x;
    register int k;  // actual point
    register float2 acceleration;

    while (i < nbodiesd) {
        k = sortd[i];
        // k = i;
        acceleration = CalculateForceOnLeafNode(k);
        leaf_node_accx[k] = acceleration.x;
        leaf_node_accy[k] = acceleration.y;
        i += inc;
    }
}


__global__
void IntegrationKernel() {
    // totally 6 registers used
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

void checkLimit() {
    size_t limit = 0;
    //cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    //printf("cudaLimitStackSize: %u\n", (unsigned)limit);
    //cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
    //printf("cudaLimitPrintfFifoSize: %u\n", (unsigned)limit);
    //cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
    //printf("cudaLimitMallocHeapSize: %u\n", (unsigned)limit);

    limit = 9999;

    cudaDeviceSetLimit(cudaLimitStackSize, limit);
    //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, limit);
    //cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit);

    limit = 0;

    cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    printf("New cudaLimitStackSize: %u\n", (unsigned)limit);
    //cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
    //printf("New cudaLimitPrintfFifoSize: %u\n", (unsigned)limit);
    //cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
    //printf("New cudaLimitMallocHeapSize: %u\n", (unsigned)limit);
}


void
cudaBHSpaceModel::update(GS_FLOAT dt) {
    size_t i;
#ifdef CONST_TIME
    dt = CONST_TIME;
#endif
    int blocks;
    int nbodies;
    float *mass, *posx, *posy, *velx, *vely;
    float *massl, *posxl, *posyl, *velxl, *velyl;
    int *countl, *startl, *sortl;
    float *leaf_node_accxl, *leaf_node_accyl;
    int *internal_node_childl;
    float *internal_node_massl, *internal_node_posxl, *internal_node_posyl;
    float dtime, dthf, epssq, itolsq;

    blocks = 15; // number of multiprocessor, specific to K40m
    nbodies = this->objects.size();

    // checkLimit();

    // for segregate information of objects into different array
    // on the host
    mass = (float *)malloc(sizeof(float) * nbodies);
    posx = (float *)malloc(sizeof(float) * nbodies);
    posy = (float *)malloc(sizeof(float) * nbodies);
    velx = (float *)malloc(sizeof(float) * nbodies);
    vely = (float *)malloc(sizeof(float) * nbodies);
    for (i = 0; i < nbodies; i++) {
        posx[i] = this->objects[i].position.x;
        posy[i] = this->objects[i].position.y;
        velx[i] = this->objects[i].speed.x;
        vely[i] = this->objects[i].speed.y;
        mass[i] = this->objects[i].mass;
    }

    // on the device
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

    // for counting how many nodes an internal node have
    cudaMalloc((void **)&countl, sizeof(int) * (nbodies + 1));
    cudaMemset(countl, 0, sizeof(int) * (nbodies + 1));
    cudaMemcpyToSymbol(countd, &countl, sizeof(void *));

    cudaMalloc((void **)&startl, sizeof(int) * (nbodies + 1));
    cudaMemcpyToSymbol(startd, &startl, sizeof(void *));

    cudaMalloc((void **)&sortl, sizeof(int) * (nbodies + 1));
    cudaMemcpyToSymbol(sortd, &sortl, sizeof(void *));

    // allocate leaf node acceleration information
    cudaMalloc((void **)&leaf_node_accxl, sizeof(float) * (nbodies + 1));
    cudaMalloc((void **)&leaf_node_accyl, sizeof(float) * (nbodies + 1));

    // copy to constant memory
    cudaMemcpyToSymbol(leaf_node_accx, &leaf_node_accxl, sizeof(void *));
    cudaMemcpyToSymbol(leaf_node_accy, &leaf_node_accyl, sizeof(void *));

    // allocate space for internal nodes
    // on the device
    cudaMalloc((void **)&internal_node_childl, sizeof(int) * (nbodies + 1) * 4);
    cudaMalloc((void **)&internal_node_massl, sizeof(float) * (nbodies + 1));
    cudaMalloc((void **)&internal_node_posxl, sizeof(float) * (nbodies + 1));
    cudaMalloc((void **)&internal_node_posyl, sizeof(float) * (nbodies + 1));

    // copy to constant memory
    cudaMemcpyToSymbol(internal_node_child, &internal_node_childl, sizeof(void *));
    cudaMemcpyToSymbol(internal_node_mass, &internal_node_massl, sizeof(void *));
    cudaMemcpyToSymbol(internal_node_posx, &internal_node_posxl, sizeof(void *));
    cudaMemcpyToSymbol(internal_node_posy, &internal_node_posyl, sizeof(void *));

    // calculation factors
    dtime = dt;
    dthf = dtime * 0.5f;
    epssq = 0.05 * 0.05;  // EPSILON; for soothing
    itolsq = 1.0f / (0.5 * 0.5);

    // copy the address of the array to constant memory
    cudaMemcpyToSymbol(nbodiesd, &nbodies, sizeof(int));
    cudaMemcpyToSymbol(dtimed, &dtime, sizeof(float));
    cudaMemcpyToSymbol(dthfd, &dthf, sizeof(float));
    cudaMemcpyToSymbol(epssqd, &epssq, sizeof(float));
    cudaMemcpyToSymbol(itolsqd, &itolsq, sizeof(float));

    InitializationKernel <<< 1, 1>>>();
    TreeBuildingKernel <<< blocks *8, 256>>>();
    SummarizationKernel <<< blocks *8, 256>>>();
    SortKernel <<< blocks *8, 256>>>();
    // printf("before ForceCalculationKernel\n");
    ForceCalculationKernel <<< blocks *8, 256>>>();
    IntegrationKernel <<< blocks *FACTOR6, 256>>>();

    // we only need to copy these four back into host
    cudaMemcpy(posx, posxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost);
    cudaMemcpy(posy, posyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost);
    cudaMemcpy(velx, velxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost);
    cudaMemcpy(vely, velyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost);

    // update leaf_node_posx, leaf_node_posy, leaf_node_velx, leaf_node_vely to objects array
    for (i = 0; i < nbodies; i++) {
        this->objects[i].position.x = posx[i];
        this->objects[i].position.y = posy[i];
        this->objects[i].speed.x = velx[i];
        this->objects[i].speed.y = vely[i];
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
    cudaFree(startl);
    cudaFree(countl);
    cudaFree(sortl);
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

