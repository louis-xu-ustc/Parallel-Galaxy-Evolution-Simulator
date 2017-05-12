#include <thrust/functional.h>
#include <thrust/sort.h>
#include "cudaMortonSpaceModel.h"
#include "log.h"
#include "cuda.h"

#define UPDIV(x, align)     (((x) + (align) - 1) / (align))
//#define CORRECTNESS_CHECK

struct GlobalConstants {
    GS_FLOAT *posx, *posy, *spdx, *spdy, *mass;
    int *parent;
    GS_FLOAT *cellPosX, *cellPosY, *cellBndX, *cellBndY, *cellMass;
    int *cellCnt;
};

__constant__ GlobalConstants cudaConstMortonParams;

static int log_level = LOG_INFO;

__device__ GS_INLINE float2 float2_add(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ GS_INLINE float2 float2_sub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ GS_INLINE float2 float2_mul(float2 a, float x) {
    return make_float2(a.x * x, a.y * x);
}

__device__ GS_INLINE float2 float2_zero() {
    return make_float2(0.f, 0.f);
}

__device__ GS_INLINE GS_FLOAT float2_length(float2 dr) {
    return sqrt((dr.x * dr.x) + (dr.y * dr.y));
}

__device__ GS_INLINE float2 calculate_force(float2 aPos, float2 bPos, float bMass) {
    register float2 dr = float2_sub(bPos, aPos);
    register GS_FLOAT r = float2_length(dr) + SOFT_CONST;
    register GS_FLOAT f = G_CONST * bMass / SQUARE(r);
    return float2_mul(dr, f/r);
}

__global__ void
applyToObjectsKernel(GS_FLOAT dt, int numObjs) {
    register int tarIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tarIdx >= numObjs) {
        return;
    }

    GS_FLOAT *devPosX = cudaConstMortonParams.posx;
    GS_FLOAT *devPosY = cudaConstMortonParams.posy;
    GS_FLOAT *devSpdX = cudaConstMortonParams.spdx;
    GS_FLOAT *devSpdY = cudaConstMortonParams.spdy;
    GS_FLOAT *devMass = cudaConstMortonParams.mass;
    int *devParent = cudaConstMortonParams.parent;
    GS_FLOAT *devCellPosX = cudaConstMortonParams.cellPosX;
    GS_FLOAT *devCellPosY = cudaConstMortonParams.cellPosY;
    GS_FLOAT *devCellBndX = cudaConstMortonParams.cellBndX;
    GS_FLOAT *devCellBndY = cudaConstMortonParams.cellBndY;
    GS_FLOAT *devCellMass = cudaConstMortonParams.cellMass;
    int *devCellCnt = cudaConstMortonParams.cellCnt;

    register float2 tarObjPos = make_float2(devPosX[tarIdx], devPosY[tarIdx]), result = float2_zero();
    register float2 acc, dv, dr;
    register GS_FLOAT s, d;
    register int i = 0;

    while (i < numObjs) {
        register int cellIdx = devParent[i];
        register float2 cellBnd = make_float2(devCellBndX[cellIdx], devCellBndY[cellIdx]);
        register float2 cellPos = make_float2(devCellPosX[cellIdx], devCellPosY[cellIdx]);
        register GS_FLOAT cellMass = devCellMass[cellIdx];
        register int currCellCnt = devCellCnt[cellIdx];
        s = MAX(cellBnd.x, cellBnd.y);
        dr = float2_sub(tarObjPos, cellPos);
        d = float2_length(dr);

        if ((s/d) < SD_TRESHOLD) {
            result = float2_add(result, calculate_force(tarObjPos, cellPos, cellMass));
        } else {
            for (int j = i; (j < i + currCellCnt) && (j < numObjs); j++) {
                if (j == tarIdx) {
                    continue;
                }
                register float2 currObjPos = make_float2(devPosX[j], devPosY[j]);
                register float currObjMass = devMass[j];
                result = float2_add(result, calculate_force(tarObjPos, currObjPos, currObjMass));
            }
        }
        i += currCellCnt;
    }
    acc = result;
    dv = float2_mul(acc, dt);
    float2 oldDevSpd = make_float2(devSpdX[tarIdx], devSpdY[tarIdx]);
    float2 newDevSpd = float2_add(oldDevSpd, dv);
    devSpdX[tarIdx] = newDevSpd.x;
    devSpdY[tarIdx] = newDevSpd.y;
}

__global__ void
updatePositionsKernel(GS_FLOAT dt, int numObjs) {
    register int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numObjs) {
        return;
    }
    GS_FLOAT *devPosX = cudaConstMortonParams.posx;
    GS_FLOAT *devPosY = cudaConstMortonParams.posy;
    GS_FLOAT *devSpdX = cudaConstMortonParams.spdx;
    GS_FLOAT *devSpdY = cudaConstMortonParams.spdy;
    register float2 oldPos = make_float2(devPosX[idx], devPosY[idx]);
    register float2 spd = make_float2(devSpdX[idx], devSpdY[idx]);
    register float2 dp = float2_mul(spd, dt);
    register float2 newPos = float2_add(oldPos, dp);
    devPosX[idx] = newPos.x;
    devPosY[idx] = newPos.y;
} 

cudaMortonSpaceModel::cudaMortonSpaceModel(RectangleD bouds, std::vector<Object> &objects, Screen *screen) : SpaceModel(bouds, objects, screen) {
    ENTER();
    this->tree = new MortonTree(bounds);
    if (this->tree == NULL) {
        ERR("Unable to initialize MortonTree in SpaceModel");
        return;
    }
    this->tree->fillMortonTreeObjects(this->objects);
    this->tree->generateMortonTree();
    this->errCnt = 0;

    numObjects = objects.size();
    boundX = bouds.origin.x;
    boundY = bouds.origin.y;
    sizeX = bounds.size.x;
    sizeY = bounds.size.y;

    firstUpdated = true;
    LEAVE();
}

cudaMortonSpaceModel::~cudaMortonSpaceModel() {
    delete this->tree;
}

void
cudaMortonSpaceModel::update(GS_FLOAT dt) {
#ifdef CONST_TIME
    dt = CONST_TIME;
#endif
    ENTER();

    struct timespec one, two, three, four, five;
    clock_gettime(CLOCK_REALTIME, &one);
    int numCells = this->tree->getCells().size();
    alloc(numObjects, numCells);

    // getObjects from the tree and then fill in buffers for CUDA
    std::vector<MortonTreeObject*> objs = this->tree->getObjects();
    fillObjectsToCuda(objs);
    std::vector<MortonCell*> cells = this->tree->getCells();
    fillCells(cells);
    clock_gettime(CLOCK_REALTIME, &two);

    int threads = 512;
    int blocks = UPDIV(numObjects, threads);
    applyToObjectsKernel<<<blocks, threads>>>(dt, numObjects);
    cudaDeviceSynchronize();
    //cudaGetErrorString(cudaGetLastError());
    //printf("Sync: %s\n", cudaGetErrorString(cudaThreadSynchronize()));

    updatePositionsKernel<<<blocks, threads>>>(dt, numObjects);
    //cudaGetErrorString(cudaGetLastError());
    //printf("Sync: %s\n", cudaGetErrorString(cudaThreadSynchronize()));

    clock_gettime(CLOCK_REALTIME, &three);
    
    // get the updated positions and fill in this->objects
    fillObjectsFromCuda(this->objects);
    dealloc();
    clock_gettime(CLOCK_REALTIME, &four);

#if 0
    this->tree->applyToObjects(dt);
    this->objects.clear();
    for (size_t i = 0; i < objs.size(); i++) {
        this->objects.push_back(*objs[i]);
        this->objects[i].update_position(dt);
    }
#endif

#ifdef CORRECTNESS_CHECK
    this->tree->applyToObjects(dt);
    std::vector<MortonTreeObject*> comp_objs = this->tree->getObjects();
    for (int i = 0; i < comp_objs.size(); i++) {
        comp_objs[i]->update_position(dt);
        GS_FLOAT seq_x = comp_objs[i]->position.x;
        GS_FLOAT seq_y = comp_objs[i]->position.y;
        GS_FLOAT cuda_x = this->objects[i].position.x;
        GS_FLOAT cuda_y = this->objects[i].position.y;

        //printf("x - comp: %f, my: %f\n", seq_x, cuda_x);
        //assert(seq_x == cuda_x);
        //printf("y - comp: %f, my: %f\n", seq_y, cuda_y);
        //assert(seq_y == cuda_y);
        if ((seq_x - cuda_x)/seq_x > 0.001 || (seq_y - cuda_y)/seq_y > 0.001) {
            printf("x - comp: %f, my: %f\n", seq_x, cuda_x);
            printf("y - comp: %f, my: %f\n", seq_y, cuda_y);
            errCnt++;
        }
        if (errCnt > 10) {
            printf("galaxy render error!\n");
        }
    }
#endif

    remove_objects_outside_bounds();
    this->numObjects = this->objects.size();

    delete this->tree;
    this->tree = new MortonTree(this->bounds);
    this->tree->fillMortonTreeObjects(this->objects);
    this->tree->generateMortonTree();
    clock_gettime(CLOCK_REALTIME, &five);

    GS_DOUBLE memory_overhead = get_timediff(two, one) + get_timediff(four, three);
    GS_DOUBLE cuda_calculation = get_timediff(three, two);
    GS_DOUBLE morton_tree_generation = get_timediff(five, four);
    GS_DOUBLE total_time = memory_overhead + cuda_calculation + morton_tree_generation;
    PERF("memorycpy overhead = %.2f%%\n", memory_overhead/total_time*100);
    PERF("cuda calculation  = %.2f%%\n", cuda_calculation/total_time*100);
    PERF("morton tree generation  = %.2f%%\n", morton_tree_generation/total_time*100);

    LEAVE();
}

void
cudaMortonSpaceModel::alloc(int numObjects, int numCells) {
    ENTER();
    DBG("numObjects: %d, numCells:%d\n", numObjects, numCells);
    // objects
    this->posx = (GS_FLOAT*)malloc(sizeof(GS_FLOAT)*numObjects);
    this->posy = (GS_FLOAT*)malloc(sizeof(GS_FLOAT)*numObjects);
    this->spdx = (GS_FLOAT*)malloc(sizeof(GS_FLOAT)*numObjects);
    this->spdy = (GS_FLOAT*)malloc(sizeof(GS_FLOAT)*numObjects);
    this->mass = (GS_FLOAT*)malloc(sizeof(GS_FLOAT)*numObjects);
    this->parent = (int*)malloc(sizeof(int)*numObjects);
    if (this->posx == NULL || this->posy == NULL || this->spdx == NULL || this->spdy == NULL || this->parent == NULL) {
        ERR("malloc failed!\n");
        return;
    }

    cudaMalloc((void **)&this->devPosX, sizeof(GS_FLOAT)*numObjects);
    cudaMalloc((void **)&this->devPosY, sizeof(GS_FLOAT)*numObjects);
    cudaMalloc((void **)&this->devSpdX, sizeof(GS_FLOAT)*numObjects);
    cudaMalloc((void **)&this->devSpdY, sizeof(GS_FLOAT)*numObjects);
    cudaMalloc((void **)&this->devMass, sizeof(GS_FLOAT)*numObjects);
    cudaMalloc((void **)&this->devParent, sizeof(int)*numObjects);

    // cells
    this->cellPosX = (GS_FLOAT*)malloc(sizeof(GS_FLOAT)*numCells);
    this->cellPosY = (GS_FLOAT*)malloc(sizeof(GS_FLOAT)*numCells);
    this->cellBndX = (GS_FLOAT*)malloc(sizeof(GS_FLOAT)*numCells);
    this->cellBndY = (GS_FLOAT*)malloc(sizeof(GS_FLOAT)*numCells);
    this->cellMass = (GS_FLOAT*)malloc(sizeof(GS_FLOAT)*numCells);
    this->cellCnt = (int *)malloc(sizeof(int)*numCells);
    if (this->cellPosX == NULL || this->cellPosY == NULL || this->cellBndX == NULL || 
            this->cellBndY == NULL || this->cellMass == NULL || this->cellCnt == NULL) {
        ERR("malloc failed!\n");
        return;
    }
    cudaMalloc((void **)&this->devCellPosX, sizeof(GS_FLOAT)*numCells);
    cudaMalloc((void **)&this->devCellPosY, sizeof(GS_FLOAT)*numCells);
    cudaMalloc((void **)&this->devCellBndX, sizeof(GS_FLOAT)*numCells);
    cudaMalloc((void **)&this->devCellBndY, sizeof(GS_FLOAT)*numCells);
    cudaMalloc((void **)&this->devCellMass, sizeof(GS_FLOAT)*numCells);
    cudaMalloc((void **)&this->devCellCnt, sizeof(int)*numCells);

    this->inited = true;

    GlobalConstants params;
    params.posx = devPosX;
    params.posy = devPosY;
    params.spdx = devSpdX;
    params.spdy = devSpdY;
    params.mass = devMass;
    params.parent = devParent;
    params.cellPosX = devCellPosX;
    params.cellPosY = devCellPosY;
    params.cellBndX = devCellBndX;
    params.cellBndY = devCellBndY;
    params.cellMass = devCellMass;
    params.cellCnt = devCellCnt;
    cudaMemcpyToSymbol(cudaConstMortonParams, &params, sizeof(GlobalConstants));

    LEAVE();
}

void
cudaMortonSpaceModel::dealloc() {
    ENTER();
    if (this->inited) {
        free(this->posx);
        free(this->posy);
        free(this->spdx);
        free(this->spdy);
        free(this->mass);
        free(this->parent);

        free(this->cellPosX);
        free(this->cellPosY);
        free(this->cellBndX);
        free(this->cellBndY);
        free(this->cellMass);
        free(this->cellCnt);

        cudaFree(this->devPosX);
        cudaFree(this->devPosY);
        cudaFree(this->devSpdX);
        cudaFree(this->devSpdY);
        cudaFree(this->devMass);
        cudaFree(this->devParent);

        cudaFree(this->devCellPosX);
        cudaFree(this->devCellPosY);
        cudaFree(this->devCellBndX);
        cudaFree(this->devCellBndY);
        cudaFree(this->devCellMass);
        cudaFree(this->devCellCnt);
    }
    this->inited = false;
    LEAVE();
}

void
cudaMortonSpaceModel::fillObjectsToCuda(std::vector<MortonTreeObject*> &objects) {
    ENTER();

    register int num = objects.size();
    for (int i = 0; i < num; i++) {
        MortonTreeObject* obj = objects[i];
        this->posx[i] = obj->position.x;
        this->posy[i] = obj->position.y;
        this->spdx[i] = obj->speed.x;
        this->spdy[i] = obj->speed.y;
        this->mass[i] = obj->mass;
        this->parent[i] = obj->parent;
    }

    cudaMemcpy(this->devPosX, this->posx, sizeof(GS_FLOAT)*num, cudaMemcpyHostToDevice);
    cudaMemcpy(this->devPosY, this->posy, sizeof(GS_FLOAT)*num, cudaMemcpyHostToDevice);
    cudaMemcpy(this->devSpdX, this->spdx, sizeof(GS_FLOAT)*num, cudaMemcpyHostToDevice);
    cudaMemcpy(this->devSpdY, this->spdy, sizeof(GS_FLOAT)*num, cudaMemcpyHostToDevice);
    cudaMemcpy(this->devMass, this->mass, sizeof(GS_FLOAT)*num, cudaMemcpyHostToDevice);
    cudaMemcpy(this->devParent, this->parent, sizeof(int)*num, cudaMemcpyHostToDevice);
    LEAVE();
}

void
cudaMortonSpaceModel::fillCells(std::vector<MortonCell*> &cells) {
    ENTER();
    register int num = cells.size();
    for (int i = 0; i < num; i++) {
        MortonCell* cell = cells[i];
        this->cellMass[i] = cell->com.mass;
        this->cellPosX[i] = cell->com.position.x;
        this->cellPosY[i] = cell->com.position.y;
        this->cellBndX[i] = cell->bound.size.x;
        this->cellBndY[i] = cell->bound.size.y;
        this->cellCnt[i] = cell->count;
    }
    cudaMemcpy(this->devCellMass, this->cellMass, sizeof(GS_FLOAT)*num, cudaMemcpyHostToDevice);
    cudaMemcpy(this->devCellPosX, this->cellPosX, sizeof(GS_FLOAT)*num, cudaMemcpyHostToDevice);
    cudaMemcpy(this->devCellPosY, this->cellPosY, sizeof(GS_FLOAT)*num, cudaMemcpyHostToDevice);
    cudaMemcpy(this->devCellBndX, this->cellBndX, sizeof(GS_FLOAT)*num, cudaMemcpyHostToDevice);
    cudaMemcpy(this->devCellBndY, this->cellBndY, sizeof(GS_FLOAT)*num, cudaMemcpyHostToDevice);
    cudaMemcpy(this->devCellCnt, this->cellCnt, sizeof(int)*num, cudaMemcpyHostToDevice);

    LEAVE();
}

void
cudaMortonSpaceModel::fillObjectsFromCuda(std::vector<Object> &objects) {
    ENTER();
    register int num = objects.size();
    cudaMemcpy(this->posx, this->devPosX, sizeof(GS_FLOAT)*num, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->posy, this->devPosY, sizeof(GS_FLOAT)*num, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->spdx, this->devSpdX, sizeof(GS_FLOAT)*num, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->spdy, this->devSpdY, sizeof(GS_FLOAT)*num, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->mass, this->devMass, sizeof(GS_FLOAT)*num, cudaMemcpyDeviceToHost);

    //this->objects.clear();
    for (int i = 0; i < num; i++) {
#if 0 
        Point2D pos = point2d_make(this->posx[i], this->posy[i]);
        Point2D spd = point2d_make(this->spdx[i], this->spdy[i]);
        GS_FLOAT mass = this->mass[i];
        this->objects.push_back(Object(pos, spd, mass));
#endif
        objects[i].position.x = this->posx[i];
        objects[i].position.y = this->posy[i];
        objects[i].speed.x = this->spdx[i];
        objects[i].speed.y = this->spdy[i];
    }
    LEAVE();
}
