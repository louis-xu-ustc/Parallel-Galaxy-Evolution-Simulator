#include <cuda.h>

#define NULL_BODY (-1)
#define LOCK (-2)

#define NUMBER_OF_CELLS 4

__device__ void
buildtree(float *_posX, float *_posY,
	float *_velX, float *_velY, 
	float *_accX, float *_accY,
	int *_step, int *_blockCount, int *_bodyCount, float *_radius, int *_maxDepth,
	int *_bottom, float *_mass, int *_child, int *_start, int *_sorted, int *_error){

	int localMaxDepth = 1;

	int stepSize = blockDim.x * gridDim.x;

	// Cache root data 
	float radius = *_radius;
	float rootX = _posX[NUMBER_OF_NODES];
	float rootY = _posY[NUMBER_OF_NODES];

	int childPath;
	bool newBody = true;
	int node;

	int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;

	while (bodyIndex < NBODIES) {
		float currentR;
		float bodyX, bodyY;

		int depth;

		if (newBody) {
			// new body, so start traversing at root
			newBody = false;
			bodyX = _posX[bodyIndex];
			bodyY = _posY[bodyIndex];
			
			node = NUMBER_OF_NODES;
			depth = 1;
			currentR = radius;

			childPath = 0;
			if (rootX < bodyX) childPath = 1;
			if (rootY < bodyY) childPath += 2;
		}

		int childIndex = _child[NUMBER_OF_CELLS * node + childPath];

		while (childIndex >= NBODIES) {
			node = childIndex;
			++depth;
			currentR *= 0.5f;

			childPath = 0;
			if (_posX[node] < bodyX) childPath = 1;
			if (_posY[node] < bodyY) childPath += 2;
			childIndex = _child[NUMBER_OF_CELLS * node + childPath];
		}

		if (childIndex != LOCK) {
			int locked = NUMBER_OF_CELLS * node + childPath;
			if (childIndex == atomicCAS((int *)&childd[locked], childIndex, NULL_BODY)) { // try locking
				if (childIndex == NULL_BODY) {
					_child[locked] = bodyIndex;
				} else {
					int patch = -1;
					do {
						depth++;
						int cell = atomicSub((int *)&bottomd, 1) - 1;
						if (cell <= NBODIES) {
							*_error = 1;
							*_bottom = NUMBER_OF_NODES;
							return;
						}
						patch = max(patch, cell);

						float x = (childPath & 1) * currentR;
						float y = ((childPath >> 1) & 1) * currentR;
						currentR *= 0.5f;
						// reset
						_mass[cell] = -1.0f;
						_start[cell] = -1;

						x = _posX[cell] = _posX[node] - currentR + x;
						y = _posY[cell] = _posY[node] - currentR + y;

						for (int k = 0; k < NUMBER_OF_CELLS; k++) 
							_child[cell * NUMBER_OF_CELLS + k] = -1;

						if (patch != cell) {
							_child[NUMBER_OF_CELLS * node + childPath] = cell;
						}

						childPath = 0;
						if (x < _posX[childIndex]) childPath = 1;
						if (y < _posY[childIndex]) childPath += 2;
						_child[NUMBER_OF_CELLS * cell + childPath] = childIndex;

						// next child
						node = cell;
						childPath = 0;
						if (x < bodyX) childPath = 1;
						if (y < bodyY) childPath += 2;

						childIndex = _child[NUMBER_OF_CELLS * node + childPath];


					} while (childIndex >= 0);
					_child[NUMBER_OF_CELLS * node + childPath] = bodyIndex;
					__threadfence();
					_child[locked] = patch;
				}
				localMaxDepth = max(depth, localMaxDepth);
				// move to next body
				bodyIndex += stepSize;
				newBody = true;
			}
		}

		__syncthreads();

	}
	atomicMax((int *)&_maxDepth, localMaxDepth);

}