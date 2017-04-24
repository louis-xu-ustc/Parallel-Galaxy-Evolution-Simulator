#include <cuda.h>

#define NULL_BODY (-1)
#define LOCK (-2)

#define WARPSIZE 16

#define NUMBER_OF_CELLS 4 // the number of cells per node

__device__ void
summarizeTree(float *_posX, float *_posY,
	float *_velX, float *_velY, 
	float *_accX, float *_accY,
	int *_step, int *_blockCount, int *_bodyCount, float *_radius, int *_maxDepth,
	int *_bottom, float *_mass, int *_child, int *_start, int *_sorted, int *_error){

	// WORKGROUP_SIZE needs definition
	int localChild[WORKGROUP_SIZE * NUMBER_OF_CELLS];

	int bottom = *_bottom;
	int stepSize = blockDim.x * gridDim.x;

	int node = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;

	if (node < bottom)
		node += stepSize;

	int missing = 0;
	int cellBodyCount = 0;
	float cellMass;
	float mass;
	float centerX, centerY;

	while (node <= NUMBER_OF_NODES) {
		if (missing == 0) {
			cellMass = 0.0f;
			centerX = 0.0f;
			centerY = 0.0f;
			cellBodyCount = 0;

			int usedChildIndex = 0;

			for (int childIndex = 0; childIndex < NUMBER_OF_CELLS; childIndex++) {
				int child = _child[node * NUMBER_OF_CELLS + childIndex];

				if (child >= 0) {
					if (childIndex != usedChildIndex) {
						_child[NUMBER_OF_CELLS * node + childIndex] = -1;
						_child[NUMBER_OF_CELLS * node + usedChildIndex] = child;
					}
					localChild[WORKGROUP_SIZE * missing + get_local_id(0)] = child;
					mass = _mass[child];

					++missing;
					if (mass >= 0.0f) {
						--missing;
						if (child >= NBODIES) {
							cellBodyCount += _bodyCount[child] - 1;
						}
						cellMass += mass;
						centerX += _posX[child] * mass;
						centerY += _posY[child] * mass;
					}
					usedChildIndex++;
				}
			}
			cellBodyCount += usedChildIndex;
		}
		if (missing != 0) {
			do {
				int child = localChild[(missing - 1) * WORKGROUP_SIZE + get_local_id(0)];
				mass = _mass[child];

				if (mass >= 0.0f) {
					--missing;
					if (child >= NBODIES) {
						cellBodyCount = _bodyCount[child] - 1;
					}
					cellMass += mass;
					centerX += _posX[child] * mass;
					centerY += _posY[child] * mass;
				}
			} while ((mass >= 0.0f) && (missing != 0));
		}
		if (missing == 0) {
			_bodyCount[node] = cellBodyCount;
			mass = 1.0f / cellMass;
			_posX[node] = centerX * mass;
			_posY[node] = centerY * mass;

			__threadfence();
			_mass[node] = cellMass;
			node += stepSize;
		}
	}
}
