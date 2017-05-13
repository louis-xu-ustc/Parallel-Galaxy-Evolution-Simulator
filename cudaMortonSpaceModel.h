#ifndef cudaMortonSpaceModel_h
#define cudaMortonSpaceModel_h

#include "SpaceModel.h"
#include "MortonTree.h"
#include "Screen.h"

class cudaMortonSpaceModel: public SpaceModel {
    private:
        MortonTree *tree;
      
        int errCnt;

        bool firstUpdated;
        bool inited;
        // Object elements
        int numObjects, boundX, boundY, sizeX, sizeY;
        // position - 1*numObjects
        GS_FLOAT* posx;
        GS_FLOAT* posy;

        // speed - 1*numObjects
        GS_FLOAT* spdx;
        GS_FLOAT* spdy;

        // mass - 1*numObjects
        GS_FLOAT* mass;

        int* parent;

        // device memory
        GS_FLOAT* devPosX;
        GS_FLOAT* devPosY;
        GS_FLOAT* devSpdX;
        GS_FLOAT* devSpdY;
        GS_FLOAT* devMass;
        int *devParent;

        // MortonCell elements
        int numCells;
        GS_FLOAT* cellPosX;
        GS_FLOAT* cellPosY;
        GS_FLOAT* cellBndX;
        GS_FLOAT* cellBndY;
        GS_FLOAT* cellMass;
        int* cellCnt;

        GS_FLOAT* devCellPosX;
        GS_FLOAT* devCellPosY;
        GS_FLOAT* devCellBndX;
        GS_FLOAT* devCellBndY;
        GS_FLOAT* devCellMass;
        int* devCellCnt;
       
    public:
        cudaMortonSpaceModel(RectangleD bouds, std::vector<Object> &objects, Screen *screen);
        ~cudaMortonSpaceModel();
            
        void alloc(int numObjects, int numCells);
        void dealloc();
        void fillObjectsToCuda(std::vector<MortonTreeObject*> &objects);
        void fillCells(std::vector<MortonCell*> &cells);
        void fillObjectsFromCuda(std::vector<Object> &objects);
        void correctnessCheck(GS_FLOAT dt);

        virtual void update(GS_FLOAT dt) override;
};
#endif
