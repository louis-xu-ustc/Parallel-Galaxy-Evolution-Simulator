#ifndef MortonSpaceModel_h
#define MortonSpaceModel_h

#include "SpaceModel.h"
#include "MortonTree.h"

class MortonSpaceModel : public SpaceModel {
    private:
        MortonTree *tree;
    public:
        MortonSpaceModel(RectangleD bounds, std::vector<Object> &objects);
        ~MortonSpaceModel();
        virtual void update(GS_FLOAT dt) override;
};

#endif
