#ifndef MortonSpaceModel_h
#define MortonSpaceModel_h

#include "SpaceModel.h"
#include "MortonTree.h"

class MortonSpaceModel : public SpaceModel {
    private:
        MortonTree *tree;
        void draw_mortonTree(MortonTree *tree);
    public:
        MortonSpaceModel(RectangleD bounds, std::vector<Object> &objects, Screen *screen);
        ~MortonSpaceModel();
        virtual void update(GS_FLOAT dt) override;
        virtual void draw_bounds() override;
};

#endif
