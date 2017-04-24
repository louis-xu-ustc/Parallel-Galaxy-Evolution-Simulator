#ifndef BHSpaceModel_h
#define BHSpaceModel_h

#include "SpaceModel.h"
#include "QuadTree.h"

class BHSpaceModel : public SpaceModel
{
private:
	QuadTree *tree;
    void draw_quadTree(QuadTree *tree);
public:
	BHSpaceModel(RectangleD bounds, std::vector<Object> &objects, Screen *screen);
	~BHSpaceModel();
	virtual void update(GS_FLOAT dt) override;
    virtual void draw_bounds() override;
};


#endif
