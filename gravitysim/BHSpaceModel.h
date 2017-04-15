#ifndef BHSpaceModel_h
#define BHSpaceModel_h

#include "SpaceModel.h"
#include "QuadTree.h"

class BHSpaceModel : public SpaceModel
{
private:
	QuadTree *tree;
public:
	BHSpaceModel(RectangleD bounds, std::vector<Object> &objects);
	~BHSpaceModel();
	virtual void update(GS_FLOAT dt) override;
};


#endif