#ifndef cudaBHSpaceModel_h
#define cudaBHSpaceModel_h

#include "SpaceModel.h"
#include "QuadTree.h"

class cudaBHSpaceModel : public SpaceModel
{
private:
	QuadTree *tree;
public:
	cudaBHSpaceModel(RectangleD bounds, std::vector<Object> &objects);
	~cudaBHSpaceModel();
	virtual void update(GS_FLOAT dt) override;
};

#endif