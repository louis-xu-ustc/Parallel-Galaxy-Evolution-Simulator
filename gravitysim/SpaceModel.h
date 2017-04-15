#ifndef GravitySim_space_model_h
#define GravitySim_space_model_h

#include <stdlib.h>
#include "Object.h"
#include "QuadTree.h"
#include <vector>

class SpaceModel {
private:
    

public:
    RectangleD bounds;
    std::vector<Object> objects;
    QuadTree *tree;
    SpaceModel(RectangleD bounds, std::vector<Object> &objects);

    ~SpaceModel();

    void add_galaxy(Point2D position, GS_FLOAT size, size_t n);

    void remove_objects_outside_bounds();

    void update(GS_FLOAT dt);
};

#endif
