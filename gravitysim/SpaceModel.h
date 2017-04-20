#ifndef GravitySim_space_model_h
#define GravitySim_space_model_h

#include <stdlib.h>
#include <vector>
#include "Object.h"

class SpaceModel {
private:
    
public:
    RectangleD bounds;
    std::vector<Object> objects;
    SpaceModel(RectangleD bounds, std::vector<Object> &objects);

    ~SpaceModel();

    void add_galaxy(Point2D position, GS_FLOAT size, size_t n);

    void remove_objects_outside_bounds();

    virtual void update(GS_FLOAT dt);
};

#endif
