#ifndef GravitySim_space_model_h
#define GravitySim_space_model_h

#include <stdlib.h>
#include <vector>
#include "Object.h"
#include "Screen.h"

class SpaceModel {
private:
    
public:
    RectangleD bounds;
    std::vector<Object> objects;
    Screen *screen;
    SpaceModel(RectangleD bounds, std::vector<Object> &objects, Screen *screen);

    ~SpaceModel();

    void add_galaxy(Point2D position, GS_FLOAT size, size_t n);

    void remove_objects_outside_bounds();

    virtual void update(GS_FLOAT dt);
    virtual void draw_bounds();
};

#endif
