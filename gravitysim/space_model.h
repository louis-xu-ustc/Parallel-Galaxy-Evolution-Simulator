#ifndef GravitySim_space_model_h
#define GravitySim_space_model_h

#include <stdlib.h>
#include "object_array.h"
#include "object.h"
#include "quad_tree.h"

class SpaceModel {
private:
    RectangleD bounds;
    ObjectArray *objects;
    QuadTree *tree;

public:
    SpaceModel(RectangleD bounds, size_t capacity);

    SpaceModel(RectangleD bounds, RectangleD galaxies_bounds, size_t n_galaxies,
                                     size_t objects_per_galaxy, GS_FLOAT galaxy_size);

    void add_galaxy(Point2D position, GS_FLOAT size, size_t n);

    void remove_objects_outside_bounds();

    void update(GS_FLOAT dt);
}

#endif
