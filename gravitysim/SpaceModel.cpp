#include "SpaceModel.h"
#include <stdlib.h>
#include <stdio.h>

SpaceModel::SpaceModel(RectangleD bounds, std::vector<Object> &objects) {
    this->bounds = bounds;
    // change to new syntax; this is where dependency injection plays its part
    this->tree = new QuadTree(this->bounds);

    if (this->tree == NULL) {
        printf("unable to initialize QuadTree in SpaceModel\n");
        return;
    }
    // deep copy
    this->objects = objects;
}

void 
SpaceModel::remove_objects_outside_bounds() {
    std::vector<Object>::iterator it = objects.begin();
    while (it != objects.end()) {
        if (!point2d_is_in_rectangled(it->position, this->bounds)) {
            it = objects.erase(it);
        } else {
            ++it;
        }
    }
}

void 
SpaceModel::update(GS_FLOAT dt) {
    size_t i;
#ifdef CONST_TIME
    dt = CONST_TIME;
#endif
    this->tree->apply_to_objects(this->objects, dt);
    for (i = 0; i < objects.size(); i++) {
        objects[i].update_position(dt);
    }
    remove_objects_outside_bounds();
    delete this->tree;
    this->tree = new QuadTree(this->bounds);
    this->tree->add_objects(this->objects);
}

SpaceModel::~SpaceModel() {
    delete this->tree;
}
