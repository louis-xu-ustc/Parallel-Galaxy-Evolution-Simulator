#include "SpaceModel.h"
#include <stdlib.h>
#include <stdio.h>

SpaceModel::SpaceModel(RectangleD bounds, std::vector<Object> &objects, Screen *screen) {
    this->bounds = bounds;
    // deep copy
    this->objects = objects;
    this->screen = screen;
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
}

void
SpaceModel::draw_bounds() {
}

SpaceModel::~SpaceModel() {
}
