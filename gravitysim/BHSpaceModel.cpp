#include "BHSpaceModel.h"


BHSpaceModel::BHSpaceModel(RectangleD bounds, std::vector<Object> &objects) 
: SpaceModel(bounds, objects) {
    this->tree = new QuadTree(this->bounds);
    if (this->tree == NULL) {
        printf("unable to initialize QuadTree in SpaceModel\n");
        return;
    }
} 


void 
BHSpaceModel::update(GS_FLOAT dt) {
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


BHSpaceModel::~BHSpaceModel() {
    delete this->tree;
}
