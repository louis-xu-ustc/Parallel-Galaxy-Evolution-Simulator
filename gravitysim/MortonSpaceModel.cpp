#include "MortonSpaceModel.h"

MortonSpaceModel::MortonSpaceModel(RectangleD bounds, std::vector<Object> &objects) : SpaceModel(bounds, objects) {
    this->tree = new MortonTree(bounds);
    if (this->tree == NULL) {
        printf("unable to initialize MortonTree in SpaceModel");
        return;
    }
}

MortonSpaceModel::~MortonSpaceModel() {
    delete this->tree;
}

void
MortonSpaceModel::update(GS_FLOAT dt) {
#ifdef CONST_TIME
    dt = CONST_TIME;
#endif
    this->tree->applyToObjects(dt);
    for (int i = 0; i < objects.size(); i++) {
        objects[i].update_position(dt);
    }
    remove_objects_outside_bounds();
    delete this->tree;
    this->tree = new MortonTree(this->bounds);
    this->tree->fillMortonTreeObjects(this->objects);
    this->tree->generateMortonTree();
}
