#include "MortonSpaceModel.h"

MortonSpaceModel::MortonSpaceModel(RectangleD bounds, std::vector<Object> &objects, Screen *screen) : SpaceModel(bounds, objects, screen) {
    this->tree = new MortonTree(bounds);
    if (this->tree == NULL) {
        printf("unable to initialize MortonTree in SpaceModel");
        return;
    }
    this->tree->fillMortonTreeObjects(this->objects);
}

MortonSpaceModel::~MortonSpaceModel() {
    delete this->tree;
}

void
MortonSpaceModel::update(GS_FLOAT dt) {
#ifdef CONST_TIME
    dt = CONST_TIME;
#endif
    this->tree->generateMortonTree();
    this->tree->applyToObjects(dt);
    this->objects.clear();
    std::vector<MortonTreeObject*> objs = this->tree->getObjects();
    for (int i = 0; i < objs.size(); i++) {
        this->objects.push_back(*objs[i]);
    }
    //printf("number of objs: %lu\n", this->objects.size());
    remove_objects_outside_bounds();
    delete this->tree;
    this->tree = new MortonTree(this->bounds);
    this->tree->fillMortonTreeObjects(this->objects);
}

/**
 * draw MortonTree in the SpaceView
 */
void
MortonSpaceModel::draw_mortonTree(MortonTree *tree) {
    for (int i = 0; i < this->tree->getCells().size(); i++) {
        this->screen->draw_empty_rectangle(this->tree->getCells()[i]->bound, RGB_BLUE);   
    }
}

void
MortonSpaceModel::draw_bounds() {
    draw_mortonTree(this->tree);
}
