#include "MortonSpaceModel.h"
#include "log.h"

static int log_level = LOG_INFO;

MortonSpaceModel::MortonSpaceModel(RectangleD bounds, std::vector<Object> &objects, Screen *screen) : SpaceModel(bounds, objects, screen) {
    this->tree = new MortonTree(bounds);
    if (this->tree == NULL) {
        printf("unable to initialize MortonTree in SpaceModel");
        return;
    }
    this->tree->fillMortonTreeObjects(this->objects);
    this->tree->generateMortonTree();
}

MortonSpaceModel::~MortonSpaceModel() {
    delete this->tree;
}

void
MortonSpaceModel::update(GS_FLOAT dt) {
#ifdef CONST_TIME
    dt = CONST_TIME;
#endif

    struct timespec one, two, three, four, five, six;

    clock_gettime(CLOCK_REALTIME, &one);
    this->tree->applyToObjects(dt);
    clock_gettime(CLOCK_REALTIME, &two);

    this->objects.clear();
    std::vector<MortonTreeObject*> objs = this->tree->getObjects();
    for (size_t i = 0; i < objs.size(); i++) {
        this->objects.push_back(*objs[i]);
        this->objects[i].update_position(dt);
    }

    clock_gettime(CLOCK_REALTIME, &three);
    //printf("number of objs: %lu\n", this->objects.size());
    remove_objects_outside_bounds();
    clock_gettime(CLOCK_REALTIME, &four);

    delete this->tree;
    this->tree = new MortonTree(this->bounds);
    this->tree->fillMortonTreeObjects(this->objects);
    clock_gettime(CLOCK_REALTIME, &five);

    this->tree->generateMortonTree();
    clock_gettime(CLOCK_REALTIME, &six);

    GS_DOUBLE apply_force_time = get_timediff(two, one);
    GS_DOUBLE update_objects = get_timediff(three, two);
    GS_DOUBLE remove_outside_objects = get_timediff(four, three);
    GS_DOUBLE fill_objects = get_timediff(five, four);
    GS_DOUBLE generate_morton_tree = get_timediff(six, five);
    GS_DOUBLE total_time = apply_force_time + update_objects + remove_outside_objects + fill_objects + generate_morton_tree;

    PERF("apply_force_time = %.2f%%\n", apply_force_time/total_time*100);
    PERF("update_objects = %.2f%%\n", update_objects/total_time*100);
    PERF("remove_outside_objects = %.2f%%\n", remove_outside_objects/total_time*100);
    PERF("fill_objects = %.2f%%\n", fill_objects/total_time*100);
    PERF("generate_morton_tree = %.2f%%\n", generate_morton_tree/total_time*100);
    PERF("total time:%.2f ms\n", total_time/1000000);
}

/**
 * draw MortonTree in the SpaceView
 */
void
MortonSpaceModel::draw_mortonTree(MortonTree *tree) {
    std::vector<MortonCell*> cells = this->tree->getCells();
    for (size_t i = 0; i < cells.size(); i++) {
        this->screen->draw_empty_rectangle(cells[i]->bound, RGB_BLUE);   
    }
}

void
MortonSpaceModel::draw_bounds() {
    draw_mortonTree(this->tree);
}
