#include "BHSpaceModel.h"
#include "log.h"

static int log_level = LOG_INFO;

BHSpaceModel::BHSpaceModel(RectangleD bounds, std::vector<Object> &objects, Screen *screen) 
: SpaceModel(bounds, objects, screen) {
    this->tree = new QuadTree(this->bounds);
    if (this->tree == NULL) {
        ERR("unable to initialize QuadTree in SpaceModel\n");
        return;
    }
} 

void 
BHSpaceModel::update(GS_FLOAT dt) {
    size_t i;
#ifdef CONST_TIME
    dt = CONST_TIME;
#endif

    struct timespec one, two, three, four, five, six;

    clock_gettime(CLOCK_REALTIME, &one);
    this->tree->apply_to_objects(this->objects, dt);
    clock_gettime(CLOCK_REALTIME, &two);

    clock_gettime(CLOCK_REALTIME, &three);
    for (i = 0; i < objects.size(); i++) {
        objects[i].update_position(dt);
    }
    clock_gettime(CLOCK_REALTIME, &four);

    remove_objects_outside_bounds();

    clock_gettime(CLOCK_REALTIME, &five);
    delete this->tree;
    this->tree = new QuadTree(this->bounds);
    this->tree->add_objects(this->objects);
    clock_gettime(CLOCK_REALTIME, &six);

    double applying_force_time = get_timediff(two, one);
    double update_position_time = get_timediff(four, three);
    double rebuild_tree_time = get_timediff(six, five);
    double total_time = applying_force_time + update_position_time + rebuild_tree_time;
    
    PERF("applying_force_time = %.16f\n", applying_force_time);
    PERF("update_position_time = %.16f\n", update_position_time);
    PERF("rebuild_tree_time = %.16f\n", rebuild_tree_time);
    PERF("apply force: %.16f%%\n", applying_force_time / total_time * 100);
    PERF("update position: %.16f%%\n", update_position_time / total_time * 100);
    PERF("rebuild tree: %.16f%%\n", rebuild_tree_time / total_time * 100);
}

/**
 * draw QuadTree in the SpaceView
 */
void
BHSpaceModel::draw_quadTree(QuadTree *tree) {
    for (int i = 0; i < 4; i++) {
        if (tree->children[i]) {
            draw_quadTree(tree->children[i]);
        }
    }
    this->screen->draw_empty_rectangle(tree->bounds, RGB_BLUE);
}

void
BHSpaceModel::draw_bounds() {
    draw_quadTree(this->tree);
}

BHSpaceModel::~BHSpaceModel() {
    delete this->tree;
}
