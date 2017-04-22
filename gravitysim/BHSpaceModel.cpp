#include "BHSpaceModel.h"
#include <ctime>


BHSpaceModel::BHSpaceModel(RectangleD bounds, std::vector<Object> &objects) 
: SpaceModel(bounds, objects) {
    this->tree = new QuadTree(this->bounds);
    if (this->tree == NULL) {
        printf("unable to initialize QuadTree in SpaceModel\n");
        return;
    }
} 


double 
get_timediff (timespec &ts1, timespec &ts2) {
    double sec_diff = difftime(ts1.tv_sec, ts2.tv_sec);
    long nsec_diff = ts1.tv_nsec - ts2.tv_nsec;
    return sec_diff * 1000000000 + nsec_diff;
}


#define TIME_UTC 1

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
    
    printf("applying_force_time = %.16f\n", applying_force_time);
    printf("update_position_time = %.16f\n", update_position_time);
    printf("rebuild_tree_time = %.16f\n", rebuild_tree_time);
    printf("apply force: %.16f%%\n", applying_force_time / total_time * 100);
    printf("update position: %.16f%%\n", update_position_time / total_time * 100);
    printf("rebuild tree: %.16f%%\n", rebuild_tree_time / total_time * 100);
}


BHSpaceModel::~BHSpaceModel() {
    delete this->tree;
}
