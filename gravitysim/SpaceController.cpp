#include "SpaceController.h"
#include "build_config.h"

/**
 * init a space controller with specified config
 */
SpaceController::SpaceController(SimulationConfig config) {
    this->view = new SpaceView(config.view_bounds);
    this->loop_times = config.loop_times;
}

/**
 * update the SpaceController with specified time interval dt
 */
void
SpaceController::update(GS_FLOAT dt, SpaceModel *model) {
    static GS_FLOAT last_update_time = 0.0;
    printf("before calling update\n");
    model->update(dt);
    last_update_time += dt;
    if (last_update_time >= (1.0 / MAX_FPS)) {
        this->view->clear();
        this->view->draw_objects(model->objects);
#if DRAW_QUADS
        this->view->draw_quadtree(model->tree);
#endif
#if PRINT_FPS
        printf("FPS: %.1f\n", 1.0 / last_update_time);
#endif
        this->view->display();
        last_update_time = 0.0;
    }
}

/**
 * dealloc SpaceController
 */
SpaceController::~SpaceController () {
    delete this->view;
}

size_t
SpaceController::get_loop_times() {
    return this->loop_times;
}

void
SpaceController::generate_objects(RectangleD galaxies_bounds, size_t n_galaxies,
                                 size_t objects_per_galaxy, GS_FLOAT galaxy_size) {
    Point2D galaxy_pos;
    size_t i;
    for (i = 0; i < n_galaxies; i++) {
        galaxy_pos.x =  ((GS_FLOAT)rand()/(GS_FLOAT)RAND_MAX) * (galaxies_bounds.size.x - galaxy_size);
        galaxy_pos.y =  ((GS_FLOAT)rand()/(GS_FLOAT)RAND_MAX) * (galaxies_bounds.size.y - galaxy_size);
        add_galaxy(galaxy_pos, galaxy_size, objects_per_galaxy);
    }
}

void 
SpaceController::add_galaxy(Point2D position, GS_FLOAT size, size_t n) {
    size_t i;
    Point2D delta_pos, direction, speed_vector;
    GS_FLOAT distance;
    Object new_object;
    RectangleD bounds = rectangled_make(position.x, position.y, size, size);
    for (i = 0; i < n; i++) {
        new_object = Object::make_random_in_ellipse(bounds, 0.0, MAX_MASS);
        delta_pos = point2d_sub(new_object.position, bounds.middle);
        direction = point2d_unit(delta_pos);
        distance = point2d_length(delta_pos);
        speed_vector = point2d_multiply(direction, distance); //yeah, that's primitive
        new_object.speed = point2d_rotate_90_ccw(speed_vector);
        this->objects.push_back(new_object);
    }
}


std::vector<Object>& 
SpaceController::get_objects() {
    return this->objects;
}

