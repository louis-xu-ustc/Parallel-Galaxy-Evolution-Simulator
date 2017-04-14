//
//  space_controller.c
//  GravitySim
//
//  Created by Krzysztof Gabis on 24.01.2013.
//  Copyright (c) 2013 Krzysztof Gabis. All rights reserved.
//

#include "space_controller.h"
#include "build_config.h"

/**
 * init a space controller with specified config
 */
SpaceController::SpaceController(SimulationConfig config) {
    this->model = new SpaceModel(config.model_bounds, config.view_bounds, config.galaxies_n, config.objects_n, config.galaxy_size);
    this->view = new SpaceView(config.view_bounds);
    this->loopTimes = config.loop_times;
}

/**
 * update the SpaceController with specified time interval dt
 */
SpaceController::void update(GS_FLOAT dt) {
    static GS_FLOAT last_update_time = 0.0;
    this->model.update(dt);
    last_update_time += dt;
    if (last_update_time >= (1.0 / MAX_FPS)) {
        this->view.clear();
        this->view.draw_objects(this->model->objects);
#if DRAW_QUADS
        this->view.draw_quadtree(this->model->tree);
#endif
#if PRINT_FPS
        printf("FPS: %.1f\n", 1.0 / last_update_time);
#endif
        this->view.display();
        last_update_time = 0.0;
    }
}

/**
 * dealloc SpaceController
 */
SpaceController::~SpaceController () {
    delete this->view;
    delete this->model;
}

SpaceController::size_t get_loop_times() {
    return this.loop_times;
}
