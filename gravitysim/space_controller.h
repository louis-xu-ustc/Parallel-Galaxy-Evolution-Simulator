//
//  space_controller.h
//  GravitySim
//
//  Created by Krzysztof Gabis on 24.01.2013.
//  Copyright (c) 2013 Krzysztof Gabis. All rights reserved.
//

#ifndef GravitySim_space_controller_h
#define GravitySim_space_controller_h

#include "space_view.h"
#include "space_model.h"

typedef struct {
    size_t loop_times;
    RectangleD view_bounds;
    RectangleD model_bounds;
    size_t objects_n;
    size_t galaxies_n;
    GS_FLOAT galaxy_size;
} SimulationConfig;

class SpaceController {
    private:
        size_t loop_times;
        SpaceView *view;
        SpaceModel *model;

    public:
        size_t get_loop_times();
        SpaceController(SimulationConfig config);
        ~SpaceController();
        void update(GS_FLOAT dt);
};
#endif
