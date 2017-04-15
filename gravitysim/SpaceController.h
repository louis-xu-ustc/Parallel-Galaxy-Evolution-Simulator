#ifndef GravitySim_space_controller_h
#define GravitySim_space_controller_h

#include "SpaceView.h"
#include "SpaceModel.h"
#include "Object.h"

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
        std::vector<Object> objects;

    public:
        size_t get_loop_times();
        SpaceController(SimulationConfig config);
        ~SpaceController();
        void update(GS_FLOAT dt, SpaceModel *model);
        void generate_objects(RectangleD galaxies_bounds, size_t n_galaxies,
                                 size_t objects_per_galaxy, GS_FLOAT galaxy_size);
        void add_galaxy(Point2D position, GS_FLOAT size, size_t n);
        std::vector<Object>& get_objects();
};
#endif
