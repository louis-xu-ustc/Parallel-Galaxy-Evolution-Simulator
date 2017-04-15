#include <stdio.h>
#include <stdlib.h>
#include "glfw.h"
#include <time.h>
#include "Screen.h"
#include "basic_types.h"
#include "SpaceController.h"
#include "build_config.h"
#include "SpaceModel.h"
#include "perf.h"


#define WINDOW_TITLE "GravitySim"
#define SUCCESS 0
#define FAILURE 1

static perf_t *p = NULL;
static int gl_init(int width, int height, const char *title);
static void gl_close(void);
bool execute_model(SpaceController *controller, SpaceModel *model);
void print_usage(const char *program_name);
SimulationConfig get_config(int argc, const char *argv[]);


int main(int argc, const char * argv[]) {
    bool loop = true;
    srand((unsigned)time(NULL));
    if (gl_init(WINDOW_W, WINDOW_H, WINDOW_TITLE) != SUCCESS) {
        return FAILURE;
    }
    SimulationConfig config = get_config(argc, argv);
    SpaceController *controller = new SpaceController(config);
    controller->generate_objects(config.view_bounds, config.galaxies_n, config.objects_n, config.galaxy_size);

    SpaceModel *seqBarnesHutModel = new SpaceModel(config.model_bounds, controller->get_objects());
    
    // TODO
    // SpaceModel *cudaBarnesHutModel = new SpaceModel(config.model_bounds, controller->get_objects());
    // SpaceModel *seqFMMModel = new SpaceModel(config.model_bounds, controller->get_objects());
    // SpaceModel *cudaFMMModel = new SpaceModel(config.model_bounds, controller->get_objects());

    if (!controller) {
        return FAILURE;
    }
    p = perf_init(config.loop_times, "reference");

    while (loop) {
        loop = execute_model(controller, seqBarnesHutModel);
    }
    perf_report(p);
    perf_deinit(p);
    // close gl after all things done
    gl_close();
    return SUCCESS;
}

bool execute_model(SpaceController *controller, SpaceModel *model) {
    GS_FLOAT old_time = glfwGetTime();
    GS_FLOAT current_time;
    GS_FLOAT dt;
    size_t cnt = 0;

    while (1) {
        current_time = glfwGetTime();
        dt = current_time - old_time;
        
        if(glfwGetKey(GLFW_KEY_ESC) || !glfwGetWindowParam(GLFW_OPENED) ||
                cnt >= controller->get_loop_times()) {
            perf_update(p, cnt);
            return false;
        }
        controller->update(dt, model);
        old_time = current_time;
        cnt++;
    }
    return true;
}

static int gl_init(int width, int height, const char *title) {
    int status;
    status = glfwInit();
    if (status != GL_TRUE) {
        return FAILURE;
    }
    status = glfwOpenWindow(width, height, 5, 6, 5, 0, 0, 0, GLFW_WINDOW);
    if (status != GL_TRUE) {
        return FAILURE;
    }
    glfwSetWindowTitle(title);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float aspect_ratio = ((float)height) / width;
    glFrustum(.5, -.5, -.5 * aspect_ratio, .5 * aspect_ratio, 1, 50);
    glMatrixMode(GL_MODELVIEW);
    return SUCCESS;
}

static void gl_close(void) {
    glfwTerminate();
}

void print_usage(const char *program_name) {
    printf("Usage:%s loop_times number_of_galaxies objects_per_galaxy galaxy_size\n", program_name);
    printf("Using default config.\n");
}

SimulationConfig get_config(int argc, const char *argv[]) {
    SimulationConfig config;
    config.loop_times = LOOP_TIMES;
    config.galaxies_n = GALAXY_NUM;
    config.galaxy_size = GALAXY_SIZE;
    config.model_bounds = MODEL_BOUNDS;
    config.view_bounds = WINDOW_BOUNDS;
    config.objects_n = OBJECT_NUM;
    if (argc != 5) {
        print_usage(argv[0]);
        return config;
    }

    config.loop_times = atoi(argv[1]);
    config.galaxies_n = atoi(argv[2]);
    config.objects_n = atoi(argv[3]);
    config.galaxy_size = atoi(argv[4]);
    return config;
}

