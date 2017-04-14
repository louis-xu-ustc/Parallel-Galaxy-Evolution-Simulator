/**
 * perf.c
 * @usage: generate the performance report for the gravity simulator
 * @author: yunpengx@andrew.cmu.edu
 */

#include "log.h"
#include "perf.h"
#include "glfw.h"
#include <stdlib.h>
#include <cstring>


static int log_level = LOG_INFO;
/**
 * init performance report
 */
perf_t* perf_init(int loop_times, std::string name) {
    if (loop_times <= 0) {
        return NULL;
    }
    perf_t *p = (perf_t*) malloc(sizeof(perf_t));
    if (!p) {
        ERR("not enough memory!");
        return NULL;
    }
    memcpy(&p->name, name.c_str(), 20);
    p->loop_times = loop_times;
    p->start_time = glfwGetTime();
    p->time_memcpy = 0.f;
    p->time_calc = 0.f;
    p->aver_time_per_loop = 0.f;
    return p;
}

/**
 * update performance report
 */
void perf_update(perf_t *p, int cnt) {
    p->loop_times = cnt;
}

/**
 * report performance after simulator terminates
 */
void perf_report(perf_t *p) {
   GS_DOUBLE curr_time = glfwGetTime();
   GS_DOUBLE total_time = curr_time - p->start_time;

   printf("gravitySim performance report for  %s:\n", p->name);
   printf("total loop times: %d\n", p->loop_times);
   printf("total time consumed: %f (ms)\n", total_time * 1000);
   printf("aver time per loop: %f (ms/loop)\n", total_time * 1000 / p->loop_times);
}

/**
 * deinit performance report
 */
void perf_deinit(perf_t *p) {
    if (p) {
        free(p);
        p = NULL;
    }
}
