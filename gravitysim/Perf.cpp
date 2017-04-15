/**
 * perf.c
 * @usage: generate the performance report for the gravity simulator
 * @author: yunpengx@andrew.cmu.edu
 */

#include "log.h"
#include "Perf.h"
#include "glfw.h"
#include <stdlib.h>
#include <cstring>


/**
 * init performance report
 */
Perf::Perf(int loop_times, std::string name) {
    if (loop_times <= 0) {
        return;
    }
    this->name = name;
    this->loop_times = loop_times;
    this->start_time = glfwGetTime();
    this->time_memcpy = 0.f;
    this->time_calc = 0.f;
    this->aver_time_per_loop = 0.f;
}

/**
 * update performance report
 */
void 
Perf::update_time(int cnt) {
    this->loop_times = cnt;
    this->end_time = glfwGetTime();
    // TODO add more
}

void
Perf::update(int cnt, Perf *perf) {
    perf->update_time(cnt);
}

std::string
Perf::getName() {
    return this->name;
}

GS_DOUBLE
Perf::getStartTime() {
    return this->start_time;
}

GS_DOUBLE
Perf::getEndTime() {
    return this->end_time;
}

int 
Perf::getLoopTimes() {
    return this->loop_times;
}
