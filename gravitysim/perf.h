#ifndef __PERF_H__
#define __PERF_H__

#include "build_config.h"
#include <string>

typedef struct perf_s {
    char name[20];
    int loop_times;                 // loop times
    GS_DOUBLE start_time;             // start time 
    GS_DOUBLE time_memcpy;           // time for memcpy
    GS_DOUBLE time_calc;      // time for calculation
    GS_DOUBLE aver_time_per_loop;    // average time per loop

} perf_t;

perf_t* perf_init(int loop_times, std::string name);
void perf_update(perf_t *p, int cnt);
void perf_report(perf_t *p);
void perf_deinit(perf_t *p);
#endif
