#ifndef __PERF_H__
#define __PERF_H__

#include "build_config.h"
#include <string>

class Perf {
    private:
        std::string name;
        int loop_times;                 // loop times
        GS_DOUBLE start_time;             // start time 
        GS_DOUBLE end_time;               // end time
        GS_DOUBLE time_memcpy;           // time for memcpy
        GS_DOUBLE time_calc;            // time for calculation
        GS_DOUBLE aver_time_per_loop;    // average time per loop
    public:
        Perf(int loop_times, std::string name);
        std::string getName();
        GS_DOUBLE getStartTime();
        GS_DOUBLE getEndTime();
        int getLoopTimes();
        void update_time(int cnt);
        static void update(int cnt, Perf *perf);
        void update();
};
#endif
