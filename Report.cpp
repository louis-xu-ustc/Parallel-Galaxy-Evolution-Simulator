#include "Report.h"

void
Report::addReport(Perf &perf) {
    this->perfs.push_back(perf);
}

void
Report::showReport() {

    for (int i = 0; i < (int)perfs.size(); i++) {
        int loop_times = perfs[i].getLoopTimes();
        GS_DOUBLE total_time = perfs[i].getEndTime() - perfs[i].getStartTime();
        printf("gravitySim performance report for  %s:\n", perfs[i].getName().c_str());
        printf("total loop times: %d\n", loop_times);
        printf("total time consumed: %f (ms)\n",  (total_time) * 1000);
        printf("aver time per loop: %f (ms/loop)\n", total_time * 1000 / loop_times);
    }
}
