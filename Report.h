#ifndef __REPORT_H__
#define __REPORT_H__

#include <vector>
#include "Perf.h"

class Report {
    private:
        std::vector<Perf> perfs;
    public:
        void addReport(Perf &perf);
        void showReport();
};
#endif
