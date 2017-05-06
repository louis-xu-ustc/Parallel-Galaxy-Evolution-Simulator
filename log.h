/**
 * File Name: log.h
 * Author: yunpengx
 * Mail: yunpengx@andrew.cmu.edu
 * Created Time: 04/09/17
 */

#ifndef __LOG_H__
#define __LOG_H__

#include <stdio.h>

typedef enum {
    LOG_ERR,
    LOG_MSG,
    LOG_INFO,
    LOG_PERF,
    LOG_DBG,
} LOG_T;

//printf("[%s:%d] " fmt "\n", __func__, __LINE__, ##__VA_ARGS__);
#define log_printf(log, fmt, ...) \
    do {    \
        if (log <= log_level) { \
            printf(fmt, ##__VA_ARGS__); \
        } \
    } while (0)

#define log_printf_verbose(log, fmt, ...) \
    do {    \
        if (log <= log_level) { \
            printf("[%s:%d] " fmt "\n", __func__, __LINE__, ##__VA_ARGS__); \
        } \
    } while (0)

#define ERR(fmt, ...)   \
    log_printf_verbose(LOG_ERR, fmt, ##__VA_ARGS__)

#define DBG(fmt, ...)   \
    log_printf_verbose(LOG_DBG, fmt, ##__VA_ARGS__)

#define PERF(fmt, ...)  \
    log_printf(LOG_PERF, fmt, ##__VA_ARGS__)

#define INFO(fmt, ...)  \
    log_printf(LOG_INFO, fmt, ##__VA_ARGS__)

#define ENTER() \
    DBG("[%s:%d] enter\n", __func__, __LINE__)

#define LEAVE() \
    DBG("[%s:%d] leave\n", __func__, __LINE__)

#endif //__LOG_H__
