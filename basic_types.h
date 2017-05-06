#ifndef GravitySim_graphic_types_h
#define GravitySim_graphic_types_h

#define RGB_WHITE rgbcolor_make(1.0, 1.0, 1.0)
#define RGB_BLACK rgbcolor_make(0.0, 0.0, 0.0)
#define RGB_RED rgbcolor_make(1.0, 0.0, 0.0)
#define RGB_GREEN rgbcolor_make(0.0, 1.0, 0.0)
#define RGB_BLUE rgbcolor_make(0.0, 0.0, 1.0)

#define SQUARE(x) ((x)*(x))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#include "build_config.h"
#include <math.h>
#include <algorithm>
#include <ctime>

typedef struct {
    int x;
    int y;
} PointI;

typedef struct {
    PointI origin;
    PointI size;
} RectangleI;

typedef struct {
    GS_FLOAT x;
    GS_FLOAT y;
} Point2D;

typedef struct {
    Point2D origin;
    Point2D size;
    Point2D middle;
} RectangleD;

typedef struct {
    float red;
    float green;
    float blue;
} RGBColor;

double fastPow(double a, double b);
Point2D point2d_make_random(RectangleD bounds);
Point2D point2d_make_random_in_circle(RectangleD bounds);

// Functions to inline
static GS_INLINE PointI pointi_make(int x, int y) {
    PointI point;
    point.x = x;
    point.y = y;
    return point;
}

static GS_INLINE RectangleI rectanglei_make_with_points(PointI origin, PointI size) {
    RectangleI rect;
    rect.origin = origin;
    rect.size = size;
    return rect;
}

static GS_INLINE RectangleI rectanglei_make(int x1, int y1, int x2, int y2) {
    return rectanglei_make_with_points(pointi_make(x1, y1), pointi_make(x2, y2));
}

static GS_INLINE Point2D point2d_make(GS_FLOAT x, GS_FLOAT y) {
    Point2D point;
    point.x = x;
    point.y = y;
    return point;
}

static GS_INLINE Point2D point2d_zero() {
    return point2d_make(0.0, 0.0);
}

static GS_INLINE Point2D point2d_add(Point2D a, Point2D b) {
    return point2d_make(a.x + b.x, a.y + b.y);
}

static GS_INLINE Point2D point2d_sub(Point2D a, Point2D b) {
    return point2d_make(a.x - b.x, a.y - b.y);
}

static GS_INLINE Point2D point2d_multiply(Point2D a, GS_FLOAT x) {
    return point2d_make(a.x * x, a.y * x);
}

static GS_INLINE int point2d_is_in_rectangled(Point2D point, RectangleD rect) {
    return point.x >= rect.origin.x
        && point.x < (rect.size.x + rect.origin.x)
        && point.y >= rect.origin.y
        && point.y < (rect.size.y + rect.origin.y);
}

static GS_INLINE Point2D point2d_rotate_90_ccw(Point2D p) {
    return point2d_make(-p.y, p.x);
}

static GS_INLINE  GS_FLOAT point2d_length(Point2D p) {
    return sqrt((p.x * p.x) + (p.y * p.y));
}

static GS_INLINE  Point2D point2d_unit(Point2D p) {
    return point2d_multiply(p, 1.0 / point2d_length(p));
}

static GS_INLINE int point2d_nquad_of_rectangled(Point2D p, RectangleD r) {
    return ((p.y >= r.middle.y) << 1) | (p.x >= r.middle.x);
}

static GS_INLINE RectangleD rectangled_make_with_point2ds(Point2D origin, Point2D size) {
    RectangleD rect;
    rect.origin = origin;
    rect.size = size;
    rect.middle = point2d_make(origin.x + (size.x / 2), origin.y + (size.y / 2));
    return rect;
}

static GS_INLINE RectangleD rectangled_make(GS_FLOAT x, GS_FLOAT y, GS_FLOAT size_x, GS_FLOAT size_y) {
    return rectangled_make_with_point2ds(point2d_make(x, y),
                                         point2d_make(size_x, size_y));
}

static GS_INLINE RectangleD rectangled_nquad(RectangleD rect, int quarter) {
    Point2D start = rect.origin;
    Point2D end = point2d_add(rect.origin, rect.size);
    Point2D quarter_size = point2d_multiply(point2d_sub(end, rect.origin), 0.5);
    Point2D middle = rect.middle;
    switch (quarter) {
        case 0:
            return rectangled_make_with_point2ds(start, quarter_size);
        case 1:
            return rectangled_make(middle.x, start.y, quarter_size.x, quarter_size.y);
        case 2:
            return rectangled_make(start.x, middle.y, quarter_size.x, quarter_size.y);
        case 3: default:
            return rectangled_make_with_point2ds(middle, quarter_size);
    }
}

static GS_INLINE RGBColor rgbcolor_make(float red, float green, float blue) {
    RGBColor color;
    color.red = red;
    color.green = green;
    color.blue = blue;
    return color;
}

static GS_INLINE Point2D point2d_unit_square(Point2D p, RectangleD rect) {
    return point2d_make((p.x - rect.origin.x) / rect.size.x, (p.y - rect.origin.y) / rect.size.y);
}

static GS_INLINE RectangleD rectangled_incr_bound(RectangleD rect, Point2D pos) {
    Point2D lower = rect.origin;
    Point2D upper = point2d_add(lower, rect.size);
    if (pos.x < lower.x) {
        lower.x = pos.x;
    }
    if (pos.y < lower.y) {
        lower.y = pos.y;
    }
    if (pos.x > upper.x) {
        upper.x = pos.x;
    }
    if (pos.y > upper.y) {
        upper.y = pos.y;
    }
    unsigned int size_x = upper.x - lower.x;
    unsigned int size_y = upper.y - lower.y;
    return rectangled_make(lower.x, lower.y, size_x, size_y);
}

static GS_INLINE RectangleD rectangled_incr_bound(Point2D pos) {
    return rectangled_make(pos.x, pos.y, 0, 0);
}

/**
 * Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit
 * [Referece] https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
 */
static GS_INLINE unsigned int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

/**
 * Calculates a 30-bit Morton code for the given 2D point located within the unit square [0,1].
 * [Referece] https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
 */
static GS_INLINE unsigned long mortan2D(GS_FLOAT x, GS_FLOAT y, RectangleD rect) {
    GS_DOUBLE xx = std::min(std::max(x * rect.size.x, 0.0f), rect.size.x - 1);
    GS_DOUBLE yy = std::min(std::max(y * rect.size.y, 0.0f), rect.size.y - 1);
    unsigned int xxHigh = (unsigned int)(((unsigned long)xx) >> 32);
    unsigned int xxLow = (unsigned int)((unsigned long)xx & 0x00000000FFFFFFFF);
    unsigned int yyHigh = (unsigned int)(((unsigned long)yy) >> 32);
    unsigned int yyLow = (unsigned int)((unsigned long)yy & 0x00000000FFFFFFFF);
    unsigned long e_xx = (((unsigned long)expandBits(xxHigh)) << 32) | (expandBits(xxLow));
    unsigned long e_yy = (((unsigned long)expandBits(yyHigh)) << 32) | (expandBits(yyLow));
    return 2 * e_xx + e_yy;
}

/**
 * This algorithm uses a hybrid approach of bi-section to find out which 8-bit chunk of the 32-bit number contains the first 1-bit, which is followed by a lookup table clz_lkup[] to find the first 1-bit within the byte
 * [Referece] http://embeddedgurus.com/state-space/2014/09/fast-deterministic-and-portable-counting-leading-zeros/
 */
static GS_INLINE unsigned int CLZ(unsigned int x) {
    static unsigned char const clz_lkup[] = {
        32U, 31U, 30U, 30U, 29U, 29U, 29U, 29U,
        28U, 28U, 28U, 28U, 28U, 28U, 28U, 28U,
        27U, 27U, 27U, 27U, 27U, 27U, 27U, 27U,
        27U, 27U, 27U, 27U, 27U, 27U, 27U, 27U,
        26U, 26U, 26U, 26U, 26U, 26U, 26U, 26U,
        26U, 26U, 26U, 26U, 26U, 26U, 26U, 26U,
        26U, 26U, 26U, 26U, 26U, 26U, 26U, 26U,
        26U, 26U, 26U, 26U, 26U, 26U, 26U, 26U,
        25U, 25U, 25U, 25U, 25U, 25U, 25U, 25U,
        25U, 25U, 25U, 25U, 25U, 25U, 25U, 25U,
        25U, 25U, 25U, 25U, 25U, 25U, 25U, 25U,
        25U, 25U, 25U, 25U, 25U, 25U, 25U, 25U,
        25U, 25U, 25U, 25U, 25U, 25U, 25U, 25U,
        25U, 25U, 25U, 25U, 25U, 25U, 25U, 25U,
        25U, 25U, 25U, 25U, 25U, 25U, 25U, 25U,
        25U, 25U, 25U, 25U, 25U, 25U, 25U, 25U,
        24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
        24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
        24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
        24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
        24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
        24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
        24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
        24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
        24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
        24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
        24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
        24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
        24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
        24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
        24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
        24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U
    };
    unsigned int n;
    if (x >= (1U << 16)) {
        if (x >= (1U << 24)) {
            n = 24U;
        }
        else {
            n = 16U;
        }
    }
    else {
        if (x >= (1U << 8)) {
            n = 8U;
        }
        else {
            n = 0U;
        }
    }
    return (unsigned int)clz_lkup[x >> n] - n;
}

/**
 * get the level mask used for constructing an OrcTree
 */
static unsigned long mask_table[] = {
    0x0000000000000000u,

    0x3000000000000000u,
    0x3C00000000000000u,
    0x3F00000000000000u,
    0x3FC0000000000000u,
    0x3FF0000000000000u,
    0x3FFC000000000000u,
    0x3FFF000000000000u,
    0x3FFFC00000000000u,
    0x3FFFF00000000000u,
    0x3FFFFC0000000000u,
    0x3FFFFF0000000000u,
    0x3FFFFFC000000000u,
    0x3FFFFFF000000000u,
    0x3FFFFFFC00000000u,
    0x3FFFFFFF00000000u,
    
    0x3FFFFFFF30000000u,
    0x3FFFFFFF3C000000u,
    0x3FFFFFFF3F000000u,
    0x3FFFFFFF3FC00000u,
    0x3FFFFFFF3FF00000u,
    0x3FFFFFFF3FFC0000u,
    0x3FFFFFFF3FFF0000u,
    0x3FFFFFFF3FFFC000u,
    0x3FFFFFFF3FFFF000u,
    0x3FFFFFFF3FFFFC00u,
    0x3FFFFFFF3FFFFF00u,
    0x3FFFFFFF3FFFFFC0u,
    0x3FFFFFFF3FFFFFF0u,
    0x3FFFFFFF3FFFFFFCu,
    0x3FFFFFFF3FFFFFFFu,
};

static GS_INLINE unsigned long get_level_mask(int level) {
    if (level <= 0) {
        return 0x0000000000000000u;
    }
    if (level >= 30) {
        return 0x3FFFFFFF3FFFFFFFu;
    }
    return mask_table[level];
}

static GS_INLINE GS_DOUBLE
get_timediff (timespec &ts1, timespec &ts2) {
    double sec_diff = difftime(ts1.tv_sec, ts2.tv_sec);
    long nsec_diff = ts1.tv_nsec - ts2.tv_nsec;
    return sec_diff * 1000000000 + nsec_diff;
}

#endif
