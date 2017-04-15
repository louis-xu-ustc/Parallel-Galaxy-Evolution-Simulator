#ifndef GravitySim_space_view_h
#define GravitySim_space_view_h

#include <stdlib.h>
#include "Object.h"
#include "Screen.h"
#include "QuadTree.h"
#include <vector>

class SpaceView {
    private:
        Screen *screen;
    public:
        SpaceView(RectangleD bounds);
        ~SpaceView();
        void clear();
        void draw_objects(std::vector<Object> &objects);
        void draw_quadTree(QuadTree *tree);
        void display();
};
#endif
