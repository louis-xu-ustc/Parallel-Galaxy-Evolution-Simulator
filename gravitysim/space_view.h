//
//  space_view.h
//  GravitySim
//
//  Created by Krzysztof Gabis on 24.01.2013.
//  Copyright (c) 2013 Krzysztof Gabis. All rights reserved.
//

#ifndef GravitySim_space_view_h
#define GravitySim_space_view_h

#include <stdlib.h>
#include "object.h"
#include "screen.h"
#include "quad_tree.h"
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
