//
//  space_view.c
//  GravitySim
//
//  Created by Krzysztof Gabis on 24.01.2013.
//  Copyright (c) 2013 Krzysztof Gabis. All rights reserved.
//

#include "space_view.h"
#include "drawing.h"
#include "glfw.h"
#include "build_config.h"

/**
 * init the SpaceView with specified bounds
 */
SpaceView::SpaceView(RectangleD bounds) {
    this->screen = new Screen(bounds.size.x, bounds.size.y);
    if (!this->screen) {
        return NULL;
    }
}

/**
 * dealloc spaceview
 */
SpaceView::~SpaceView() {
    delete this->screen;
}

/**
 * clear the SpaceView with white color
 */
SpaceView::void clear() {
#if BG_BLACK_FG_WHITE
    screen_fill(this->screen, RGB_BLACK);
#else
    screen_fill(this->screen, RGB_WHITE);
#endif
}

/**
 * draw all the objects in the specified SpaceView
 */
SpaceView::void draw_objects(ObjectArray *objects) {
    size_t i;
    Point2D pos;
    for (i = 0; i < objects->len; i++) {
        pos = objects->objects[i].position;
#if BG_BLACK_FG_WHITE

#if DRAW_SOLID
        draw_set_pixel(this->screen, pos.x, pos.y, RGB_WHITE);
#elif DRAW_BIG
        draw_rectangle(this->screen, rectanglei_make(pos.x, pos.y, 3, 3), RGB_WHITE);
#elif DRAW_LIGHTEN_OR_LIGHTEN
        draw_lighten_pixel_bw(this->screen, pos.x, pos.y, DARKEN_OR_LIGHTEN_STEP);
#endif

#else // white background and black foreground

#if DRAW_SOLID
        draw_set_pixel(this->screen, pos.x, pos.y, RGB_BLACK);
#elif DRAW_BIG
        draw_rectangle(this->screen, rectanglei_make(pos.x, pos.y, 3, 3), RGB_BLACK);
#elif DRAW_DARKEN_OR_LIGHTEN
        draw_darken_pixel_bw(this->screen, pos.x, pos.y, DARKEN_OR_LIGHTEN_STEP);
#endif

#endif
    }
}

/**
 * SpaceView display
 */
SpaceView::void display() {
    this->screen.display();
}

/**
 * draw QuadTree in the SpaceView
 */
SpaceView::void draw_quadtree(QuadTree *tree) {
    for (int i = 0; i < 4; i++) {
        if (tree->children[i]) {
            draw_quadtree(tree->children[i]);
        }
    }
    draw_empty_rectangle(this->screen, tree->bounds, RGB_BLUE);
}
