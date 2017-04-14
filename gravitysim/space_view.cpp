//
//  space_view.c
//  GravitySim
//
//  Created by Krzysztof Gabis on 24.01.2013.
//  Copyright (c) 2013 Krzysztof Gabis. All rights reserved.
//

#include "space_view.h"
#include "glfw.h"
#include "build_config.h"
#include <stdio.h>

/**
 * init the SpaceView with specified bounds
 */
SpaceView::SpaceView(RectangleD bounds) {
    this->screen = new Screen(bounds.size.x, bounds.size.y);
    if (!this->screen) {
        printf("unable to initialize Screen in SpaceView\n");
        return;
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
void
SpaceView::clear() {
#if BG_BLACK_FG_WHITE
    this->screen->fill(RGB_BLACK);
#else
    this->screen->fill(RGB_WHITE);
#endif
}

/**
 * draw all the objects in the specified SpaceView
 */
void
SpaceView::draw_objects(ObjectArray *objects) {
    size_t i;
    Point2D pos;
    for (i = 0; i < objects->len; i++) {
        pos = objects->objects[i].position;
#if BG_BLACK_FG_WHITE

#if DRAW_SOLID
        this->screen->draw_set_pixel(pos.x, pos.y, RGB_WHITE);
#elif DRAW_BIG
        this->screen->draw_rectangle(rectanglei_make(pos.x, pos.y, 3, 3), RGB_WHITE);
#elif DRAW_LIGHTEN_OR_LIGHTEN
        this->screen->draw_lighten_pixel_bw(pos.x, pos.y, DARKEN_OR_LIGHTEN_STEP);
#endif

#else // white background and black foreground

#if DRAW_SOLID
        this->screen->draw_set_pixel(pos.x, pos.y, RGB_BLACK);
#elif DRAW_BIG
        this->screen->draw_rectangle(rectanglei_make(pos.x, pos.y, 3, 3), RGB_BLACK);
#elif DRAW_DARKEN_OR_LIGHTEN
        this->screen->draw_darken_pixel_bw(pos.x, pos.y, DARKEN_OR_LIGHTEN_STEP);
#endif

#endif
    }
}

/**
 * SpaceView display
 */
void
SpaceView::display() {
    this->screen->display();
}

/**
 * draw QuadTree in the SpaceView
 */
void
SpaceView::draw_quadTree(QuadTree *tree) {
    for (int i = 0; i < 4; i++) {
        if (tree->children[i]) {
            draw_quadTree(tree->children[i]);
        }
    }
    this->screen->draw_empty_rectangle(tree->bounds, RGB_BLUE);
}
