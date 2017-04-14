//
//  screen.h
//  GravitySim
//
//  Created by Krzysztof Gabis on 23.01.2013.
//  Copyright (c) 2013 Krzysztof Gabis. All rights reserved.
//

#ifndef GravitySim_screen_h
#define GravitySim_screen_h

#include <stdio.h>
#include "basic_types.h"

class Screen {
    private:
        int width;
        int height;
        RGBColor  *pixels;

    public:
        Screen(int width, int height);
        ~Screen();
        void fill(RGBColor color);
        void display();

        void draw_darken_pixel(int x, int y, float step);
        void draw_darken_pixel_bw(int x, int y, float step);
        void draw_lighten_pixel(int x, int y, float step);
        void draw_lighten_pixel_bw(int x, int y, float step);
        void draw_set_pixel(int x, int y, RGBColor color);
        RGBColor draw_get_pixel(int x, int y);
        void draw_rectangle(RectangleI rect, RGBColor color);
        void draw_empty_rectangle(RectangleD rect, RGBColor color);
};

#endif
