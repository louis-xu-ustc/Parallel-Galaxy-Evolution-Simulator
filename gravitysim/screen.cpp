//
//  screen.c
//  GravitySim
//
//  Created by Krzysztof Gabis on 23.01.2013.
//  Copyright (c) 2013 Krzysztof Gabis. All rights reserved.
//

#include <stdlib.h>
#include "glfw.h"
#include "build_config.h"
#include "screen.h"

/**
 * init a screen with specified width and height
 */
Screen::Screen(int width, int height) {
    this->width = width;
    this->height = height;
    this->pixels = (RGBColor*)malloc(width * height * sizeof(RGBColor));
    if (!this->pixels) {
        printf("unable to initialize pixels in Screen\n");
        return;
    }
}

Screen::~Screen() {
    if (this->pixels) {
        free(this->pixels);
    }
}

/**
 * fill the screen with a specified color
 */
void
Screen::fill(RGBColor color) {
    int i;
    for (i = 0; i < (this->width * this->height); i++) {
        this->pixels[i] = color;
    }
}

/**
 * screen display
 */
void
Screen::display() {
    //    glClear( GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);
    //    glLoadIdentity();
    glDrawPixels(this->width, this->height, GL_RGB, GL_FLOAT, this->pixels);
    glfwSwapBuffers();
}


void
Screen::draw_darken_pixel_bw(int x, int y, float step) {
    if (x < 0 ||
            y < 0 ||
            x >= this->width ||
            y >= this->height) {
        return;
    }
    float color = draw_get_pixel(x, y).red;
    color = color - step;
    color = MAX(0, color);
    draw_set_pixel(x, y, rgbcolor_make(color, color, color));
}

void
Screen::draw_lighten_pixel_bw(int x, int y, float step) {
    if (x < 0 || y < 0 || x >= this->width || y >= this->height) {
        return;
    }

    float color = draw_get_pixel(x, y).red;
    color = color + step;
    color = MIN(1, color);
    draw_set_pixel(x, y, rgbcolor_make(color, color, color));
}

void
Screen::draw_darken_pixel(int x, int y, float step) {
    if (x < 0 ||
            y < 0 ||
            x >= this->width ||
            y >= this->height) {
        return;
    }
    RGBColor color = draw_get_pixel(x, y);
    color.red = color.red - step;
    color.red = MAX(0, color.red);
    color.green = color.green - step;
    color.green = MAX(0, color.green);
    color.blue = color.blue - step;
    color.blue = MAX(0, color.blue);
    draw_set_pixel(x, y, color);
}

void
Screen::draw_lighten_pixel(int x, int y, float step) {
    if (x < 0 ||
            y < 0 ||
            x >= this->width ||
            y >= this->height) {
        return;
    }
    RGBColor color = draw_get_pixel(x, y);
    color.red = color.red + step;
    color.red = MIN(1.0, color.red);
    color.green = color.green + step;
    color.green = MIN(1.0, color.green);
    color.blue = color.blue + step;
    color.blue = MIN(1.0, color.blue);
    draw_set_pixel(x, y, color);
}

GS_INLINE void
Screen::draw_set_pixel(int x, int y, RGBColor color) {
    if (x < 0 ||
            y < 0 ||
            x >= this->width ||
            y >= this->height) {
        return;
    }
    int index = y * this->width + x;
    this->pixels[index] = color;
}

GS_INLINE RGBColor
Screen::draw_get_pixel(int x, int y) {
    if (x < 0 ||
            y < 0 ||
            x >= this->width ||
            y >= this->height) {
        return RGB_WHITE;
    }
    int index = y * this->width + x;
    return this->pixels[index];
}

void
Screen::draw_rectangle(RectangleI r, RGBColor color) {
    for (int x = r.origin.x; x < r.origin.x + r.size.x; x++) {
        for (int y = r.origin.y; y < r.origin.y + r.size.y; y++) {
            draw_set_pixel(x, y, color);
        }
    }
}

void
Screen::draw_empty_rectangle(RectangleD rect, RGBColor color) {
    int x, y;
    Point2D end = point2d_add(rect.origin, rect.size);
    for (x = rect.origin.x; x < end.x; x++) {
        draw_set_pixel(x, rect.origin.y, color);
        draw_set_pixel(x, end.y, color);
    }
    for (y = rect.origin.y; y < end.y; y++) {
        draw_set_pixel(rect.origin.x, y, color);
        draw_set_pixel(end.x, y, color);
    }
}
