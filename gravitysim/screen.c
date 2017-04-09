//
//  screen.c
//  GravitySim
//
//  Created by Krzysztof Gabis on 23.01.2013.
//  Copyright (c) 2013 Krzysztof Gabis. All rights reserved.
//

#include <stdlib.h>
//#include <GL/glfw.h>
#include "glfw.h"
#include "build_config.h"
#include "screen.h"

/**
 * init a screen with specified width and height
 */
Screen *screen_init(int width, int height) {
    Screen *new_screen = (Screen*)malloc(sizeof(Screen));
    if (!new_screen) {
        return NULL;
    }
    new_screen->pixels = (RGBColor*)malloc(width * height * sizeof(RGBColor));
    if (!new_screen->pixels) {
        free(new_screen);
        return NULL;
    }
    new_screen->width = width;
    new_screen->height = height;
    return new_screen;
}

/**
 * fill the screen with a specified color
 */
void screen_fill(Screen *screen, RGBColor color) {
    size_t i;
    for (i = 0; i < (screen->width * screen->height); i++) {
        screen->pixels[i] = color;
    }
}

/**
 * screen display
 */
void screen_display(Screen *screen) {
//    glClear( GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);
//    glLoadIdentity();
    glDrawPixels(screen->width, screen->height, GL_RGB, GL_FLOAT, screen->pixels);
    glfwSwapBuffers();
}

/**
 * dealloc a screen
 */
void screen_dealloc(Screen *screen) {
    free(screen->pixels);
    free(screen);
}
