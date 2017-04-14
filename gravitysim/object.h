//
//  object.h
//  GravitySim
//
//  Created by Krzysztof Gabis on 24.01.2013.
//  Copyright (c) 2013 Krzysztof Gabis. All rights reserved.
//

#ifndef GravitySim_object_h
#define GravitySim_object_h

#include <stdbool.h>
#include "basic_types.h"

class Object {
private:
    Point2D position;
    Point2D speed;
    GS_FLOAT mass;

public:
    /**
     * make an object with the specified position, speed and mass
     */
    static Object make(Point2D position, Point2D speed, GS_FLOAT mass);

    /**
     * randomly make an object with specified bounds, max_speed and max_mass
     */
    static Object make_random(RectangleD bounds, GS_FLOAT max_speed, GS_FLOAT max_mass);

    /**
     * randomly make an obejct with specified bounds, max_speed and max_mass in an ellipse
     */
    static Object make_random_in_ellipse(RectangleD bounds, GS_FLOAT max_speed, GS_FLOAT max_mass);

    /**
     * object with zero position and mass
     */
    static Object make_zero();

    /**
     * calculating the sum of two objects, weighted by mass, return the new object
     */
    static Object object_add(Object a, Object b);

    /**
     * calculate the force between object a and b
     */
    static Point2D calculate_force(Object a, Object b);

    /**
     * object's position update with a specified dt
     */
    void update_position(GS_FLOAT dt);
}

#endif
