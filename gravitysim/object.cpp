#include <math.h>
#include <stdlib.h>
#include "object.h"
#include "build_config.h"


/**
 * make an object with the specified position, speed and mass
 */
Object 
Object::make(Point2D position, Point2D speed, GS_FLOAT mass) {
    Object obj;
    obj.position = position;
    obj.speed = speed;
    obj.mass = mass;
    return obj;
}

/**
 * randomly make an object with specified bounds, max_speed and max_mass
 */
Object 
Object::make_random(RectangleD bounds, GS_FLOAT max_speed, GS_FLOAT max_mass) {
    Point2D position = point2d_make_random(bounds);
    Point2D speed;
    speed.x = (((GS_FLOAT)rand()/(GS_FLOAT)RAND_MAX) - 0.5) * 2 * max_speed;
    speed.y = (((GS_FLOAT)rand()/(GS_FLOAT)RAND_MAX) - 0.5) * 2 * max_speed;
    GS_FLOAT mass = ((GS_FLOAT)rand()/(GS_FLOAT)RAND_MAX) * max_mass;
    return make(position, speed, mass);
}

/**
 * randomly make an obejct with specified bounds, max_speed and max_mass in an ellipse
 */
Object 
Object::make_random_in_ellipse(RectangleD bounds, GS_FLOAT max_speed, GS_FLOAT max_mass) {
    Point2D position = point2d_make_random_in_circle(bounds);
    Point2D speed;
    speed.x = (((GS_FLOAT)rand()/(GS_FLOAT)RAND_MAX) - 0.5) * 2 * max_speed;
    speed.y = (((GS_FLOAT)rand()/(GS_FLOAT)RAND_MAX) - 0.5) * 2 * max_speed;
    GS_FLOAT mass = ((GS_FLOAT)rand()/(GS_FLOAT)RAND_MAX) * max_mass;
    return make(position, speed, mass);
}

/**
 * object with zero position and mass
 */
Object 
Object::make_zero() {
    return make(point2d_zero(), point2d_zero(), 0.0);
}

/**
 * calculating the sum of two objects, weighted by mass, return the new object
 */
Object 
Object::add(Object a, Object b) {
    GS_FLOAT mass = a.mass + b.mass;
    Point2D position = point2d_add(point2d_multiply(a.position, a.mass / mass),
                             point2d_multiply(b.position, b.mass / mass));
    Point2D speed = point2d_add(point2d_multiply(a.speed, a.mass / mass),
                                point2d_multiply(b.speed, b.mass / mass));
    return make(position, speed, mass);
}

/**
 * calculate the force between object a and b
 */
Point2D 
Object::calculate_force(Object a, Object b) {
    Point2D dr;
    GS_FLOAT r, f;
    dr = point2d_sub(b.position, a.position);
    r = (point2d_length(dr) + SOFT_CONST);
    f = G_CONST * b.mass / SQUARE(r); //fastPow(r, GFACTOR);
    return point2d_multiply(dr, f/r);
}

/**
 * object's position update with a specified dt
 */
void 
Object::update_position(GS_FLOAT dt) {
    Point2D dr = point2d_multiply(this->speed, dt);
    this->position = point2d_add(this->position, dr);
}
