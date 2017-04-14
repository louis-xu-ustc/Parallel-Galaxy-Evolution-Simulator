#include "space_model.h"
#include <stdlib.h>

SpaceModel::SpaceModel(RectangleD bounds, size_t capacity) {
    this->bounds = bounds;
    // change to new syntax; this is where dependency injection plays its part
    this->tree = new QuadTree(this->bounds);

    if (this->tree == NULL) {
        return NULL;
    }
    // change to new syntax
    this->objects = new objectarray(capacity);
    if (this->objects == NULL) {
        return NULL;
    }
    this->tree->add_objects(this->objects);
}

SpaceModel::SpaceModel(RectangleD bounds, RectangleD galaxies_bounds, size_t n_galaxies,
                                 size_t objects_per_galaxy, GS_FLOAT galaxy_size) {
    Point2D galaxy_pos;
    size_t i;
    for (i = 0; i < n_galaxies; i++) {
        galaxy_pos.x =  ((GS_FLOAT)rand()/(GS_FLOAT)RAND_MAX) * (galaxies_bounds.size.x - galaxy_size);
        galaxy_pos.y =  ((GS_FLOAT)rand()/(GS_FLOAT)RAND_MAX) * (galaxies_bounds.size.y - galaxy_size);
        add_galaxy(galaxy_pos, galaxy_size, objects_per_galaxy);
    }
    this->bounds = bounds;
    this->tree->add_objects(this->objects);
}

void 
SpaceModel::add_galaxy(Point2D position, GS_FLOAT size, size_t n) {
    size_t i;
    Point2D delta_pos, direction, speed_vector;
    GS_FLOAT distance;
    Object new_object;
    RectangleD bounds = rectangled_make(position.x, position.y, size, size);
    for (i = 0; i < n; i++) {
        new_object = object_make_random_in_ellipse(bounds, 0.0, MAX_MASS);
        delta_pos = point2d_sub(new_object.position, bounds.middle);
        direction = point2d_unit(delta_pos);
        distance = point2d_length(delta_pos);
        speed_vector = point2d_multiply(direction, distance); //yeah, that's primitive
        new_object.speed = point2d_rotate_90_ccw(speed_vector);
        this->objects->add(new_object);
    }
}

void 
SpaceModel::remove_objects_outside_bounds() {
    size_t i;
    for (i = 0; i < this->objects->len; i++) {
        if (!point2d_is_in_rectangled(this->objects->objects[i].position, this->bounds)) {
            this->objects->remove_object_at(i);
            i--;
        }
    }
}

void 
SpaceModel::update(GS_FLOAT dt) {
    size_t i;
#ifdef CONST_TIME
    dt = CONST_TIME;
#endif
    this->tree->apply_to_objects(this->objects, dt);
    for (i = 0; i < this->objects->len; i++) {
        &this->objects->objects[i]->update_position(dt);
    }
    remove_objects_outside_bounds();
    delete this->tree;
    this->tree = new QuadTree(this->bounds);
    this->tree->add_objects(this->objects);
}

~SpaceModel::SpaceModel() {
    delete this->objects;
    delete this->tree;
}
