#ifndef GravitySim_QuadTree_h
#define GravitySim_QuadTree_h

#include "basic_types.h"
#include "object.h"
#include "object_array.h"


class quadtree{
private:
    quadtree *children[4]; // four children
    RectangleD bounds;  // the bound of a QuadTree
    Object com;         // the common object of a QuadTree
    Object *object;     // temporarily store the object
    char has_children;  // has_children flag

public:
    quadtree(RectangleD bounds);

    ~quadtree();

    void add_objects(ObjectArray *objects) {
        size_t i;
        for (i = 0; i < objects->len; i++) {
            if (point2d_is_in_rectangled(objects->objects[i].position, this->bounds)) {
                this->add_object(&objects->objects[i]);
            }
        }
    }

    void add_object(Object *object);

    Point2D get_force_on_object(Object *object);

    void apply_to_objects(ObjectArray *objects, GS_FLOAT dt);
}

#endif
