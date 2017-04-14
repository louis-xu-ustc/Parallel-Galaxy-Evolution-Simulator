#ifndef GravitySim_QuadTree_h
#define GravitySim_QuadTree_h

#include "basic_types.h"
#include "object.h"
#include "object_array.h"


class QuadTree {
    private:
        

    public:
        QuadTree *children[4]; // four children
        RectangleD bounds;  // the bound of a QuadTree
        Object com;         // the common object of a QuadTree
        Object *object;     // temporarily store the object
        char has_children;  // has_children flag

        QuadTree(RectangleD bounds);

        ~QuadTree();

        void add_objects(ObjectArray *objects);

        void add_object(Object *object);

        Point2D get_force_on_object(Object *object);

        void apply_to_objects(ObjectArray *objects, GS_FLOAT dt);
};

#endif
