//
//  object_array.h
//  GravitySim
//
//  Created by Krzysztof Gabis on 25.01.2013.
//  Copyright (c) 2013 Krzysztof Gabis. All rights reserved.
//

#ifndef GravitySim_object_array_h
#define GravitySim_object_array_h

#include "object.h"
#include "basic_types.h"


class ObjectArray {
private:
    Object *objects;
    size_t len;
    size_t capacity;

public:
    /**
     * init an obejct array with a specified capacity
     */
    ObjectArray(size_t capacity);

    /**
     * remove an object at a specified index i, replace the i-th object with the last obejct in the ObjectArray
     */
    void remove_object_at(size_t i);

    /**
     * add object into the ObjectArray
     */
    void add(Object object);
};

#endif
