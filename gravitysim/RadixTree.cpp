#include <algorithm>
#include <vector>
#include "RadixTree.h"

RadixTree::RadixTree() {
    this->internals = NULL;
    this->leafs = NULL;
    this->bounds = MODEL_BOUNDS;
}

RadixTree::RadixTree(RectangleD bouds) {
    this->internals = NULL;
    this->leafs = NULL;
    this->bounds = bouds;
}

RadixTree::~RadixTree() {

}

/**
 * fill the radix objects with objects and sort them according to mcode
 */
void
RadixTree::fillRadixObjects(std::vector<Object> objects) {
    for (int i = 0; i < objects.size(); i++) {
        Object o = objects[i];
        this->radixObjects.push_back(RadixObject(o.position, o.speed, o.mass, this->bounds));
    }
    std::sort(radixObjects.begin(), radixObjects.end(), RadixObjectEval());
}

/**
 * init hierarchy for the RadixTree
 */
void
RadixTree::initHierarchy() {
    int n = radixObjects.size();
   
    // the number of leafs is equal to radixObjects in a RadixTree
    leafs = new Node[n];

    // the number of internals is 1 less than the number of leafs
    internals = new Node[n-1];

    RadixObject *p = radixObjects.data();
    for (int i = 0; i < n; i++) {
        leafs[i].obj = p + i;
        radixObjects[i].nodePtr = &leafs[i]; 
    }
}

/**
 * deinit the hierarchy for the RadixTree
 */
void 
RadixTree::deinitHierarchy() {
    if (leafs != NULL) {
        delete[] leafs;
        leafs = NULL;
    }
    if (internals != NULL) {
        delete[] internals;
        internals = NULL;
    }
}

/**
 * generate the hierarchy 
 */
void
RadixTree::generateHierarchy() {
    int internal_size = radixObjects.size() - 1;
    for (int i = 0; i < internal_size; i++) {
        generateNode(i);
    }
}

/**
 * generate the node with specified idx
 */
void
RadixTree::generateNode(int idx) {
    Point2D range = determineRange(idx);

    int first = range.x;
    int last = range.y;

    // determine where to split the range
    int split = findSplit(first, last);
    Node *left, *right;

    //if the left Child is at the first position of the range, i.e. the
    //split is at the beg of the range, it will be a leaf node to object x
    if (split == first) {
        left = &leafs[split];
    } else {
        left = &internals[split];     
    }

    // the same to right Child
    if (split+1 == last) {
        right = &leafs[split+1];
    } else {
        right = &internals[split+1];
    }

    // Record parent-child relationships.
    internals[idx].left = left;
    internals[idx].right = right;
    
    left->parent = &internals[idx];
    right->parent = &internals[idx];
}

/**
 * Find out which range of objects the node corresponds to with a specified index
 */
Point2D 
RadixTree::determineRange(int idx) {
    int n = radixObjects.size() - 1; // size of internals

    if (idx == 0) {
        return point2d_make(0, n);
    }
    int initIdx = idx;

    unsigned int minusOneCode = radixObjects[idx-1].mcode;
    unsigned int preciousCode = radixObjects[idx].mcode;
    unsigned int plusOneCode = radixObjects[idx+1].mcode;

    if (minusOneCode == preciousCode && plusOneCode == preciousCode) {
        while (idx > 0 && idx < n) {
            idx += 1;
            if (idx >= n) {
                break;
            }
            if (radixObjects[idx].mcode != radixObjects[idx+1].mcode) {
                break;
            }
        }
        return point2d_make(initIdx, idx);
    }

    // check the direction we need to go and the minimal commonPrefix to start
    int dir; // direction to walk, 1 - to right, -1 - to left
    int min; // the minimal value of the search
    Point2D leftOrRight = point2d_make(CLZ(preciousCode ^ minusOneCode), 
            CLZ(preciousCode ^ plusOneCode));
    if (leftOrRight.x > leftOrRight.y) {
        dir = -1;
        min = leftOrRight.y;
    } else {
        dir = 1;
        min = leftOrRight.x;
    }

    // search and find the potential largest commonPrefix in that direction, the initial value is given 2,
    // since the minus and plus one element has already been compared
    int stepMax = 2;
    int testIdx = idx + dir * stepMax;
    while ((testIdx >= 0 && testIdx <= n) ? 
            (CLZ(preciousCode ^ radixObjects[testIdx].mcode) > min) : (false)) {
        stepMax *= 2;
        testIdx = idx + dir * stepMax;
    }

    // preciously find the new split
    int l = 0;
    for (int step = 2; step <= stepMax; step *= 2) {
        int t = stepMax / step;
        int newTest = idx + (l + t) * dir;
        if (newTest >= 0 && newTest <= n) {
            int newSplit = CLZ(preciousCode ^ radixObjects[newTest].mcode);
            if (newSplit > min) {
                l = l + t;
            }
        }
    }
    // return the correct value according to the direction
    if (dir == 1) {
        return point2d_make(idx, idx + l * dir);
    } else {
        return point2d_make(idx + l * dir, idx);
    }
}

/**
 * find highest differing zero within a range of these objects
 */
int
RadixTree::findSplit(int first, int last) {
    unsigned int firstCode = radixObjects[first].mcode;
    unsigned int lastCode = radixObjects[last].mcode;

    // Identical Morton codes => split the range in the middle.
    if (firstCode == lastCode) {
        return first;
    }
    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.
    int commonPrefix = CLZ(firstCode ^ lastCode);
    
    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.
    int split = first;
    int step = last - first;

    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;

        if (newSplit < last) {
            unsigned int splitCode = radixObjects[newSplit].mcode;
            int splitPrefix = CLZ(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix) {
                split = newSplit;
            }
        }
    } while (step > 1);
    return split;
}



