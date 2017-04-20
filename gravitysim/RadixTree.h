#ifndef __RADIXTREE_H__
#define __RADIXTREE_H__

#include "basic_types.h"
#include "Object.h"
#include <vector>
struct Node;

class RadixObject : public Object {
    public:
    /* morton code to represent an object */
    unsigned int mcode;
    
    /* link to a leaf node */
    Node *nodePtr;

    RadixObject (Point2D position, Point2D speed, GS_FLOAT mass, RectangleD rect) : Object(position, speed, mass) {
        // transfer a position to unified one
        Point2D p = point2d_unit_square(position, rect);
        mcode = mortan2D(p.x, p.y, rect);
        nodePtr = NULL;
    }
};

/* operator used to sort */
struct RadixObjectEval {
    bool operator() (const RadixObject &i, const RadixObject &j) const {
        return (i.mcode < j.mcode);
    }
};

/* Node structure */
struct Node {
    Node* left;     // left child
    Node* right;    // right child

    RadixObject *obj;    // pointer to RadixObject, if NULL, it's a internal node, else a leaf node
    
    Node *parent;   // pointer to parent Node
    Node() {
        left = NULL; 
        right = NULL;
        obj = NULL;
        parent = NULL;
    }
};

class RadixTree {
    private:
        std::vector<RadixObject> radixObjects;
        Node* internals;    // the number of internals is 1 less than the number of leafs
        Node* leafs;
        RectangleD bounds;
        void generateNode(int idx);
        int findSplit(int first, int last);
        Point2D determineRange(int idx);

    public:
        RadixTree();
        RadixTree(RectangleD bouds);
        ~RadixTree();
        void fillRadixObjects(std::vector<Object> objects);
        void initHierarchy();
        void deinitHierarchy();
        void generateHierarchy();

};
#endif
