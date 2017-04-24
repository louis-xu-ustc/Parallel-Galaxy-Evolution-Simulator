#ifndef __MORTON_TREE_H__
#define __MORTON_TREE_H__

#include "basic_types.h"
#include "Object.h"
#include <vector>

class MortonTreeObject : public Object {
    public:
        unsigned int mcode;
        // store the index of cell that contians the MortonTreeObject
        int parent;

        MortonTreeObject(Point2D pos, Point2D speed, GS_FLOAT mass, RectangleD rect) : Object(pos, speed, mass) {
            // transfer a position to the unified format
            Point2D p = point2d_unit_square(pos, rect);
            mcode = mortan2D(p.x, p.y, rect);
            parent = -1;
        }
};

/**
 * operator used to sort MortonTreeObject
 */
struct MortonTreeObjectEval {
    bool operator ()(const MortonTreeObject *i, const MortonTreeObject *j) const {
        return (i->mcode < j->mcode);
    }
};

/**
 * MortonCell class
 */
class MortonCell {
    public:
        // indicates whether the Cell is a leaf or node
        bool is_leaf;       

        // if the cell is leaf, then first_index represents the index of first object
        // if the cell is node, then first_index represents the index of first child cell
        int first_index;
        
        // if the cell is a leaf, count the number of particles
        // if the cell is a node, count the number of children
        int count;

        // store the index of objects for all the children
        std::vector<int> children;

        // store the index of parent for this cell
        int parent;

        // the common object to represent a Morton cell
        Object com;     
        
        // the bound of a Morton Cell
        RectangleD bound;

        // the level of this cell
        int level;

        MortonCell() {
            is_leaf = false;
            first_index = 0;
            count = 0;
            com = Object::make_zero();
            level = 0;
        }
};

/**
 * cellInfo: a helper structure for constructing MortonTree
 */
struct CellInfo {
    int level;
    int start;
    int size;
    int parent;
    
    CellInfo(int _level, int _start, int _size, int _parent) {
        level = _level;
        start = _start;
        size = _size;
        parent = _parent;
    }
};

/**
 * MortonTree class
 */
class MortonTree {
    private:
        std::vector<MortonTreeObject*> mortonObjects;
        std::vector<MortonCell*> cells;
        RectangleD bound;

        // update the max bound for a MortonCell specified via index
        void updateMaxBound(int index);
        bool isValidObjectsIndex(int start, int size);

    public:
        MortonTree(RectangleD bound);
        ~MortonTree();

        std::vector<MortonCell*>& getCells();
        std::vector<MortonTreeObject*>& getObjects();
        // fill in MortonTreeObject vector using the specified objects
        void fillMortonTreeObjects(std::vector<Object> &objects);
        // generate the whole MortonTree, until all the cells become leafs or the max depth of the tree reaches
        void generateMortonTree();
        // generate the MortonCell based on the CellInfo
        void generateMortonCell(std::vector<CellInfo> &old_info, std::vector<CellInfo> &new_info);
        void generateMortonCell(int level, int start, int size);
        // calculate the total force exerted on the object
        Point2D getForceOnObject(int obj_idx);
        // update all the objects' speed
        void applyToObjects(GS_FLOAT dt);
        // traverse the tree
        void traverseCells();
        // traverse the objects
        void traverseObjects();
};

#endif
