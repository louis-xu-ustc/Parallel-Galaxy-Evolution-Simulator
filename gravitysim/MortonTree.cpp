#include <algorithm>
#include <assert.h>     /* assert */
#include <vector>
#include "MortonTree.h"
#include "log.h"

static int log_level = LOG_INFO;

MortonTree::MortonTree(RectangleD bound) {
    //INFO("bound.x:%f, bound.y:%f, bound.size.x:%f, bound.size.y:%f\n", bound.origin.x, bound.origin.y, bound.size.x, bound.size.y);
    this->bound = bound;
}

MortonTree::~MortonTree() {
    for (int i = 0; i < this->cells.size(); i++) {
        MortonCell *cell = this->cells[i];
        if (cell != NULL) {
            delete cell;
        }
    }
    for (int i = 0; i < this->mortonObjects.size(); i++) {
        MortonTreeObject *obj = this->mortonObjects[i];
        if (obj != NULL) {
            delete obj;
        }
    }
}

// fill in MortonTreeObject vector using the specified objects
void 
MortonTree::fillMortonTreeObjects(std::vector<Object> &objects) {
    ENTER();
    int n = objects.size();
    for (int i = 0; i < n; i++) {
        Object *o = &objects[i];
        this->mortonObjects.push_back(new MortonTreeObject(o->position, o->speed, o->mass, this->bound));
    }
    std::sort(this->mortonObjects.begin(), this->mortonObjects.end(), MortonTreeObjectEval());
    LEAVE();
}

// generate the whole MortonTree, until all the cells become leafs or the max depth of the tree reaches
void 
MortonTree::generateMortonTree() {
    ENTER();
    if (mortonObjects.size() == 0) {
        //ERR("No mortonObjects available\n");
        return;
    }

    std::vector<CellInfo> old_info;
    std::vector<CellInfo> new_info;
    MortonCell *cell = new MortonCell();
    this->cells.push_back(cell);
    old_info.push_back(CellInfo(1, 0, this->mortonObjects.size(), 0)); // root is at level 0

    while (old_info.size() > 0) {
        generateMortonCell(old_info, new_info);
        new_info.swap(old_info);
        new_info.clear();
    }
    //traverseObjects();
    LEAVE();
}

// generate the MortonCell based on the tree leve, and start, size of cells
void
MortonTree::generateMortonCell(std::vector<CellInfo> &old_info, std::vector<CellInfo> &new_info) {
    //ENTER();
    for (int i = 0; i < old_info.size(); i++) {
        int level = old_info[i].level;
        int start = old_info[i].start;
        int size = old_info[i].size;
        int parent = old_info[i].parent;
        int end = start + size;

        if (!isValidObjectsIndex(start, size)) {
            ERR("Invalid index, %d, %d\n", start, size);
            return;
        }
        
        unsigned int level_mask = get_level_mask(level);
        MortonCell *cell = new MortonCell();
        MortonTreeObject *o = this->mortonObjects[start];
        unsigned int old_mask = o->mcode & level_mask;
        cell->bound = rectangled_incr_bound(o->position);
        cell->com = Object::add(cell->com, *o);
        cell->level = level;
        o->parent = this->cells.size();  // the index of cell for this object
        int first_index = start;
        int count = 1;

        for (int j = start + 1; j < end; j++) {
            MortonTreeObject *o = this->mortonObjects[j];
            unsigned int new_mask = o->mcode & level_mask;

            if (new_mask == old_mask) {
                count++;
                cell->bound = rectangled_incr_bound(cell->bound, o->position);
                cell->com = Object::add(cell->com, *o);
                o->parent = this->cells.size();  // the index of cell for this object
                //printf("\e[1;34mobj: %d, parent: %d\e[0m\n", j, o->parent);
            } else {
                // if the group size of these objects is not greater than THRES_OBJS, then regard the cell as a leaf
                if (count <= THRES_OBJS) {
                    cell->is_leaf = true;
                } else {
                    cell->is_leaf = false;
                    // the index of the newest cell is the parent
                    new_info.push_back(CellInfo(level + 1, first_index, count, this->cells.size() + 1));
                }
                cell->first_index = first_index;
                cell->count = count;
                cell->parent = parent;
                this->cells.push_back(cell);
                // updat the children list for the new cell
                this->cells[parent]->children.push_back(this->cells.size() - 1);

                // create a new cell and update related info
                cell = new MortonCell();
                cell->bound = rectangled_incr_bound(o->position);
                cell->com = Object::add(cell->com, *o);
                o->parent = this->cells.size();  // the index of cell for this object
                first_index = j;
                count = 1;
            }
            cell->level = level;
            old_mask = new_mask;
        }

        // update info for the last new cell
        if (count <= THRES_OBJS) {
            cell->is_leaf = true; 
        } else {
            cell->is_leaf = false;
            new_info.push_back(CellInfo(level + 1, first_index, count, this->cells.size() + 1));
        }
        cell->first_index = first_index;
        cell->count = count;
        cell->parent = parent;
        this->cells.push_back(cell);
        this->cells[parent]->children.push_back(this->cells.size() - 1);
    }
    //LEAVE();
}

// calculate the total force exerted on an object specified with the index
Point2D 
MortonTree::getForceOnObject(int obj_idx) {
    ENTER();
    
    GS_FLOAT s, d;
    Point2D dr, result = point2d_zero();
    MortonTreeObject *tar_obj = this->mortonObjects[obj_idx];
    int i = 0;
    int end = this->mortonObjects.size();
    int access_idx = 0;

    while (i < end) {
        // every loop check a leaf cell
        MortonTreeObject *curr_obj = this->mortonObjects[i];
        int curr_cell_idx = curr_obj->parent;
        assert(curr_cell_idx >= 0);
        MortonCell *curr_cell = this->cells[curr_cell_idx];
        s = MAX(curr_cell->bound.size.x, curr_cell->bound.size.y);
        dr = point2d_sub(tar_obj->position, curr_cell->com.position);
        d = point2d_length(dr);

        // FIXME only try the leaf layer, maybe can go upstairs further
        if ((s/d) < SD_TRESHOLD) {
            //INFO("cell %d in the SD_TRESHOLD\n", access_idx);
            result = point2d_add(result, Object::calculate_force(*tar_obj, curr_cell->com));
        } else {
            //INFO("cell %d not in the SD_TRESHOLD\n", access_idx);
            for (int j = i; j < curr_cell->count; j++) {
                if (j == obj_idx) {
                    continue;
                }
                curr_obj = this->mortonObjects[j];
                result = point2d_add(result, Object::calculate_force(*tar_obj, *curr_obj));
            } 
        }
        i += curr_cell->count;
        access_idx++;
    }

    LEAVE();
    return result;
}

// update all the objects' speed
void 
MortonTree::applyToObjects(GS_FLOAT dt) {
    ENTER();
    MortonTreeObject *obj;

    for (int i = 0; i < this->mortonObjects.size(); i++) {
        Point2D acc = getForceOnObject(i);
        Point2D dv = point2d_multiply(acc, dt);
        obj = this->mortonObjects[i];
        //printf("obj:%d, pos.x:%f, pos.y:%f, acc.x:%f, acc.y:%f, dv.x:%f, dv.y:%f\n", i, obj->position.x, obj->position.y, acc.x, acc.y, dv.x, dv.y);
        obj->speed = point2d_add(obj->speed, dv);
        obj->update_position(dt);
    }
    LEAVE();
}

void
MortonTree::traverseCells() {
    ENTER();
    for (int i = 0; i < this->cells.size(); i++) {
        MortonCell *cell = this->cells[i];
        printf("cell:%d, is_leaf:%d, start:%d, size:%d, parent:%d\n", i, cell->is_leaf, cell->first_index, cell->count, cell->parent);
        printf("children: ");
        for (int j = 0; j < cell->children.size(); j++) {
            printf("%d, ", cell->children[j]);
        }
        printf("\n");
    }
    LEAVE();
}

void
MortonTree::traverseObjects() {
    ENTER();
    for (int i = 0; i < this->mortonObjects.size(); i++) {
        MortonTreeObject *o = this->mortonObjects[i];
        MortonCell *cell = this->cells[o->parent];
        unsigned int level_mask = get_level_mask(cell->level);
        printf("obj: %d, parent:%d, mcode:0x%u, level:%d, level_mask:0x%u, mask:0x%u\n", 
                i, o->parent, o->mcode, cell->level, level_mask, level_mask & o->mcode);
    }   
    LEAVE();
}

bool
MortonTree::isValidObjectsIndex(int start, int size) {
    int n = this->mortonObjects.size();
    //DBG("n: %d, start: %d, size: %d\n", n, start, size);
    return (n > 0) && (start >= 0) && (size >= 0) 
        && (start + size <= n); 
}

std::vector<MortonCell*>&
MortonTree::getCells() {
    return cells;
}

std::vector<MortonTreeObject*>& 
MortonTree::getObjects() {
    return mortonObjects;
}
