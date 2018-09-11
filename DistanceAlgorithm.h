#pragma once

#include <vector>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

#include <glm/glm.hpp>

class DistanceAlgorithm {
   public:
    virtual void load(const tinyobj::attrib_t& attrib, const std::vector<tinyobj::shape_t>& shapes) = 0;
    virtual std::vector<float> propagate(int src) = 0;
};