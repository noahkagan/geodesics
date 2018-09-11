#pragma once

#include <iostream>
#include <limits>
#include <queue>
#include <set>
#include <utility>

class WorldSpaceAlgorithm : public DistanceAlgorithm {
   public:
    WorldSpaceAlgorithm() : vertices() {}

    void load(const tinyobj::attrib_t& attrib, const std::vector<tinyobj::shape_t>& shapes) override final {
        size_t numVerts = attrib.vertices.size() / 3;
        vertices.reserve(numVerts);

        for (size_t i = 0; i < numVerts; ++i) {
            glm::vec3 v;
            for (int c = 0; c < 3; c++) {
                v[0] = attrib.vertices[3 * i + 0];
                v[1] = attrib.vertices[3 * i + 1];
                v[2] = attrib.vertices[3 * i + 2];
            }
            vertices.push_back(v);
        }
    }

    std::vector<float> propagate(int src) override final {
        std::vector<float> dist(vertices.size(), std::numeric_limits<float>::max());
        const glm::vec3& v = vertices[src];
        for (size_t i = 0; i < vertices.size(); ++i) { dist[i] = glm::length(v - vertices[i]); }
        return dist;
    }

   private:
    std::vector<glm::vec3> vertices;
};
