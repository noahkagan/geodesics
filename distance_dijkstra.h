#pragma once

#include <iostream>
#include <limits>
#include <queue>
#include <set>
#include <utility>

#include "DistanceAlgorithm.h"

class DijkstraAlgorithm : public DistanceAlgorithm {
   public:
    DijkstraAlgorithm() : adjacencies() {}

    void load(const tinyobj::attrib_t& attrib, const std::vector<tinyobj::shape_t>& shapes) override final {
        size_t numVerts = attrib.vertices.size() / 3;
        adjacencies.resize(numVerts);

        for (size_t s = 0; s < shapes.size(); ++s) {
            const tinyobj::shape_t& shape = shapes[s];
            size_t numFaces = shape.mesh.num_face_vertices.size();

            for (size_t f = 0; f < numFaces; f++) {
                int idx0 = shape.mesh.indices[3 * f + 0].vertex_index;
                int idx1 = shape.mesh.indices[3 * f + 1].vertex_index;
                int idx2 = shape.mesh.indices[3 * f + 2].vertex_index;

                glm::vec3 v[3];
                for (int c = 0; c < 3; c++) {
                    v[0][c] = attrib.vertices[3 * idx0 + c];
                    v[1][c] = attrib.vertices[3 * idx1 + c];
                    v[2][c] = attrib.vertices[3 * idx2 + c];
                }

                addEdge(idx0, idx1, glm::length(v[1] - v[0]));
                addEdge(idx0, idx2, glm::length(v[2] - v[0]));
                addEdge(idx1, idx2, glm::length(v[2] - v[1]));
            }
        }
    }

    std::vector<float> propagate(int src) override final {
        typedef std::pair<float, size_t> queue_entry_t;  // distance, index
        std::vector<float> dist(adjacencies.size(), std::numeric_limits<float>::max());
        dist[src] = 0;

        std::priority_queue<queue_entry_t, std::vector<queue_entry_t>, std::greater<queue_entry_t>> queue;
        queue.push(std::make_pair(dist[0], 0));

        while (!queue.empty()) {
            const queue_entry_t& entry = queue.top();
            float u_dist = entry.first;
            size_t u = entry.second;
            queue.pop();
            if (dist[u] != u_dist) continue;  // stale entry

            for (auto iter = adjacencies[u].begin(); iter != adjacencies[u].end(); ++iter) {
                size_t v = iter->first;
                float alt = u_dist + iter->second;

                if (alt < dist[v]) {
                    dist[v] = alt;
                    queue.push(std::make_pair(alt, v));
                }
            }
        }

        return dist;
    }

   private:
    void addEdge(size_t u, size_t v, float w) {
        adjacencies[u].push_back(std::make_pair(v, w));
        adjacencies[v].push_back(std::make_pair(u, w));
    }

    std::vector<std::vector<std::pair<size_t, float>>> adjacencies;
};
