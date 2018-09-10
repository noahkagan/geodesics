#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <glm/gtx/normal.hpp>

#include "geodesic_dijkstra.h"
#include "math.h"
#include "trackball.h"

typedef struct {
    GLuint vb_id;
    int numTriangles;
} DrawObject;

typedef struct {
    GLuint vb_id;
    int numPoints;
} DrawPoints;

float g_geodesic_radius_mod = 1.f;
float g_geodesic_radius = 10.f;
std::vector<DrawObject> g_draw_objects;
DrawPoints g_draw_points;

tinyobj::attrib_t g_attrib;
std::vector<float> g_geodesic_distance;

int width = 768;
int height = 768;

double prevMouseX, prevMouseY;
bool mouseLeftPressed = false;
bool mouseMiddlePressed = false;
bool mouseRightPressed = false;
bool leftShiftPressed = false;
float curr_quat[4];
float prev_quat[4];
float eye[3], lookat[3], up[3];

GLFWwindow* window;

static void check_gl_errors(const std::string& desc) {
    GLenum e = glGetError();
    if (e != GL_NO_ERROR) {
        fprintf(stderr, "OpenGL error in \"%s\": %d (%d)\n", desc.c_str(), e, e);
        exit(20);
    }
}

static bool convert(glm::vec3& bmin, glm::vec3& bmax, std::vector<DrawObject>* drawObjects,
                    const tinyobj::attrib_t& attrib, const std::vector<tinyobj::shape_t>& shapes) {
    bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<float>::max();
    bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<float>::max();

    for (size_t s = 0; s < shapes.size(); s++) {
        DrawObject o;
        std::vector<float> buffer;

        for (size_t f = 0; f < shapes[s].mesh.indices.size() / 3; f++) {
            tinyobj::index_t idx0 = shapes[s].mesh.indices[3 * f + 0];
            tinyobj::index_t idx1 = shapes[s].mesh.indices[3 * f + 1];
            tinyobj::index_t idx2 = shapes[s].mesh.indices[3 * f + 2];

            glm::vec3 v[3];
            for (int k = 0; k < 3; k++) {
                int f0 = idx0.vertex_index;
                int f1 = idx1.vertex_index;
                int f2 = idx2.vertex_index;
                assert(f0 >= 0);
                assert(f1 >= 0);
                assert(f2 >= 0);

                v[0][k] = attrib.vertices[3 * f0 + k];
                v[1][k] = attrib.vertices[3 * f1 + k];
                v[2][k] = attrib.vertices[3 * f2 + k];
                bmin[k] = std::min(v[0][k], bmin[k]);
                bmin[k] = std::min(v[1][k], bmin[k]);
                bmin[k] = std::min(v[2][k], bmin[k]);
                bmax[k] = std::max(v[0][k], bmax[k]);
                bmax[k] = std::max(v[1][k], bmax[k]);
                bmax[k] = std::max(v[2][k], bmax[k]);
            }

            glm::vec3 n[3];
            n[0] = n[1] = n[2] = glm::triangleNormal(v[0], v[1], v[2]);

            for (int k = 0; k < 3; k++) {
                buffer.push_back(v[k][0]);
                buffer.push_back(v[k][1]);
                buffer.push_back(v[k][2]);
                buffer.push_back(n[k][0]);
                buffer.push_back(n[k][1]);
                buffer.push_back(n[k][2]);
                glm::vec3 diffuse{0.4f, 0.4f, 0.4f};
                glm::vec3 light_pos{-50.f, -50.f, -25.f};
                glm::vec3 norm = glm::normalize(n[k]);
                glm::vec3 light_dir = glm::normalize(light_pos - v[k]);
                float diff = std::max(0.f, glm::dot(norm, light_dir));
                glm::vec3 color{diffuse[0] * diff, diffuse[1] * diff, diffuse[2] * diff};
                buffer.push_back(color[0]);
                buffer.push_back(color[1]);
                buffer.push_back(color[2]);
            }
        }

        o.vb_id = 0;
        o.numTriangles = 0;

        if (buffer.size() > 0) {
            glGenBuffers(1, &o.vb_id);
            glBindBuffer(GL_ARRAY_BUFFER, o.vb_id);
            glBufferData(GL_ARRAY_BUFFER, buffer.size() * sizeof(float), &buffer.at(0), GL_STATIC_DRAW);
            o.numTriangles = buffer.size() / (3 + 3 + 3) / 3;
        }

        drawObjects->push_back(o);
    }

    return true;
}

void update_draw_points(const tinyobj::attrib_t& attrib) {
    std::vector<float> buffer;
    for (size_t i = 0; i < g_geodesic_distance.size(); ++i) {
        if (g_geodesic_distance[i] <= g_geodesic_radius) {
            buffer.push_back(attrib.vertices[3 * i + 0]);
            buffer.push_back(attrib.vertices[3 * i + 1]);
            buffer.push_back(attrib.vertices[3 * i + 2]);
        }
    }
    if (!buffer.empty()) {
        glGenBuffers(1, &g_draw_points.vb_id);
        glBindBuffer(GL_ARRAY_BUFFER, g_draw_points.vb_id);
        glBufferData(GL_ARRAY_BUFFER, buffer.size() * sizeof(float), &buffer.at(0), GL_STATIC_DRAW);
    }
    g_draw_points.numPoints = buffer.size() / 3;
}

static void window_size_callback(GLFWwindow* window, int w, int h) {
    int fb_w, fb_h;
    glfwGetFramebufferSize(window, &fb_w, &fb_h);
    glViewport(0, 0, fb_w, fb_h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (float)w / (float)h, 0.01f, 100.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    width = w;
    height = h;
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)window;
    (void)scancode;
    (void)mods;
    if (key == GLFW_KEY_LEFT_SHIFT) {
        if (action == GLFW_PRESS) {
            leftShiftPressed = true;
        } else if (action == GLFW_RELEASE) {
            leftShiftPressed = false;
        }
    }
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    (void)window;
    (void)mods;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mouseLeftPressed = true;
            trackball(prev_quat, 0.0, 0.0, 0.0, 0.0);
        } else if (action == GLFW_RELEASE) {
            mouseLeftPressed = false;
        }
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            mouseRightPressed = true;
        } else if (action == GLFW_RELEASE) {
            mouseRightPressed = false;
        }
    }
    if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        if (action == GLFW_PRESS) {
            mouseMiddlePressed = true;
        } else if (action == GLFW_RELEASE) {
            mouseMiddlePressed = false;
        }
    }
}

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (leftShiftPressed) {
        g_geodesic_radius_mod *= yoffset > 0 ? 2.f : 0.5f;
        printf("radius mod: %f\n", g_geodesic_radius_mod);
    } else {
        g_geodesic_radius += yoffset*g_geodesic_radius_mod;
        printf("radius: %f\n", g_geodesic_radius);
        update_draw_points(g_attrib);
    }
}

static void cursor_pos_callback(GLFWwindow* window, double mouse_x, double mouse_y) {
    (void)window;
    float rotScale = 1.0f;
    float transScale = 2.0f;

    if (mouseLeftPressed) {
        trackball(prev_quat, rotScale * (2.0f * prevMouseX - width) / (float)width,
                  rotScale * (height - 2.0f * prevMouseY) / (float)height,
                  rotScale * (2.0f * mouse_x - width) / (float)width,
                  rotScale * (height - 2.0f * mouse_y) / (float)height);

        add_quats(prev_quat, curr_quat, curr_quat);
    } else if (mouseMiddlePressed) {
        eye[0] -= transScale * (mouse_x - prevMouseX) / (float)width;
        lookat[0] -= transScale * (mouse_x - prevMouseX) / (float)width;
        eye[1] += transScale * (mouse_y - prevMouseY) / (float)height;
        lookat[1] += transScale * (mouse_y - prevMouseY) / (float)height;
    } else if (mouseRightPressed) {
        eye[2] += transScale * (mouse_y - prevMouseY) / (float)height;
        lookat[2] += transScale * (mouse_y - prevMouseY) / (float)height;
    }

    prevMouseX = mouse_x;
    prevMouseY = mouse_y;
}

static void draw(const std::vector<DrawObject>& drawObjects) {
    GLsizei stride = (3 + 3 + 3) * sizeof(float);

    // draw mesh
    glPolygonMode(GL_FRONT, GL_FILL);
    glPolygonMode(GL_BACK, GL_FILL);

    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0, 100.0);
    for (size_t i = 0; i < drawObjects.size(); i++) {
        DrawObject o = drawObjects[i];
        if (o.vb_id < 1) { continue; }

        glBindBuffer(GL_ARRAY_BUFFER, o.vb_id);
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);

        glVertexPointer(3, GL_FLOAT, stride, (const void*)0);
        glNormalPointer(GL_FLOAT, stride, (const void*)(sizeof(float) * 3));
        glColorPointer(3, GL_FLOAT, stride, (const void*)(sizeof(float) * 6));

        glDrawArrays(GL_TRIANGLES, 0, 3 * o.numTriangles);
        check_gl_errors("drawarrays");
    }

    // draw wireframe
    glDisable(GL_POLYGON_OFFSET_FILL);
    glEnable(GL_POLYGON_OFFSET_LINE);
    glPolygonMode(GL_FRONT, GL_LINE);
    glPolygonMode(GL_BACK, GL_LINE);

    glPolygonOffset(1.0, 10.0);
    glColor3f(0.3f, 0.3f, 0.4f);
    for (size_t i = 0; i < drawObjects.size(); i++) {
        DrawObject o = drawObjects[i];
        if (o.vb_id < 1) { continue; }

        glBindBuffer(GL_ARRAY_BUFFER, o.vb_id);
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
        glVertexPointer(3, GL_FLOAT, stride, (const void*)0);
        glNormalPointer(GL_FLOAT, stride, (const void*)(sizeof(float) * 3));
        glColorPointer(3, GL_FLOAT, stride, (const void*)(sizeof(float) * 6));

        glDrawArrays(GL_TRIANGLES, 0, 3 * o.numTriangles);
        check_gl_errors("drawarrays");
    }

    // draw vertices
    glDisable(GL_POLYGON_OFFSET_LINE);
    glEnable(GL_POLYGON_OFFSET_POINT);
    glPolygonMode(GL_FRONT, GL_POINT);
    glPolygonMode(GL_BACK, GL_POINT);

    glPolygonOffset(1.0, 1.0);
    glColor3f(0.6f, 0.6f, 0.6f);
    glPointSize(1.f);
    for (size_t i = 0; i < drawObjects.size(); i++) {
        DrawObject o = drawObjects[i];
        if (o.vb_id < 1) { continue; }

        glBindBuffer(GL_ARRAY_BUFFER, o.vb_id);
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
        glVertexPointer(3, GL_FLOAT, stride, (const void*)0);
        glNormalPointer(GL_FLOAT, stride, (const void*)(sizeof(float) * 3));
        glColorPointer(3, GL_FLOAT, stride, (const void*)(sizeof(float) * 6));

        glDrawArrays(GL_TRIANGLES, 0, 3 * o.numTriangles);
        check_gl_errors("drawarrays");
    }
}

static void draw(const DrawPoints& drawPoints, float color[3]) {
    GLsizei stride = (3 + 0) * sizeof(float);

    // draw vertices
    glEnable(GL_POLYGON_OFFSET_POINT);
    glPolygonMode(GL_FRONT, GL_POINT);
    glPolygonMode(GL_BACK, GL_POINT);
    glPolygonOffset(1.0, -10.0);
    glColor3f(color[0], color[1], color[2]);
    glPointSize(5.f);
    if (drawPoints.vb_id >= 1) {
        glBindBuffer(GL_ARRAY_BUFFER, drawPoints.vb_id);
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(3, GL_FLOAT, stride, (const void*)0);

        glDrawArrays(GL_POINTS, 0, drawPoints.numPoints);
        check_gl_errors("drawarrays");
    }
}

static void init() {
    trackball(curr_quat, 0, 0, 0, 0);

    eye[0] = 0.0f;
    eye[1] = 0.0f;
    eye[2] = 3.0f;

    lookat[0] = 0.0f;
    lookat[1] = 0.0f;
    lookat[2] = 0.0f;

    up[0] = 0.0f;
    up[1] = 1.0f;
    up[2] = 0.0f;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Needs input.obj\n" << std::endl;
        return 0;
    }

    init();

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW." << std::endl;
        return -1;
    }

    window = glfwCreateWindow(width, height, "Obj viewer", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to open GLFW window. " << std::endl;
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glfwSetWindowSizeCallback(window, window_size_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glewExperimental = true;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW." << std::endl;
        return -1;
    }

    window_size_callback(window, width, height);

    std::vector<tinyobj::shape_t> shapes;

    std::string err;
    if (!tinyobj::LoadObj(&g_attrib, &shapes, nullptr, &err, argv[1])) {
        if (!err.empty()) { std::cerr << err << std::endl; }
        glfwTerminate();
        return -1;
    }

    size_t src_vertex_id = 0;
    DrawPoints source;
    {
        std::vector<float> buffer;
        buffer.push_back(g_attrib.vertices[3 * src_vertex_id + 0]);
        buffer.push_back(g_attrib.vertices[3 * src_vertex_id + 1]);
        buffer.push_back(g_attrib.vertices[3 * src_vertex_id + 2]);
        glGenBuffers(1, &source.vb_id);
        glBindBuffer(GL_ARRAY_BUFFER, source.vb_id);
        glBufferData(GL_ARRAY_BUFFER, buffer.size() * sizeof(float), &buffer.at(0), GL_STATIC_DRAW);
        source.numPoints = buffer.size() / 3;
    }

    glm::vec3 bmin, bmax;
    if (!convert(bmin, bmax, &g_draw_objects, g_attrib, shapes)) {
        glfwTerminate();
        return -1;
    }

    Geodesic g;
    g.load(g_attrib, shapes);
    g_geodesic_distance = g.propagate(src_vertex_id);

    update_draw_points(g_attrib);

    float maxExtent = 0.5f * (bmax[0] - bmin[0]);
    if (maxExtent < 0.5f * (bmax[1] - bmin[1])) { maxExtent = 0.5f * (bmax[1] - bmin[1]); }
    if (maxExtent < 0.5f * (bmax[2] - bmin[2])) { maxExtent = 0.5f * (bmax[2] - bmin[2]); }

    while (glfwWindowShouldClose(window) == GL_FALSE) {
        glfwPollEvents();
        glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);

        // camera & rotate
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        GLfloat mat[4][4];
        gluLookAt(eye[0], eye[1], eye[2], lookat[0], lookat[1], lookat[2], up[0], up[1], up[2]);
        build_rotmatrix(mat, curr_quat);
        glMultMatrixf(&mat[0][0]);

        // Fit to -1, 1
        glScalef(1.0f / maxExtent, 1.0f / maxExtent, 1.0f / maxExtent);

        // center object
        glTranslatef(-0.5 * (bmax[0] + bmin[0]), -0.5 * (bmax[1] + bmin[1]), -0.5 * (bmax[2] + bmin[2]));

        draw(g_draw_objects);

        float green[3] = {0.f, 1.f, 0.f};
        float red[3] = {1.f, 0.f, 0.f};
        draw(source, green);
        if (g_draw_points.numPoints) { draw(g_draw_points, red); }

        glfwSwapBuffers(window);
    }

    glfwTerminate();
}
