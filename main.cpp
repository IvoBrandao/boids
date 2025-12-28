// main.cpp
#include <GL/glew.h>
#include <GL/glut.h>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

struct Vector3 {
    float x, y, z;
    Vector3() : x(0.0f), y(0.0f), z(0.0f) {}
    Vector3(float a, float b, float c) : x(a), y(b), z(c) {}
    Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }
    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }
    Vector3 operator*(float scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }
    Vector3 operator/(float scalar) const {
        if (scalar == 0.0f) return *this;
        return Vector3(x / scalar, y / scalar, z / scalar);
    }
    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }
    Vector3 normalized() const {
        float len = length();
        if (len == 0.0f) return *this;
        return *this / len;
    }
    float lengthSquared() const {
        return x * x + y * y + z * z;
    }
    // Normalize in-place safe; returns zero vector if length is zero
    Vector3 normalized_safe() const {
        float ls = lengthSquared();
        if (ls == 0.0f) return *this;
        float inv = 1.0f / std::sqrt(ls);
        return Vector3(x * inv, y * inv, z * inv);
    }
};

class Boid {
public:
    Vector3 position;
    Vector3 velocity;
    Boid() {}
    Boid(const Vector3& pos, const Vector3& vel) : position(pos), velocity(vel) {}
};

// Use contiguous storage for cache friendliness
std::vector<Boid> flock;
// CPU-side float buffer for uploading points to GL
std::vector<float> pointBuffer;
// GPU VBO handle
static GLuint boidVBO = 0;
// Color VBO handle
static GLuint colorVBO = 0;
// GPU SSBOs for compute shader
static GLuint posSSBO = 0;
static GLuint velSSBO = 0;
// Spatial hashing structures on GPU
static GLuint headSSBO = 0; // int per cell, head of linked list (-1 == empty)
static GLuint nextSSBO = 0; // int per boid, next index in linked list (-1 == end)
// Compute program
static GLuint computeProgram = 0;
// Cached uniform locations for compute shader
static GLint uni_numBoids = -1;
static GLint uni_dt = -1;
static GLint uni_maxSpeed = -1;
static GLint uni_cohesionWeight = -1;
static GLint uni_bounds = -1;
// Additional uniforms for spatial hashing compute shader
static GLint uni_mode = -1; // 0=clear heads,1=build lists,2=simulate
static GLint uni_numCells = -1;
static GLint uni_gridNx = -1;
static GLint uni_gridNy = -1;
static GLint uni_gridNz = -1;
static GLint uni_cellSize = -1;
static GLint uni_separationDist = -1;
static GLint uni_alignmentDist = -1;
static GLint uni_cohesionDist = -1;
static GLint uni_sepWeight = -1;
static GLint uni_aliWeight = -1;
static GLint uni_cohWeight = -1;
// CPU-side velocity buffer for initialization (vec4 per boid)
std::vector<float> velBuffer;

// Spatial grid for neighbor queries (initialized in initializeFlock)
float CELL_SIZE = 1.0f;
int GRID_NX = 0, GRID_NY = 0, GRID_NZ = 0;
std::vector<std::vector<int>> grid; // cell -> list of boid indices

const int NUM_BOIDS = 4 * 1024*1024; // Configurable number of boids (16 million)
// Target density (boids per unit^3) used to size the cube automatically
const float TARGET_DENSITY = 0.4f;
// CUBE_SIZE and BOUNDS are runtime-adjusted in initializeFlock()
float CUBE_SIZE = 800.0f;
float BOUNDS = CUBE_SIZE / 2.0f;
// Increase max speed so boids move noticeably faster across larger cubes
float MAX_SPEED = 2.8f;
const float SEPARATION_DISTANCE = 0.5f;
const float ALIGNMENT_DISTANCE = 1.0f;
const float COHESION_DISTANCE = 1.0f;
const float SEPARATION_WEIGHT = 0.5f;
const float ALIGNMENT_WEIGHT = 0.3f;
const float COHESION_WEIGHT = 0.2f;

// Start in an isometric-ish view
float yaw = 0.78539816339f; // ~45 degrees
float pitch = 0.61547970867f; // ~35.264 degrees (isometric angle)
// Increase default camera distance to accommodate larger cube
float radius = CUBE_SIZE * 2.5f;
int lastX = 0;
int lastY = 0;
bool mouseDown = false;
int windowWidth = 1280;
int windowHeight = 900;
// Zoom parameters (mouse wheel)
const float ZOOM_STEP = 10.0f;
const float MIN_RADIUS = 5.0f;
const float MAX_RADIUS = 6000.0f;

void initializeFlock() {
    // Auto-adjust cube size to keep visual density approximately constant
    {
        double desired = std::cbrt(static_cast<double>(NUM_BOIDS) / static_cast<double>(TARGET_DENSITY));
        if (desired < 20.0) desired = 20.0;
        CUBE_SIZE = static_cast<float>(desired);
        BOUNDS = CUBE_SIZE * 0.5f;
        // Adjust camera to fit the new cube
        radius = CUBE_SIZE * 2.5f;
    }
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    flock.clear();
    flock.reserve(NUM_BOIDS);
    for (int i = 0; i < NUM_BOIDS; ++i) {
        float px = (static_cast<float>(std::rand()) / RAND_MAX * CUBE_SIZE) - BOUNDS;
        float py = (static_cast<float>(std::rand()) / RAND_MAX * CUBE_SIZE) - BOUNDS;
        float pz = (static_cast<float>(std::rand()) / RAND_MAX * CUBE_SIZE) - BOUNDS;
        float vx = (static_cast<float>(std::rand()) / RAND_MAX * 0.02f) - 0.01f;
        float vy = (static_cast<float>(std::rand()) / RAND_MAX * 0.02f) - 0.01f;
        float vz = (static_cast<float>(std::rand()) / RAND_MAX * 0.02f) - 0.01f;
        flock.emplace_back(Vector3(px, py, pz), Vector3(vx, vy, vz));
    }

    // Prepare point buffer
    pointBuffer.resize(static_cast<size_t>(NUM_BOIDS) * 3);

    // Setup spatial grid
    CELL_SIZE = std::max(std::max(SEPARATION_DISTANCE, ALIGNMENT_DISTANCE), COHESION_DISTANCE);
    GRID_NX = static_cast<int>(std::ceil(CUBE_SIZE / CELL_SIZE));
    GRID_NY = GRID_NX;
    GRID_NZ = GRID_NX;
    int totalCells = GRID_NX * GRID_NY * GRID_NZ;
    grid.clear();
    grid.resize(totalCells);

    // Prepare CPU velocity buffer and SSBOs
    velBuffer.resize(static_cast<size_t>(NUM_BOIDS) * 4);
    std::vector<float> posInit(static_cast<size_t>(NUM_BOIDS) * 4);
    for (size_t i = 0; i < flock.size(); ++i) {
        posInit[i * 4 + 0] = flock[i].position.x;
        posInit[i * 4 + 1] = flock[i].position.y;
        posInit[i * 4 + 2] = flock[i].position.z;
        posInit[i * 4 + 3] = 1.0f;
        velBuffer[i * 4 + 0] = flock[i].velocity.x;
        velBuffer[i * 4 + 1] = flock[i].velocity.y;
        velBuffer[i * 4 + 2] = flock[i].velocity.z;
        velBuffer[i * 4 + 3] = 0.0f;
    }

    // Create GPU VBO for boid positions (vec4 per boid)
    glGenBuffers(1, &boidVBO);
    glBindBuffer(GL_ARRAY_BUFFER, boidVBO);
    glBufferData(GL_ARRAY_BUFFER, posInit.size() * sizeof(float), posInit.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Create color VBO and assign groups of colors
    std::vector<float> colorInit(static_cast<size_t>(NUM_BOIDS) * 4);
    const int numGroups = 8;
    for (size_t i = 0; i < static_cast<size_t>(NUM_BOIDS); ++i) {
        int grp = static_cast<int>((i * numGroups) / static_cast<size_t>(NUM_BOIDS));
        float r = 0.0f, g = 0.0f, b = 0.0f;
        switch (grp) {
            case 0: r=1; g=0; b=0; break; // red
            case 1: r=0; g=1; b=0; break; // green
            case 2: r=0; g=0; b=1; break; // blue
            case 3: r=1; g=1; b=0; break; // yellow
            case 4: r=1; g=0; b=1; break; // magenta
            case 5: r=0; g=1; b=1; break; // cyan
            case 6: r=1; g=0.5f; b=0; break; // orange
            default: r=0.7f; g=0.7f; b=0.7f; break; // grey
        }
        colorInit[i*4+0] = r;
        colorInit[i*4+1] = g;
        colorInit[i*4+2] = b;
        colorInit[i*4+3] = 1.0f;
    }
    glGenBuffers(1, &colorVBO);
    glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
    glBufferData(GL_ARRAY_BUFFER, colorInit.size() * sizeof(float), colorInit.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &posSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, posSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, posInit.size() * sizeof(float), posInit.data(), GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posSSBO);

    glGenBuffers(1, &velSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, velSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, velBuffer.size() * sizeof(float), velBuffer.data(), GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, velSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // Create head and next buffers for GPU spatial hashing (linked-list per cell)
    std::vector<int> headInit(static_cast<size_t>(totalCells), -1);
    std::vector<int> nextInit(static_cast<size_t>(NUM_BOIDS), -1);
    glGenBuffers(1, &headSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, headSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, headInit.size() * sizeof(int), headInit.data(), GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, headSSBO);

    glGenBuffers(1, &nextSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, nextSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, nextInit.size() * sizeof(int), nextInit.data(), GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, nextSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void update() {
    // If a GPU compute program exists, skip the CPU-side heavy update entirely.
    // The GPU compute shader will be dispatched in display() and will update
    // positions/velocities on the GPU. This prevents CPU going to 100% for large N.
    if (computeProgram != 0) {
        return;
    }
    // Build spatial grid
    for (auto &cell : grid) cell.clear();
    auto cellIndex = [&](const Vector3 &p) {
        int ix = static_cast<int>((p.x + BOUNDS) / CELL_SIZE);
        int iy = static_cast<int>((p.y + BOUNDS) / CELL_SIZE);
        int iz = static_cast<int>((p.z + BOUNDS) / CELL_SIZE);
        if (ix < 0) ix = 0; if (ix >= GRID_NX) ix = GRID_NX - 1;
        if (iy < 0) iy = 0; if (iy >= GRID_NY) iy = GRID_NY - 1;
        if (iz < 0) iz = 0; if (iz >= GRID_NZ) iz = GRID_NZ - 1;
        return ix + iy * GRID_NX + iz * GRID_NX * GRID_NY;
    };

    for (size_t i = 0; i < flock.size(); ++i) {
        int idx = cellIndex(flock[i].position);
        grid[idx].push_back(static_cast<int>(i));
    }

    // Compute accelerations in parallel (reads grid only)
    std::vector<Vector3> accelerations(flock.size(), Vector3());
    const float sepDist2 = SEPARATION_DISTANCE * SEPARATION_DISTANCE;
    const float aliDist2 = ALIGNMENT_DISTANCE * ALIGNMENT_DISTANCE;
    const float cohDist2 = COHESION_DISTANCE * COHESION_DISTANCE;

    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < static_cast<int>(flock.size()); ++ii) {
        size_t i = static_cast<size_t>(ii);
        // Simplified neighborhood: only check boids in the same cell (faster)
        const Vector3 &pos = flock[i].position;
        int ix = static_cast<int>((pos.x + BOUNDS) / CELL_SIZE);
        int iy = static_cast<int>((pos.y + BOUNDS) / CELL_SIZE);
        int iz = static_cast<int>((pos.z + BOUNDS) / CELL_SIZE);
        if (ix < 0) ix = 0; if (ix >= GRID_NX) ix = GRID_NX - 1;
        if (iy < 0) iy = 0; if (iy >= GRID_NY) iy = GRID_NY - 1;
        if (iz < 0) iz = 0; if (iz >= GRID_NZ) iz = GRID_NZ - 1;
        int cellId = ix + iy * GRID_NX + iz * GRID_NX * GRID_NY;
        Vector3 separation(0.0f, 0.0f, 0.0f);
        Vector3 alignment(0.0f, 0.0f, 0.0f);
        Vector3 cohesion(0.0f, 0.0f, 0.0f);
        int separationCount = 0, alignmentCount = 0, cohesionCount = 0;

        for (int j : grid[cellId]) {
            if (static_cast<int>(i) == j) continue;
            const Vector3 &otherPos = flock[j].position;
            Vector3 diff = pos - otherPos;
            float dist2 = diff.lengthSquared();
            if (dist2 <= 0.0f) continue;
            if (dist2 < sepDist2) {
                // approximate normalized vector by scaling with reciprocal sqrt
                float inv = 1.0f / std::sqrt(dist2);
                separation = separation + (diff * inv);
                ++separationCount;
            }
            if (dist2 < aliDist2) {
                alignment = alignment + flock[j].velocity;
                ++alignmentCount;
            }
            if (dist2 < cohDist2) {
                cohesion = cohesion + otherPos;
                ++cohesionCount;
            }
        }

        Vector3 sepSteer(0.0f,0.0f,0.0f);
        if (separationCount > 0) {
            separation = separation / static_cast<float>(separationCount);
            sepSteer = (separation.normalized_safe() * MAX_SPEED - flock[i].velocity) * SEPARATION_WEIGHT;
        }

        Vector3 aliSteer(0.0f,0.0f,0.0f);
        if (alignmentCount > 0) {
            alignment = alignment / static_cast<float>(alignmentCount);
            aliSteer = (alignment.normalized_safe() * MAX_SPEED - flock[i].velocity) * ALIGNMENT_WEIGHT;
        }

        Vector3 cohSteer(0.0f,0.0f,0.0f);
        if (cohesionCount > 0) {
            cohesion = cohesion / static_cast<float>(cohesionCount);
            Vector3 toCenter = cohesion - flock[i].position;
            cohSteer = (toCenter.normalized_safe() * MAX_SPEED - flock[i].velocity) * COHESION_WEIGHT;
        }

        accelerations[i] = sepSteer + aliSteer + cohSteer;
    }

    // Apply accelerations and integrate in parallel, including bounds
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < static_cast<int>(flock.size()); ++ii) {
        size_t i = static_cast<size_t>(ii);
        flock[i].velocity = flock[i].velocity + accelerations[i];
        float speed2 = flock[i].velocity.lengthSquared();
        if (speed2 > MAX_SPEED * MAX_SPEED) {
            flock[i].velocity = flock[i].velocity.normalized_safe() * MAX_SPEED;
        }
        flock[i].position = flock[i].position + flock[i].velocity;

        // Bound within cube by reflecting
        if (flock[i].position.x > BOUNDS) {
            flock[i].position.x = BOUNDS;
            flock[i].velocity.x = -flock[i].velocity.x;
        } else if (flock[i].position.x < -BOUNDS) {
            flock[i].position.x = -BOUNDS;
            flock[i].velocity.x = -flock[i].velocity.x;
        }
        if (flock[i].position.y > BOUNDS) {
            flock[i].position.y = BOUNDS;
            flock[i].velocity.y = -flock[i].velocity.y;
        } else if (flock[i].position.y < -BOUNDS) {
            flock[i].position.y = -BOUNDS;
            flock[i].velocity.y = -flock[i].velocity.y;
        }
        if (flock[i].position.z > BOUNDS) {
            flock[i].position.z = BOUNDS;
            flock[i].velocity.z = -flock[i].velocity.z;
        } else if (flock[i].position.z < -BOUNDS) {
            flock[i].position.z = -BOUNDS;
            flock[i].velocity.z = -flock[i].velocity.z;
        }
    }

    // Update point buffer for rendering
    for (size_t i = 0; i < flock.size(); ++i) {
        pointBuffer[i * 3 + 0] = flock[i].position.x;
        pointBuffer[i * 3 + 1] = flock[i].position.y;
        pointBuffer[i * 3 + 2] = flock[i].position.z;
    }
}

void drawCube() {
    // Draw semi-transparent faces to help depth perception
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0f, 1.0f);
    // Draw translucent faces without writing depth so interior boids are visible
    glColor4f(0.15f, 0.15f, 0.15f, 0.25f);
    glDepthMask(GL_FALSE);
    glBegin(GL_QUADS);
    // Front (-Z)
    glVertex3f(-BOUNDS, -BOUNDS, -BOUNDS);
    glVertex3f(BOUNDS, -BOUNDS, -BOUNDS);
    glVertex3f(BOUNDS, BOUNDS, -BOUNDS);
    glVertex3f(-BOUNDS, BOUNDS, -BOUNDS);
    // Back (+Z)
    glVertex3f(-BOUNDS, -BOUNDS, BOUNDS);
    glVertex3f(BOUNDS, -BOUNDS, BOUNDS);
    glVertex3f(BOUNDS, BOUNDS, BOUNDS);
    glVertex3f(-BOUNDS, BOUNDS, BOUNDS);
    // Left (-X)
    glVertex3f(-BOUNDS, -BOUNDS, -BOUNDS);
    glVertex3f(-BOUNDS, BOUNDS, -BOUNDS);
    glVertex3f(-BOUNDS, BOUNDS, BOUNDS);
    glVertex3f(-BOUNDS, -BOUNDS, BOUNDS);
    // Right (+X)
    glVertex3f(BOUNDS, -BOUNDS, -BOUNDS);
    glVertex3f(BOUNDS, BOUNDS, -BOUNDS);
    glVertex3f(BOUNDS, BOUNDS, BOUNDS);
    glVertex3f(BOUNDS, -BOUNDS, BOUNDS);
    // Top (+Y)
    glVertex3f(-BOUNDS, BOUNDS, -BOUNDS);
    glVertex3f(BOUNDS, BOUNDS, -BOUNDS);
    glVertex3f(BOUNDS, BOUNDS, BOUNDS);
    glVertex3f(-BOUNDS, BOUNDS, BOUNDS);
    // Bottom (-Y)
    glVertex3f(-BOUNDS, -BOUNDS, -BOUNDS);
    glVertex3f(BOUNDS, -BOUNDS, -BOUNDS);
    glVertex3f(BOUNDS, -BOUNDS, BOUNDS);
    glVertex3f(-BOUNDS, -BOUNDS, BOUNDS);
    glEnd();
    glEnd();
    glDepthMask(GL_TRUE);
    glDisable(GL_POLYGON_OFFSET_FILL);

    // Draw bold edges on top
    glColor3f(1.0f, 1.0f, 1.0f);
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    // Front face
    glVertex3f(-BOUNDS, -BOUNDS, -BOUNDS);
    glVertex3f(BOUNDS, -BOUNDS, -BOUNDS);
    glVertex3f(BOUNDS, -BOUNDS, -BOUNDS);
    glVertex3f(BOUNDS, BOUNDS, -BOUNDS);
    glVertex3f(BOUNDS, BOUNDS, -BOUNDS);
    glVertex3f(-BOUNDS, BOUNDS, -BOUNDS);
    glVertex3f(-BOUNDS, BOUNDS, -BOUNDS);
    glVertex3f(-BOUNDS, -BOUNDS, -BOUNDS);
    // Back face
    glVertex3f(-BOUNDS, -BOUNDS, BOUNDS);
    glVertex3f(BOUNDS, -BOUNDS, BOUNDS);
    glVertex3f(BOUNDS, -BOUNDS, BOUNDS);
    glVertex3f(BOUNDS, BOUNDS, BOUNDS);
    glVertex3f(BOUNDS, BOUNDS, BOUNDS);
    glVertex3f(-BOUNDS, BOUNDS, BOUNDS);
    glVertex3f(-BOUNDS, BOUNDS, BOUNDS);
    glVertex3f(-BOUNDS, -BOUNDS, BOUNDS);
    // Connecting edges
    glVertex3f(-BOUNDS, -BOUNDS, -BOUNDS);
    glVertex3f(-BOUNDS, -BOUNDS, BOUNDS);
    glVertex3f(BOUNDS, -BOUNDS, -BOUNDS);
    glVertex3f(BOUNDS, -BOUNDS, BOUNDS);
    glVertex3f(BOUNDS, BOUNDS, -BOUNDS);
    glVertex3f(BOUNDS, BOUNDS, BOUNDS);
    glVertex3f(-BOUNDS, BOUNDS, -BOUNDS);
    glVertex3f(-BOUNDS, BOUNDS, BOUNDS);
    glEnd();
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // Set far plane large enough for big cube; base on radius for safety
    double farPlane = std::max(100.0, static_cast<double>(radius * 2.0));
    gluPerspective(45.0, static_cast<double>(windowWidth) / windowHeight, 0.1, farPlane);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    float camX = radius * std::sin(yaw) * std::cos(pitch);
    float camY = radius * std::sin(pitch);
    float camZ = radius * std::cos(yaw) * std::cos(pitch);
    gluLookAt(camX, camY, camZ, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    drawCube();

    // If compute shader + SSBO are available, dispatch compute to update positions on GPU
    if (computeProgram != 0) {
        glUseProgram(computeProgram);
        // bind SSBOs (already bound in init, but ensure binding)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, velSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, headSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, nextSSBO);
        // set uniforms
        // Use cached uniform locations where available
        if (uni_numBoids >= 0) glUniform1i(uni_numBoids, NUM_BOIDS);
        if (uni_dt >= 0) glUniform1f(uni_dt, 0.016f);
        if (uni_maxSpeed >= 0) glUniform1f(uni_maxSpeed, MAX_SPEED);
        if (uni_cohWeight >= 0) glUniform1f(uni_cohWeight, COHESION_WEIGHT);
        if (uni_bounds >= 0) glUniform1f(uni_bounds, BOUNDS);
        int totalCells = GRID_NX * GRID_NY * GRID_NZ;
        if (uni_numCells >= 0) glUniform1i(uni_numCells, totalCells);
        if (uni_gridNx >= 0) glUniform1i(uni_gridNx, GRID_NX);
        if (uni_gridNy >= 0) glUniform1i(uni_gridNy, GRID_NY);
        if (uni_gridNz >= 0) glUniform1i(uni_gridNz, GRID_NZ);
        if (uni_cellSize >= 0) glUniform1f(uni_cellSize, CELL_SIZE);
        if (uni_separationDist >= 0) glUniform1f(uni_separationDist, SEPARATION_DISTANCE);
        if (uni_alignmentDist >= 0) glUniform1f(uni_alignmentDist, ALIGNMENT_DISTANCE);
        if (uni_cohesionDist >= 0) glUniform1f(uni_cohesionDist, COHESION_DISTANCE);
        if (uni_sepWeight >= 0) glUniform1f(uni_sepWeight, SEPARATION_WEIGHT);
        if (uni_aliWeight >= 0) glUniform1f(uni_aliWeight, ALIGNMENT_WEIGHT);
        if (uni_cohWeight >= 0) glUniform1f(uni_cohWeight, COHESION_WEIGHT);

        // Multi-pass dispatch: 0=clear heads, 1=build lists, 2=simulate
        const int localSize = 256;
        // clear heads
        if (uni_mode >= 0) glUniform1i(uni_mode, 0);
        int groupsCells = ( (GRID_NX * GRID_NY * GRID_NZ) + localSize - 1) / localSize;
        glDispatchCompute(groupsCells, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // build lists
        if (uni_mode >= 0) glUniform1i(uni_mode, 1);
        int groupsBoids = (NUM_BOIDS + localSize - 1) / localSize;
        glDispatchCompute(groupsBoids, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // simulate
        if (uni_mode >= 0) glUniform1i(uni_mode, 2);
        if (uni_dt >= 0) glUniform1f(uni_dt, 0.016f);
        glDispatchCompute(groupsBoids, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

        // Copy updated positions from posSSBO to boidVBO and render from boidVBO
        glBindBuffer(GL_COPY_READ_BUFFER, posSSBO);
        glBindBuffer(GL_COPY_WRITE_BUFFER, boidVBO);
        GLsizeiptr copySize = static_cast<GLsizeiptr>(NUM_BOIDS) * 4 * sizeof(float);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, copySize);
        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);

        glBindBuffer(GL_ARRAY_BUFFER, boidVBO);
        // bind color VBO
        if (colorVBO != 0) {
            glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
            glEnableClientState(GL_COLOR_ARRAY);
            glColorPointer(4, GL_FLOAT, 0, reinterpret_cast<void*>(0));
        }
        glBindBuffer(GL_ARRAY_BUFFER, boidVBO);
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(3, GL_FLOAT, 4 * sizeof(float), reinterpret_cast<void*>(0));
        glPointSize(4.0f);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(NUM_BOIDS));
        glDisableClientState(GL_VERTEX_ARRAY);
        if (colorVBO != 0) {
            glDisableClientState(GL_COLOR_ARRAY);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glUseProgram(0);
    } else {
        // Fallback: draw boids using VBO-backed vertex arrays
        glPointSize(4.0f);
        if (boidVBO != 0 && !pointBuffer.empty()) {
            // upload CPU buffer to GPU VBO
            glBindBuffer(GL_ARRAY_BUFFER, boidVBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, pointBuffer.size() * sizeof(float), pointBuffer.data());
            // bind color VBO
            if (colorVBO != 0) {
                glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
                glEnableClientState(GL_COLOR_ARRAY);
                glColorPointer(4, GL_FLOAT, 0, 0);
            }
            glBindBuffer(GL_ARRAY_BUFFER, boidVBO);
            glEnableClientState(GL_VERTEX_ARRAY);
            glVertexPointer(3, GL_FLOAT, 0, 0);
            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(flock.size()));
            glDisableClientState(GL_VERTEX_ARRAY);
            if (colorVBO != 0) {
                glDisableClientState(GL_COLOR_ARRAY);
            }
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
    }

    glutSwapBuffers();
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            mouseDown = true;
            lastX = x;
            lastY = y;
        } else {
            mouseDown = false;
        }
    }
    // Support scroll wheel via buttons 3 (up) and 4 (down) on many GLUT implementations
    if (state == GLUT_DOWN) {
        if (button == 3) { // scroll up
            radius -= ZOOM_STEP;
            if (radius < MIN_RADIUS) radius = MIN_RADIUS;
            glutPostRedisplay();
        } else if (button == 4) { // scroll down
            radius += ZOOM_STEP;
            if (radius > MAX_RADIUS) radius = MAX_RADIUS;
            glutPostRedisplay();
        }
    }
}

// If using freeglut, this callback will be called for mouse wheel events
void mouseWheel(int wheel, int direction, int x, int y) {
    // direction is typically +1 (up) or -1 (down)
    radius -= static_cast<float>(direction) * ZOOM_STEP;
    if (radius < MIN_RADIUS) radius = MIN_RADIUS;
    if (radius > MAX_RADIUS) radius = MAX_RADIUS;
    glutPostRedisplay();
}

// Helper: compile a shader and return ID (or 0 on failure)
static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0');
        glGetShaderInfoLog(s, len, nullptr, &log[0]);
        std::cerr << "Shader compile error: " << log << std::endl;
        glDeleteShader(s);
        return 0;
    }
    return s;
}

// Create a simple compute shader that does a cheap update
static void createComputeProgram() {
#if 1
#define STR(x) #x
    const char* computeSrc = R"GLSL(
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Pos { vec4 pos[]; };
layout(std430, binding = 1) buffer Vel { vec4 vel[]; };
layout(std430, binding = 2) buffer Head { int head[]; };
layout(std430, binding = 3) buffer Next { int next[]; };
uniform int mode; // 0=clear heads, 1=build lists, 2=simulate
uniform int numBoids;
uniform int numCells;
uniform int gridNx;
uniform int gridNy;
uniform int gridNz;
uniform float cellSize;
uniform float dt;
uniform float maxSpeed;
uniform float bounds;
uniform float separationDist;
uniform float alignmentDist;
uniform float cohesionDist;
uniform float sepWeight;
uniform float aliWeight;
uniform float cohWeight;

// Helper: clamp integer into [0,ub)
int clampi(int v, int ub) {
    if (v < 0) return 0;
    if (v >= ub) return ub - 1;
    return v;
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (mode == 0) {
        // clear head array
        if (gid < uint(numCells)) {
            head[gid] = -1;
        }
        return;
    }
    if (mode == 1) {
        // build linked lists: atomic push to head[cell]
        if (gid >= uint(numBoids)) return;
        vec3 p = pos[gid].xyz;
        int ix = int(floor((p.x + bounds) / cellSize));
        int iy = int(floor((p.y + bounds) / cellSize));
        int iz = int(floor((p.z + bounds) / cellSize));
        ix = clampi(ix, gridNx);
        iy = clampi(iy, gridNy);
        iz = clampi(iz, gridNz);
        int cell = ix + iy * gridNx + iz * gridNx * gridNy;
        int prev = atomicExchange(head[cell], int(gid));
        next[gid] = prev;
        return;
    }
    if (mode == 2) {
        if (gid >= uint(numBoids)) return;
        vec3 p = pos[gid].xyz;
        vec3 v = vel[gid].xyz;
        int ix = int(floor((p.x + bounds) / cellSize));
        int iy = int(floor((p.y + bounds) / cellSize));
        int iz = int(floor((p.z + bounds) / cellSize));
        ix = clampi(ix, gridNx);
        iy = clampi(iy, gridNy);
        iz = clampi(iz, gridNz);

        float sepDist2 = separationDist * separationDist;
        float aliDist2 = alignmentDist * alignmentDist;
        float cohDist2 = cohesionDist * cohesionDist;

        vec3 separation = vec3(0.0);
        vec3 alignment = vec3(0.0);
        vec3 cohesion = vec3(0.0);
        int sepCount = 0;
        int aliCount = 0;
        int cohCount = 0;

        // iterate neighbor cells (3x3x3)
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = clampi(ix + dx, gridNx);
                    int ny = clampi(iy + dy, gridNy);
                    int nz = clampi(iz + dz, gridNz);
                    int cell = nx + ny * gridNx + nz * gridNx * gridNy;
                    int j = head[cell];
                    while (j != -1) {
                        if (j != int(gid)) {
                            vec3 op = pos[j].xyz;
                            vec3 ov = vel[j].xyz;
                            vec3 diff = p - op;
                            float dist2 = dot(diff, diff);
                            if (dist2 < sepDist2) {
                                separation += normalize(diff);
                                sepCount++;
                            }
                            if (dist2 < aliDist2) {
                                alignment += ov;
                                aliCount++;
                            }
                            if (dist2 < cohDist2) {
                                cohesion += op;
                                cohCount++;
                            }
                        }
                        j = next[j];
                    }
                }
            }
        }

        vec3 steer = vec3(0.0);
        if (sepCount > 0) {
            separation /= float(max(1, sepCount));
            steer += (normalize(separation) * maxSpeed - v) * sepWeight;
        }
        if (aliCount > 0) {
            alignment /= float(max(1, aliCount));
            steer += (normalize(alignment) * maxSpeed - v) * aliWeight;
        }
        if (cohCount > 0) {
            cohesion /= float(max(1, cohCount));
            vec3 toCenter = cohesion - p;
            steer += (normalize(toCenter) * maxSpeed - v) * cohWeight;
        }

        v += steer;
        v *= 0.999; // damping
        float sp = length(v);
        if (sp > maxSpeed) v = normalize(v) * maxSpeed;
        p += v * dt;
        // reflect
        if (p.x > bounds) { p.x = bounds; v.x = -v.x; }
        if (p.x < -bounds) { p.x = -bounds; v.x = -v.x; }
        if (p.y > bounds) { p.y = bounds; v.y = -v.y; }
        if (p.y < -bounds) { p.y = -bounds; v.y = -v.y; }
        if (p.z > bounds) { p.z = bounds; v.z = -v.z; }
        if (p.z < -bounds) { p.z = -bounds; v.z = -v.z; }

        pos[gid] = vec4(p, 1.0);
        vel[gid] = vec4(v, 0.0);
        return;
    }
}
)GLSL";
#undef STR
#endif

    GLuint cs = compileShader(GL_COMPUTE_SHADER, computeSrc);
    if (!cs) return;
    GLuint prog = glCreateProgram();
    glAttachShader(prog, cs);
    glLinkProgram(prog);
    GLint ok = 0; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0');
        glGetProgramInfoLog(prog, len, nullptr, &log[0]);
        std::cerr << "Compute link error: " << log << std::endl;
        glDeleteProgram(prog);
        glDeleteShader(cs);
        return;
    }
    glDeleteShader(cs);
    computeProgram = prog;

    // Cache uniform locations to avoid per-frame lookups
    uni_mode = glGetUniformLocation(computeProgram, "mode");
    uni_numBoids = glGetUniformLocation(computeProgram, "numBoids");
    uni_numCells = glGetUniformLocation(computeProgram, "numCells");
    uni_gridNx = glGetUniformLocation(computeProgram, "gridNx");
    uni_gridNy = glGetUniformLocation(computeProgram, "gridNy");
    uni_gridNz = glGetUniformLocation(computeProgram, "gridNz");
    uni_cellSize = glGetUniformLocation(computeProgram, "cellSize");
    uni_dt = glGetUniformLocation(computeProgram, "dt");
    uni_maxSpeed = glGetUniformLocation(computeProgram, "maxSpeed");
    uni_separationDist = glGetUniformLocation(computeProgram, "separationDist");
    uni_alignmentDist = glGetUniformLocation(computeProgram, "alignmentDist");
    uni_cohesionDist = glGetUniformLocation(computeProgram, "cohesionDist");
    uni_sepWeight = glGetUniformLocation(computeProgram, "sepWeight");
    uni_aliWeight = glGetUniformLocation(computeProgram, "aliWeight");
    uni_cohWeight = glGetUniformLocation(computeProgram, "cohWeight");
    uni_bounds = glGetUniformLocation(computeProgram, "bounds");
}

void motion(int x, int y) {
    if (mouseDown) {
        yaw += static_cast<float>(x - lastX) * 0.01f;
        pitch += static_cast<float>(y - lastY) * 0.01f;
        if (pitch > 1.5f) pitch = 1.5f;
        if (pitch < -1.5f) pitch = -1.5f;
        lastX = x;
        lastY = y;
        glutPostRedisplay();
    }
}

void reshape(int w, int h) {
    windowWidth = w;
    windowHeight = h;
    glViewport(0, 0, w, h);
}

void timer(int value) {
    update();
    glutPostRedisplay();
    glutTimerFunc(16, timer, 0);
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(windowWidth, windowHeight);
    glutCreateWindow("3D Boids Simulation");

    // Initialize GLEW to get modern GL functions (compute shaders, SSBOs)
    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        std::cerr << "GLEW init failed: " << glewGetErrorString(glewErr) << std::endl;
    }

    // Create compute shader program if supported
    if (GLEW_VERSION_4_3) createComputeProgram();

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    // Make cube faces visible via blending and thicker lines
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLineWidth(3.0f);

    initializeFlock();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    // Register freeglut mouse wheel callback if available
    #ifdef GLUT_MOUSE_WHEEL
    glutMouseWheelFunc(mouseWheel);
    #else
    // Many GLUT implementations map wheel to buttons 3/4 which we already handle in mouse()
    #endif
    glutTimerFunc(0, timer, 0);

    glutMainLoop();
    return 0;
}