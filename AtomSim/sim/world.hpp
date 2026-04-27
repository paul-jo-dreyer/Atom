#pragma once

#include "types.hpp"

#include <box2d/box2d.h>

#include <array>

namespace sim {

// Owns the Box2D world plus the four field-boundary walls. Walls are static
// b2_segmentShape bodies with category=CATEGORY_WALL, mask=MASK_WALL — by
// design the ball's mask excludes them, so the ball never registers wall
// contacts. Robots collide with walls normally.
class World {
public:
    explicit World(const WorldConfig& cfg = {});
    ~World();

    World(const World&)            = delete;
    World& operator=(const World&) = delete;
    World(World&&)                 = delete;
    World& operator=(World&&)      = delete;

    // Advance the simulation by `dt` seconds.
    void step(float dt);

    const WorldConfig& config()   const { return config_; }
    b2WorldId          world_id() const { return world_id_; }

    // The four wall body IDs (left, right, bottom, top), exposed for tests
    // and debugging. They live for the lifetime of the World.
    const std::array<b2BodyId, 4>& wall_bodies() const { return walls_; }

private:
    void create_walls();

    WorldConfig             config_;
    b2WorldId               world_id_;
    std::array<b2BodyId, 4> walls_{};
};

}  // namespace sim
