#pragma once

#include "types.hpp"

#include <box2d/box2d.h>

#include <vector>

namespace sim {

// Owns the Box2D world plus the field walls. Field-perimeter walls are
// CATEGORY_WALL (ball passes through them, kept in by Ball's soft pull-back
// force). When goals are enabled (goal_y_half > 0 && goal_extension > 0) the
// left and right field walls are split around a gap, and three CATEGORY_
// GOAL_WALL segments bound a chamber behind each opening — the ball collides
// with these so it can't escape past the back of the goal.
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

    // All wall body IDs, in the order they were created. Exposed for tests
    // and debugging. Lifetime is tied to the World.
    const std::vector<b2BodyId>& wall_bodies() const { return walls_; }

private:
    void create_walls();

    WorldConfig            config_;
    b2WorldId              world_id_;
    std::vector<b2BodyId>  walls_{};
};

}  // namespace sim
