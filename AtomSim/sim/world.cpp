#include "world.hpp"

namespace sim {

namespace {

b2BodyId create_wall_segment(
    b2WorldId world, b2Vec2 a, b2Vec2 b, uint64_t category, uint64_t mask) {
    b2BodyDef body_def = b2DefaultBodyDef();
    body_def.type = b2_staticBody;
    const b2BodyId body = b2CreateBody(world, &body_def);

    b2Segment segment;
    segment.point1 = a;
    segment.point2 = b;

    b2ShapeDef shape_def = b2DefaultShapeDef();
    shape_def.filter.categoryBits = category;
    shape_def.filter.maskBits     = mask;
    b2CreateSegmentShape(body, &shape_def, &segment);
    return body;
}

}  // namespace

World::World(const WorldConfig& cfg) : config_(cfg) {
    b2WorldDef def = b2DefaultWorldDef();
    def.gravity = {config_.gravity_x, config_.gravity_y};
    world_id_ = b2CreateWorld(&def);
    create_walls();
}

World::~World() {
    if (b2World_IsValid(world_id_)) {
        b2DestroyWorld(world_id_);
    }
}

void World::step(float dt) {
    b2World_Step(world_id_, dt, config_.substeps);
}

void World::create_walls() {
    const float xh = config_.field_x_half   * kBox2dScale;
    const float yh = config_.field_y_half   * kBox2dScale;
    const float gh = config_.goal_y_half    * kBox2dScale;
    const float gx = config_.goal_extension * kBox2dScale;
    const bool has_goals =
        (config_.goal_y_half > 0.0f) && (config_.goal_extension > 0.0f);

    walls_.clear();
    walls_.reserve(has_goals ? 12 : 4);

    // Top + bottom walls (full-width) — same in both modes.
    walls_.push_back(create_wall_segment(
        world_id_, {-xh,  yh}, { xh,  yh}, CATEGORY_WALL, MASK_WALL));
    walls_.push_back(create_wall_segment(
        world_id_, {-xh, -yh}, { xh, -yh}, CATEGORY_WALL, MASK_WALL));

    if (!has_goals) {
        // Solid left + right walls.
        walls_.push_back(create_wall_segment(
            world_id_, {-xh, -yh}, {-xh,  yh}, CATEGORY_WALL, MASK_WALL));
        walls_.push_back(create_wall_segment(
            world_id_, { xh, -yh}, { xh,  yh}, CATEGORY_WALL, MASK_WALL));
        return;
    }

    // Left field wall split around the goal mouth (|y| ≤ gh).
    walls_.push_back(create_wall_segment(
        world_id_, {-xh,  gh}, {-xh,  yh}, CATEGORY_WALL, MASK_WALL));
    walls_.push_back(create_wall_segment(
        world_id_, {-xh, -yh}, {-xh, -gh}, CATEGORY_WALL, MASK_WALL));
    // Right field wall split.
    walls_.push_back(create_wall_segment(
        world_id_, { xh,  gh}, { xh,  yh}, CATEGORY_WALL, MASK_WALL));
    walls_.push_back(create_wall_segment(
        world_id_, { xh, -yh}, { xh, -gh}, CATEGORY_WALL, MASK_WALL));

    // Left goal chamber — three segments forming a U opening to the right.
    walls_.push_back(create_wall_segment(
        world_id_, {-xh - gx,  gh}, {-xh,  gh}, CATEGORY_GOAL_WALL, MASK_GOAL_WALL));
    walls_.push_back(create_wall_segment(
        world_id_, {-xh - gx, -gh}, {-xh, -gh}, CATEGORY_GOAL_WALL, MASK_GOAL_WALL));
    walls_.push_back(create_wall_segment(
        world_id_, {-xh - gx, -gh}, {-xh - gx, gh}, CATEGORY_GOAL_WALL, MASK_GOAL_WALL));

    // Right goal chamber — opens to the left.
    walls_.push_back(create_wall_segment(
        world_id_, { xh,  gh}, { xh + gx,  gh}, CATEGORY_GOAL_WALL, MASK_GOAL_WALL));
    walls_.push_back(create_wall_segment(
        world_id_, { xh, -gh}, { xh + gx, -gh}, CATEGORY_GOAL_WALL, MASK_GOAL_WALL));
    walls_.push_back(create_wall_segment(
        world_id_, { xh + gx, -gh}, { xh + gx, gh}, CATEGORY_GOAL_WALL, MASK_GOAL_WALL));
}

}  // namespace sim
