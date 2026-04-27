#include "world.hpp"

namespace sim {

namespace {

b2BodyId create_wall_segment(b2WorldId world, b2Vec2 a, b2Vec2 b) {
    b2BodyDef body_def = b2DefaultBodyDef();
    body_def.type = b2_staticBody;
    const b2BodyId body = b2CreateBody(world, &body_def);

    b2Segment segment;
    segment.point1 = a;
    segment.point2 = b;

    b2ShapeDef shape_def = b2DefaultShapeDef();
    shape_def.filter.categoryBits = CATEGORY_WALL;
    shape_def.filter.maskBits     = MASK_WALL;
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
    const float xh = config_.field_x_half;
    const float yh = config_.field_y_half;

    // left, right, bottom, top
    walls_[0] = create_wall_segment(world_id_, {-xh, -yh}, {-xh,  yh});
    walls_[1] = create_wall_segment(world_id_, { xh, -yh}, { xh,  yh});
    walls_[2] = create_wall_segment(world_id_, {-xh, -yh}, { xh, -yh});
    walls_[3] = create_wall_segment(world_id_, {-xh,  yh}, { xh,  yh});
}

}  // namespace sim
