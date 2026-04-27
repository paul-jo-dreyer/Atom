#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "world.hpp"

TEST_CASE("WorldConfig defaults match the agreed field dimensions") {
    sim::WorldConfig cfg;
    CHECK(cfg.field_x_half == doctest::Approx(0.375f));   // 0.75 m / 2
    CHECK(cfg.field_y_half == doctest::Approx(0.225f));   // 0.45 m / 2
    CHECK(cfg.gravity_x    == doctest::Approx(0.0f));
    CHECK(cfg.gravity_y    == doctest::Approx(0.0f));
    CHECK(cfg.substeps     == 4);
}

TEST_CASE("World creates, steps, and destroys cleanly") {
    sim::World world{sim::WorldConfig{}};
    CHECK(b2World_IsValid(world.world_id()));

    for (int i = 0; i < 100; ++i) {
        world.step(0.01f);
    }
    CHECK(b2World_IsValid(world.world_id()));
    // Destruction happens at scope exit; doctest will fail if it crashes.
}

TEST_CASE("Custom WorldConfig is honored") {
    sim::WorldConfig cfg;
    cfg.field_x_half = 1.0f;
    cfg.field_y_half = 0.5f;
    cfg.substeps     = 8;

    sim::World world(cfg);
    CHECK(world.config().field_x_half == doctest::Approx(1.0f));
    CHECK(world.config().field_y_half == doctest::Approx(0.5f));
    CHECK(world.config().substeps     == 8);
}
