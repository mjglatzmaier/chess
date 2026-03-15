#include "havoc/version.hpp"

#include <gtest/gtest.h>

/// Smoke test: verify the test framework is working and version constants are sane.
TEST(Smoke, VersionStringNotEmpty) {
    EXPECT_FALSE(havoc::VERSION_STRING.empty());
    EXPECT_EQ(havoc::VERSION_MAJOR, 2);
}

TEST(Smoke, EngineNameIsHaVoc) {
    EXPECT_EQ(havoc::ENGINE_NAME, "haVoc");
}
