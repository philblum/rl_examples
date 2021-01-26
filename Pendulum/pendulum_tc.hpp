#pragma once

#include <cmath>
#include <limits>
#include "tc.hpp"

using namespace tc;

namespace pendulum_tc {

class PendulumTileCoder : TileCoder
{
public:
    static constexpr float pi = 4 * std::atan(1);

    PendulumTileCoder() : TileCoder() {};
    ~PendulumTileCoder() {};

    void initialize(const std::size_t capacity, const std::uint32_t _num_tilings, const std::uint32_t _num_tiles)
    {
      tc.clear();
      tc.set_capacity(capacity);
      num_tilings = _num_tilings;
      num_tiles = _num_tiles;
    }

    /* Takes in an angle and angular velocity from the pendulum environment
     * and returns a vector of active tiles.
     * Arguments:
     *   angle -- float, the angle of the pendulum between -pi and pi
     *   velocity -- float, the angular velocity of the pendulum between -2pi and 2pi 
     * Returns:
     *   tiles - vector of active tiles
     */
    std::vector<std::uint32_t> get_tiles(const float angle, const float velocity)
    {
        static constexpr float min_float = std::numeric_limits<float>::epsilon();

        // Use the ranges above and num_tiles to scale position and velocity to the range [0, 1]
        // then multiply by num_tiles to scale the range to [0, num_tiles]
        float angle_scaled = (angle - (-pi)) / (pi - (-pi)) * num_tiles + min_float;
        float velocity_scaled = (velocity - (-2*pi)) / (2*pi - (-2*pi)) * num_tiles + min_float;

        std::vector<float> floats = {angle_scaled, velocity_scaled};
        //std::printf("%f, %f, %f, %f\n", angle, velocity, floats[0], floats[1]);

        // Get tiles by calling get_tileswrap method
        // wrap_widths specify which dimension to wrap over and its wrap_width
        const std::vector<std::uint32_t> wrap_widths{num_tiles, 0};
        return tc.get_tileswrap(num_tilings, floats, wrap_widths);
    }

private:
    TileCoder tc;
    std::uint32_t  num_tilings{0};
    std::uint32_t num_tiles{0};
};

} // namespace pendulum_tc
