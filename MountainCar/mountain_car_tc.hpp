//#pragma once

#include "tc.hpp"

using namespace tc;

namespace mctc {

class MountainCarTileCoder : TileCoder
{
public:
    MountainCarTileCoder() : TileCoder() {};
    ~MountainCarTileCoder() {};

    void initialize(const std::size_t capacity, const std::uint32_t _num_tilings, const std::uint32_t _num_tiles)
    {
      tc.set_capacity(capacity);
      num_tilings = _num_tilings;
      num_tiles = _num_tiles;
    }

    /* Takes in a position and velocity from the mountain car environment
     * and returns a vector of active tiles.
     * Arguments:
     *   position -- float, the position of the agent between -1.2 and 0.5
     *   velocity -- float, the velocity of the agent between -0.07 and 0.07
     * Returns:
     *   tiles - vector of active tiles
     */
    std::vector<std::uint32_t> get_tiles(const float position, const float velocity)
    {
        static constexpr float min_float = std::numeric_limits<float>::epsilon();

        // Use the ranges above and num_tiles to scale position and velocity to the range [0, 1]
        // then multiply by num_tiles to scale the range to [0, num_tiles]
        float position_scaled = (position - (-1.2)) / (0.5 - (-1.2)) * num_tiles + min_float;
        float velocity_scaled = (velocity - (-0.07)) / (0.07 - (-0.07)) * num_tiles + min_float;

        std::vector<float> floats = {position_scaled, velocity_scaled};
        //std::printf("%f, %f, %f, %f\n", position, velocity, floats[0], floats[1]);
        return tc.get_tiles(num_tilings, floats);
    }

private:
    TileCoder tc;
    std::uint32_t  num_tilings{0};
    std::uint32_t num_tiles{0};
};

} // namespace mctc
