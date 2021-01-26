
#include <cmath>
#include <stdexcept>
#include "tc.hpp"

namespace tc
{

TileCoder::TileCoder(const std::size_t capacity)
    : capacity(capacity)
{
}

TileCoder::TileCoder()
{
}

TileCoder::~TileCoder()
{
}

std::size_t TileCoder::get_size()
{
    return iht.size();
}

void TileCoder::set_capacity(const std::size_t _capacity)
{
    capacity = _capacity;
}

std::uint32_t TileCoder::get_index(const KeyType& k)
{
    auto search = iht.find(k);
    if (search != iht.end())
        return iht[k];

    auto size = iht.size();
    if (size >= capacity)
    {
        if (overflow_count == 0)
            std::printf("TileCoder: index hash table full, allowing collisions");
        ++overflow_count;
        //return KeyHash{}();
        throw(std::out_of_range("TileCoder: index hash table full"));
    }

    iht[k] = size;
    return size;
}

std::vector<std::uint32_t> TileCoder::get_tiles(const std::uint32_t num_tilings, const std::vector<float>& floats)
{
    std::vector<uint32_t> tiles;
    std::vector<int> qfloats;

    for (const auto& f : floats)
        qfloats.push_back(static_cast<int>(std::floor(f * num_tilings)));

    for (std::uint32_t tiling=0; tiling < num_tilings; ++tiling)
    {
        auto tiling_x2 = tiling * 2;
        auto b = tiling;

        KeyType coords =  // TODO: make this a length-n tuple
                std::make_tuple(
                                    tiling,
                                    (qfloats[0] + b) / num_tilings,
                                    (qfloats[1] + b + tiling_x2) / num_tilings
                               );

        tiles.push_back(get_index(coords));
    }

    return tiles;
}

std::vector<std::uint32_t> TileCoder::get_tileswrap(
        const std::uint32_t num_tilings, const std::vector<float>& floats,
        const std::vector<std::uint32_t>& wrap_widths)
{
    std::vector<uint32_t> tiles;
    std::vector<int> qfloats;

    for (const auto& f : floats)
        qfloats.push_back(static_cast<int>(std::floor(f * num_tilings)));

    for (std::uint32_t tiling=0; tiling < num_tilings; ++tiling)
    {
        auto tiling_x2 = tiling * 2;
        auto b = tiling;

        std::uint32_t c0 = tiling;
        std::uint32_t c1 = (qfloats[0] + b % num_tilings) / num_tilings;
        if (wrap_widths[0] > 0)
            c1 %= wrap_widths[0];
        std::uint32_t c2 = (qfloats[1] + (b + tiling_x2) % num_tilings) / num_tilings;
        if (wrap_widths[1] > 0)
            c2 %= wrap_widths[1];

        KeyType coords =  // TODO: make this a length-n tuple
                std::make_tuple(c0, c1, c2);

        tiles.push_back(get_index(coords));
    }

    return tiles;
}

} /* namespace tc */
