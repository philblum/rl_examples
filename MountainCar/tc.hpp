#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "tuple_hash.hpp"

namespace tc {

// Simplified two coordinate tile coder
class TileCoder
{
public:
    TileCoder(const std::size_t capacity);
    TileCoder();
    virtual ~TileCoder();

    std::vector<std::uint32_t> get_tiles(const std::uint32_t num_tilings, const std::vector<float>& floats);
    void set_capacity(const std::size_t capacity);
    bool is_full() { return get_size() == capacity; }
    std::size_t get_size();

private:
    using KeyType = std::tuple<int, int, int>;
    using KeyHash = std::hash<KeyType>;

    std::uint32_t get_index(const KeyType& k);
    //std::uint32_t hash_coords;
    std::size_t capacity;
    std::size_t size{0};
    unsigned int overflow_count{0};

    struct KeyEqual : public std::binary_function<KeyType, KeyType, bool>
    {
        bool operator()(const KeyType& v1, const KeyType& v2) const
        {
            return(
                       std::get<0>(v1) == std::get<0>(v2) &&
                       std::get<1>(v1) == std::get<1>(v2) &&
                       std::get<2>(v1) == std::get<2>(v2)
                  );
        }
    };

    std::unordered_map<KeyType, std::size_t, KeyHash, KeyEqual> iht;
};

} // namespace tc
