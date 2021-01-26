#pragma once

namespace rl {

struct State
{
    float position;
    float velocity;
};
using Action = unsigned int;

struct Observation {
    float reward{0.0};
    State state{0, 0};
    bool termination{false};
};

} // rl
