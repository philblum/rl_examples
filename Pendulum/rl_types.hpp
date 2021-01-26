#pragma once

namespace rl {

struct State
{
    double angle;
    double velocity;
};
using Action = unsigned int;

struct Observation {
    double reward{0.0};
    State state{0, 0};
    bool termination{false};
};

} // rl
