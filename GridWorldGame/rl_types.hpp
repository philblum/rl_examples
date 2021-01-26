#pragma once

namespace rl {

using State = unsigned int;
using Action = unsigned int;

struct Observation {
    float reward{0.0};
    State state{0};
    bool termination{false};
};

} // rl
