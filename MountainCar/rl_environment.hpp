#pragma once

#include <string>
#include "rl_types.hpp"

namespace rl {
namespace env {

struct EnvironmentInit {
};

class Environment {
public:
    Environment() = default;
    virtual ~Environment() = default;

    virtual void env_init(const EnvironmentInit params) = 0;
    virtual Observation env_start() = 0;
    virtual Observation env_step(const Action action) = 0;
    virtual void env_cleanup() = 0;
    virtual std::string env_message(const std::string& message) = 0;

protected:
    Observation observation;
    unsigned int num_steps{0};

};

} // env
} // rl

