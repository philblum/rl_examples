#pragma once

#include <cmath>
#include <random>
#include <vector>

#include "rl_env.hpp"

using namespace rl;
using namespace env;

class PendulumEnvironment : public Environment
{
public:
    PendulumEnvironment() : Environment() { }
    ~PendulumEnvironment() {};

    virtual void env_init(const EnvironmentInit params) override;
    virtual Observation env_start() override;
    virtual Observation env_step(const Action action) override;
    virtual void env_cleanup() override;
    virtual std::string env_message(const std::string& message) override;

private:
    State last_state{};
    Action last_action{};
    Observation observation;

    std::mt19937 gen;
    std::uniform_real_distribution<> rand_real;

    double dt{};
    static constexpr double pi = 4 * std::atan(1);
    static constexpr double gravity = 9.8;
    static constexpr double mass = 1.0 / 3.0;
    static constexpr double length = 3.0 / 2.0;

    State convert_to_linear_state(const State& current_state) const;

};
