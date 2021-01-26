#pragma once

#include <random>
#include <vector>
#include "rl_environment.hpp"

using namespace rl;
using namespace env;

class MountainCarEnvironment : public Environment
{
public:
	MountainCarEnvironment() : Environment() { }
    ~MountainCarEnvironment() {};

    virtual void env_init(const EnvironmentInit params) override;
    virtual Observation env_start() override;
    virtual Observation env_step(const Action action) override;
    virtual void env_cleanup() override;
    virtual std::string env_message(const std::string& message) override;

private:
    State current_state{};

    std::mt19937 gen;
    std::uniform_real_distribution<> rand_real;

    State convert_to_linear_state(const State& current_state) const;

};
