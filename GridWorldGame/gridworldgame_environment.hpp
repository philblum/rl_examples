#pragma once

#include <random>
#include <vector>
#include "rl_environment.hpp"

using namespace rl;
using namespace env;

class GridWorldGameEnvironment : public Environment
{
public:
    GridWorldGameEnvironment() : Environment() { }
    ~GridWorldGameEnvironment() {};

    virtual void env_init(const EnvironmentInit params) override;
    virtual Observation env_start() override;
    virtual Observation env_step(const Action action) override;
    virtual void env_cleanup() override;
    virtual std::string env_message(const std::string& message) override;

private:
    struct Position { unsigned int row; unsigned int col; };

    struct InternalState
    {
        unsigned int row; unsigned int col;
        unsigned int prize_idx;
        bool damaged;
    };

    const unsigned int num_rows{5};
    const unsigned int num_cols{5};

    Position start_position{};

    const std::vector<Position> monster_positions = {
            {1, 2}, {2, 4}, {3, 0}, {3, 1}, {3, 3}
    };
    const float monster_prob = 0.4;

    const unsigned int num_prizes = 4;
    const std::vector<Position> prize_positions = {
            {1, 2}, {2, 4}, {3, 0}, {3, 1}, {3, 3}
    };
    unsigned int prize_idx = num_prizes; // prize = 4 means no prize
    const float prize_prob = 0.3;

    const Position repair_position = {0, 1};
    bool damaged = false;

    InternalState current_state{};

    std::mt19937 gen;
    std::uniform_int_distribution<> rand_position;
    std::uniform_int_distribution<> rand_prize;
    std::uniform_real_distribution<> rand_real;

    State convert_to_linear_state(const InternalState& current_state) const;
};
