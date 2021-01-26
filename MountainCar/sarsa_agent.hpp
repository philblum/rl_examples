#pragma once

#include <cstdio>
#include <vector>
#include "rl_agent.hpp"
#include "mountain_car_tc.hpp"

using namespace rl;
using namespace agent;
using namespace mctc;

class SarsaAgent : public Agent
{
public:
    SarsaAgent() : Agent() { };
    ~SarsaAgent() {};

    virtual void agent_init(const AgentInit& params) override;
    virtual Action agent_start(const State state) override;
    virtual Action agent_step(const float reward, const State state) override;
    virtual void agent_end(const float reward) override;
    virtual void agent_cleanup() override;
    virtual std::string agent_message(const std::string& message) override;

protected:
    // Additional parameters for tile coding
    unsigned int num_tilings{0};
    unsigned int num_tiles{0};
    unsigned int index_hash_table_size{0};
    MountainCarTileCoder tc;
    std::vector<uint32_t> prev_tiles;
    float prev_q_value{0};
};
