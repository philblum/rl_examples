#pragma once

#include <cstdio>
#include <vector>
#include "rl_agent.hpp"
#include "pendulum_tc.hpp"

using namespace rl;
using namespace agent;
using namespace pendulum_tc;

struct ActorCriticInit : public AgentInit
{
    // Additional parameters for tile coding
    unsigned int num_tilings{0};
    unsigned int num_tiles{0};
    unsigned int index_hash_table_size{0};

    // Additional parameters for actor/critic
    double actor_step_size{0};
    double critic_step_size{0};
    double avg_reward_step_size{0};
};

class ActorCriticAgent : public Agent
{
public:
    ActorCriticAgent() : Agent() { };
    ~ActorCriticAgent() {};

    virtual void agent_init(const AgentInit& params) override;
    virtual Action agent_start(const State state) override;
    virtual Action agent_step(const double reward, const State state) override;
    virtual void agent_end(const double reward) override;
    virtual void agent_cleanup() override;
    virtual std::string agent_message(const std::string& message) override;

protected:
    // Additional parameters for tile coding
    unsigned int num_tilings{0};
    unsigned int num_tiles{0};
    unsigned int index_hash_table_size{0};
    PendulumTileCoder tc;
    std::vector<uint32_t> prev_tiles;

    double actor_step_size{0};
    double critic_step_size{0};
    double avg_reward_step_size{0};

    double avg_reward{0};

    std::vector<double> get_softmax_prob(
        const Float2D& actor_weights, const std::vector<uint32_t>& tiles);
    Action agent_policy(const std::vector<uint32_t>& tiles);

    Float2D actor_weights;
    std::vector<double> critic_weights;

    std::vector<double> softmax_prob;

};
