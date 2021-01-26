#pragma once
#include <cstdio>
#include "rl_agent.hpp"

using namespace rl;
using namespace agent;

class ExpectedSarsaAgent : public Agent
{
public:
    ExpectedSarsaAgent() : Agent() { }
    ~ExpectedSarsaAgent() {};

    virtual void agent_init(const AgentInit& params) override;
    virtual Action agent_start(const State state) override;
    virtual Action agent_step(const float reward, const State state) override;
    virtual void agent_end(const float reward) override;
    virtual void agent_cleanup() override;
    virtual std::string agent_message(const std::string& message) override;
};
