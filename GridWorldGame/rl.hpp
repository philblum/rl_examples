#pragma once

#include <cstdio>
#include <memory>
#include <string>
#include <tuple>
#include "rl_types.hpp"
#include "rl_environment.hpp"
#include "rl_agent.hpp"

namespace rl {

using namespace env;
using namespace agent;

class RL {
public:
    RL(std::shared_ptr<Environment> env, std::shared_ptr<Agent> agent)
    : env(env), agent(agent) { };
    virtual ~RL() = default;

    virtual void rl_init(const EnvironmentInit env_init, const AgentInit agent_init);
    virtual std::pair<State, Action> rl_start();
    virtual std::tuple<Observation, Action> rl_step();
    virtual void rl_cleanup();
    virtual bool rl_episode(unsigned int max_steps);
    virtual float rl_return() const { return total_reward; }
    virtual unsigned int rl_num_steps() const { return num_steps; }
    virtual unsigned int rl_num_episodes() const { return num_episodes; }

protected:
    virtual Observation rl_env_start();
    virtual Observation rl_env_step(const Action action);
    virtual std::string rl_env_message(const std::string& message)
            { return env->env_message(message); }
    virtual Action rl_agent_start(const State state);
    virtual Action rl_agent_step(const float reward, const State state);
    virtual void rl_agent_end(const float reward);
    virtual std::string rl_agent_message(const std::string& message)
            { return agent->agent_message(message); }


private:
    std::shared_ptr<Environment> env;
    std::shared_ptr<Agent> agent;
    float total_reward{0.0};
    Action last_action{};
    unsigned int num_steps{0};
    unsigned int num_episodes{0};

};

} // rl
