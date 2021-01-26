
#include <memory>
#include <string>
#include "rl.hpp"

namespace rl {

using namespace env;
using namespace agent;

void RL::rl_init(const EnvironmentInit& env_params, const AgentInit& agent_params)
{
    env->env_init(env_params);
    agent->agent_init(agent_params);

    total_reward = 0.0;
    last_action = 0;
    num_steps = 0;
    num_episodes = 0;
}

std::pair<State, Action> RL::rl_start()
{
    total_reward = 0.0;
    num_steps = 1;

    Observation obs = env->env_start();
    State last_state = obs.state;
    last_action = agent->agent_start(last_state);

    return std::make_pair(last_state, last_action);
}

Action RL::rl_agent_start(const State state)
{
    return agent->agent_start(state);
}

Action RL::rl_agent_step(const float reward, const State state)
{
    return agent->agent_step(reward, state);
}

void RL::rl_agent_end(const float reward)
{
    agent->agent_end(reward);
}

/* Starts the environment.
 * Returns:
 *     observation - the initial reward, state, and termination
 */
Observation RL::rl_env_start()
{
    total_reward = 0;
    num_steps = 1;

    return env->env_start();
}

Observation RL::rl_env_step(const Action action)
{
    auto observation = env->env_step(action);
    total_reward += observation.reward;

    if (observation.termination)
        num_episodes++;
    else
        num_steps++;

    return observation;
}

std::tuple<Observation, Action> RL::rl_step()
{
    auto obs = env->env_step(last_action);
    total_reward += obs.reward;

    if (obs.termination)
    {
        num_episodes++;
        agent->agent_end(obs.reward);
    }
    else
    {
        num_steps++;
        last_action = agent->agent_step(obs.reward, obs.state);
    }

    return std::make_tuple(obs, last_action);
}

void RL::rl_cleanup()
{
    env->env_cleanup();
    agent->agent_cleanup();
}

/* Run an episode
 *     Inputs:
 *         max_steps - the maximum number of steps in an episode
 *     Returns:
 *         is_terminal - if the episode should or has terminated
 */
bool RL::rl_episode(const unsigned int max_steps)
{
    Observation obs;
    bool is_terminal = false;
    rl_start();

    while (!is_terminal && ((max_steps == 0) || (num_steps < max_steps)))
    {
        std::tie(obs, std::ignore) = rl_step();
        is_terminal = obs.termination;
    }
    return true;
}




} // rl

