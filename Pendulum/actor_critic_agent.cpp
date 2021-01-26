
#include <algorithm>
#include <random>
#include <vector>
#include "actor_critic_agent.hpp"

using namespace rl;
using namespace agent;
using namespace pendulum_tc;

using Float2D = boost::multi_array<double, 2>;
using range = boost::multi_array_types::index_range;


/* Setup for the agent when the RL environment starts.
 *     AgentInit is a structured class of parameters used to initialize the agent.
 */
void ActorCriticAgent::agent_init(const AgentInit& params)
{
    num_actions = params.num_actions;
    num_states = params.num_states;
    step_size = params.step_size;
    epsilon = params.epsilon;
    discount = params.discount;
    seed = params.seed;

    // Additional parameters for tile coding
    num_tilings = params.num_tilings;
    num_tiles = params.num_tiles;
    index_hash_table_size = params.index_hash_table_size;

    // Additional parameters for actor/critic agent
    actor_step_size = params.actor_step_size;
    critic_step_size = params.critic_step_size;
    avg_reward_step_size = params.avg_reward_step_size;

    gen.seed(seed);  // seed the random number generator
    avg_reward = 0;
    // Initialize the tile coder
    tc.initialize(index_hash_table_size, num_tilings, num_tiles);

    // Using linear function approximation; need a set of weights for each action
    // The weights essentially replace the q_values which are simply weights^T * x(s, a)
    // where the feature vector, x(s,a), is just the one-hot vector of active tiles
    Float2D::extent_gen extents;
    actor_weights.resize(extents[num_actions][index_hash_table_size]);
    critic_weights.resize(index_hash_table_size);

    using index = Float2D::index;

    for(index i = 0; i < num_actions; ++i)
      for(index j = 0; j < index_hash_table_size; ++j)
          actor_weights[i][j] = 0.0;

      for(int j = 0; j < index_hash_table_size; ++j)
          critic_weights[j] = 0.0;

    softmax_prob.resize(num_actions);

    prev_state = {0, 0};
    prev_action = 0;
}

/*
 * Computes softmax probability for all actions
 * Args:
 *   actor_weights - vector of actor weights for each action
 *   tiles - vector of active tiles
 *
 * Returns:
 * softmax_prob - array of probabilities for each action which sum to 1.
 */
std::vector<double> ActorCriticAgent::get_softmax_prob(
    const Float2D& actor_weights, const std::vector<uint32_t>& tiles)
{
    //auto num_actions = actor_weights.shape()[0];
    std::vector<double> p(num_actions, 0);

    // Form the action preferences, h(s, a, theta)

    std::vector<double> q_values(num_actions, 0);
    for(Action a = 0; a < num_actions; ++a)
    {
        for (std::size_t j=0; j < tiles.size(); ++j)
            q_values[a] += actor_weights[a][tiles[j]];
    }

    auto c = *std::max_element(q_values.cbegin(), q_values.cend());

    std::vector<double> numerator(num_actions, 0);
    double denominator{0};
    for (std::size_t i = 0; i < num_actions; ++i)
    {
        numerator[i] = std::exp(q_values[i] - c);
        denominator += numerator[i];
    }

    for (std::size_t i = 0; i < num_actions; ++i)
        p[i] = static_cast<double>(numerator[i] / denominator);

    return p;
}

Action ActorCriticAgent::agent_policy(const std::vector<std::uint32_t>& tiles)
{
    // Compute the softmax probability
    softmax_prob = get_softmax_prob(actor_weights, tiles);

    // Sample action from the softmax probability array
    // Select an element from the array with the specified probability
    static const std::vector<double> i{ 0, 1, 2, 3 };
    std::piecewise_constant_distribution<double> d(i.begin(), i.end(),
            softmax_prob.begin());
    Action action = d(gen);

    return action;
}

/* The first method called after the RL environment starts.
 *     Input: the state from the environmnent's env_start method.
 *     Returns: the first action taken by the agent.
 */
Action ActorCriticAgent::agent_start(const State state)
{
    auto tiles = tc.get_tiles(state.angle, state.velocity);
    Action action = agent_policy(tiles);

    //prev_state = state;
    prev_action = action;
    prev_tiles = tiles;

    return action;
}

/* A generic step taken by the agent. Called from rl_step/rl_agent_step
 * after the environment take a step.
 *     Input: reward from the last action taken
 *            state from the environment's last step
 *     Returns: the action the agent is taking
 */
Action ActorCriticAgent::agent_step(const double reward, const State state)
{
    auto tiles = tc.get_tiles(state.angle, state.velocity);
    Action action = agent_policy(tiles);

    double vhat{0};
    double prev_vhat{0};

    for (std::size_t j=0; j < tiles.size(); ++j)
        vhat += critic_weights[tiles[j]];

    for (std::size_t j=0; j < prev_tiles.size(); ++j)
        prev_vhat += critic_weights[prev_tiles[j]];

    // Compute delta
    double delta = (reward - avg_reward) + (vhat - prev_vhat);

    // Update average reward
    avg_reward += avg_reward_step_size * delta;

    // Update critic weights
    // grad = np.ones(len(self.prev_tiles))  # grad(vhat(S, w)) = x(S)
    // critic_weights[prev_tiles] += critic_step_size * delta * grad
    for (std::size_t j=0; j < prev_tiles.size(); ++j)
        critic_weights[prev_tiles[j]] += critic_step_size * delta;

    // Update actor weights
    // Use softmax_prob saved from the previous time step
    for (Action a = 0; a < num_actions; ++a)
    {
        if (a == prev_action)
            for (std::size_t j=0; j < prev_tiles.size(); ++j)
                actor_weights[a][prev_tiles[j]] += actor_step_size * delta * (1.0 - softmax_prob[a]);
        else
            for (std::size_t j=0; j < prev_tiles.size(); ++j)
                actor_weights[a][prev_tiles[j]] += actor_step_size * delta * (0.0 - softmax_prob[a]);
    }
    prev_state = state;
    prev_action = action;
    prev_tiles = tiles;

    return action;
}

/* Runs when the agent terminates.
 *     Input: reward the agent received for reaching the terminal state
 */
void ActorCriticAgent::agent_end(const double reward)
{
    // No need to implement agent_end() as there is no termination in a
    // continuing task.
    (void)reward;
}

void ActorCriticAgent::agent_cleanup() { }

std::string ActorCriticAgent::agent_message(const std::string& message)
{
    if (message == "get avg reward")
        return std::to_string(avg_reward);
    else
        return std::string("");
}
