
#include <vector>
#include "sarsa_agent.hpp"

using namespace rl;
using namespace agent;
using namespace mctc;

using Float2D = boost::multi_array<float, 2>;
using range = boost::multi_array_types::index_range;

/* Setup for the agent when the RL environment starts.
 *     AgentInit is a structured class of parameters used to initialize the agent.
 */
void SarsaAgent::agent_init(const AgentInit& params)
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

    prev_state = {0, 0};
    prev_action = 0;

    gen = std::mt19937(seed);
    rand_real = std::uniform_real_distribution<>(0, 1);
    rand_int = std::uniform_int_distribution<>(0, num_actions-1);

    // Create an array for action-value estimates and initialize it to zero.
    Float2D::extent_gen extents;
    //q_values.resize(extents[num_states][num_actions]);

    // Using linear function approximation; need a set of weights for each action
    // The weights essential replace the q_values which are simply weights^T * x(s, a)
    // where the feature vector, x(s,a), is just the one-hot vector of active tiles
    weights.resize(extents[num_actions][index_hash_table_size]);

    using index = Float2D::index;

    for(index i = 0; i < num_actions; ++i)
      for(index j = 0; j < index_hash_table_size; ++j)
        weights[i][j] = 0.0;

    tc.initialize(index_hash_table_size, num_tilings, num_tiles);
}

/* The first method called after the RL environment starts.
 *     Input: the state from the environmnent's env_start method.
 *     Returns: the first action taken by the agent.
 */
Action SarsaAgent::agent_start(const State state)
{
    Action action{0};
    float q_value{0};
    auto tiles = tc.get_tiles(state.position, state.velocity);

    // Select epsilon greedy action
    std::tie(action, q_value) = select_action(tiles);

    prev_state = state;
    prev_action = action;
    prev_tiles = tiles;
    prev_q_value = q_value;

    return action;
}

/* A generic step taken by the agent. Called from rl_step/rl_agent_step
 * after the environment take a step.
 *     Input: reward from the last action taken
 *            state from the environment's last step
 *     Returns: the action the agent is taking
 */
Action SarsaAgent::agent_step(const float reward, const State state)
{
    Action action{0};
    float q_value{0};
    auto tiles = tc.get_tiles(state.position, state.velocity);

    // Choose action using epsilon greedy
    std::tie(action, q_value) = select_action(tiles);

    float update_target = reward + discount * q_value - prev_q_value;

    for (std::size_t j=0; j < prev_tiles.size(); ++j)
        weights[prev_action][prev_tiles[j]] += step_size * update_target;

    prev_state = state;
    prev_action = action;
    prev_tiles = tiles;
    prev_q_value = q_value;

    return action;
}

/* Runs when the agent terminates.
 *     Input: reward the agent received for reaching the terminal state
 */
void SarsaAgent::agent_end(const float reward)
{
    // Same action-value update as in agent_step but with expected_return = 0
    float update_target = reward - prev_q_value;

    for (std::size_t j=0; j < prev_tiles.size(); ++j)
        weights[prev_action][prev_tiles[j]] += step_size * update_target;
}

void SarsaAgent::agent_cleanup() { }

std::string SarsaAgent::agent_message(const std::string& message)
{
    (void)message;
    return std::string("");
}
