
#include <vector>
#include "expected_sarsa_agent.hpp"

using namespace rl;
using namespace agent;

using Float2D = boost::multi_array<float, 2>;
using range = boost::multi_array_types::index_range;

/* Setup for the agent when the RL environment starts.
 *     AgentInit is a structured class of parameters used to initialize the agent.
 */
void ExpectedSarsaAgent::agent_init(const AgentInit& params)
{
    num_actions = params.num_actions;
    num_states = params.num_states;
    step_size = params.step_size;
    epsilon = params.epsilon;
    discount = params.discount;
    seed = params.seed;

    prev_state = 0;
    prev_action = 0;

    gen = std::mt19937(seed);
    rand_real = std::uniform_real_distribution<>(0, 1);
    rand_int = std::uniform_int_distribution<>(0, num_actions-1);

    // Create an array for action-value estimates and initialize it to zero.
    Float2D::extent_gen extents;
    q_values.resize(extents[num_states][num_actions]);

    using index = Float2D::index;

    for(index i = 0; i < num_states; ++i)
      for(index j = 0; j < num_actions; ++j)
          q_values[i][j] = 0.0;
}

/* The first method called after the RL environment starts.
 *     Input: the state from the environmnent's env_start method.
 *     Returns: the first action taken by the agent.
 */
Action ExpectedSarsaAgent::agent_start(const State state)
{
    // Create a 1-d view (slice) indexed by state into the q_values 2-d array
    // current_q = q_values[state][...]
    Float2D::array_view<1>::type current_q =
            q_values[ boost::indices[state][range(0,num_actions)] ];

    // Choose action using epsilon greedy
    Action action{0};
    if (rand_real(gen) < epsilon)
        action = rand_int(gen);
    else
        std::tie(action, std::ignore) = argmax(current_q);

#if 0
    std::printf("%s: current_q(%lu,%lu)\n",__func__, current_q.shape()[0], current_q.shape()[1]);
    using index = Float2D::index;
    for(index i = 0; i < num_actions; ++i)
        std::printf("%f, ", current_q[i]);
    std::printf("\n");
#endif

    prev_state = state;
    prev_action = action;

    return action;
}
/* A generic step taken by the agent. Called from rl_step/rl_agent_step
 * after the environment take a step.
 *     Input: reward from the last action taken
 *            state from the environment's last step
 *     Returns: the action the agent is taking
 */
Action ExpectedSarsaAgent::agent_step(const float reward, const State state)
{
    // Create a 1-d view (slice) indexed by state into the q_values 2-d array
    // current_q = q_values[state][...]
    Float2D::array_view<1>::type current_q =
            q_values[ boost::indices[state][range(0,num_actions)] ];

    // Choose action using epsilon greedy
    Action action{0};
    float q_max{0};
    if (rand_real(gen) < epsilon)
        action = rand_int(gen);
    else
        std::tie(action, q_max) = argmax(current_q);

    std::vector<float> policy(num_actions, epsilon / num_actions);

    unsigned int q_max_count{0};
    float q_sum{0};
    using index = Float2D::index;
    for(index i = 0; i < num_actions; ++i)
    {
        q_sum += current_q[i];
        if (current_q[i] == q_max)
            q_max_count++;
    }

    for(index i = 0; i < num_actions; ++i)
    {
        if (current_q[i] == q_max)
            policy[i] += (1.0 - epsilon) / q_max_count;
    }

    float expected_return = epsilon / num_actions * q_sum + (1.0 - epsilon) * q_max;

    q_values[prev_state][prev_action] +=
            step_size * (reward + discount * expected_return - q_values[prev_state][prev_action]);

    prev_state = state;
    prev_action = action;

    return action;
}

/* Runs when the agent terminates.
 *     Input: reward the agent received for reaching the terminal state
 */
void ExpectedSarsaAgent::agent_end(const float reward)
{
    // Same action-value update as in agent_step but with expected_return = 0
    q_values[prev_state][prev_action] +=
            step_size * (reward + discount * 0.0 - q_values[prev_state][prev_action]);
}

void ExpectedSarsaAgent::agent_cleanup() { }

std::string ExpectedSarsaAgent::agent_message(const std::string& message)
{
    (void)message;
    return std::string("");
}
