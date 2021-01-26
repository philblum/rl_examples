
#include <stdexcept>
#include <type_traits>
#include "gridworldgame_environment.hpp"

using namespace rl;
using namespace env;

void GridWorldGameEnvironment::env_init(const EnvironmentInit params)
{
    if (std::is_empty<EnvironmentInit>::value)
        (void)params;


    gen = std::mt19937(std::random_device{}());
    rand_real = std::uniform_real_distribution<>(0, 1);
    rand_position = std::uniform_int_distribution<>(0, num_rows-1);
    rand_prize = std::uniform_int_distribution<>(0, num_prizes-1);

    prize_idx = num_prizes; // prize = 4 means no prize
    damaged = false;
    current_state = {};
}

/* The first method called when the episode starts; called before the
 * agent starts.
 * Returns:
 *     observation - the first state observation from the environment
 */
Observation GridWorldGameEnvironment::env_start()
{
    // this game is continuous; termination criteria will be some number of steps
    num_steps = 1;

    start_position = { rand_position(gen), rand_position(gen) };
    prize_idx = num_prizes; // prize = 4 means no prize
    current_state = {start_position.row, start_position.col, prize_idx, damaged};

    Observation observation = { 0, convert_to_linear_state(current_state), false };
    return observation;
}

Observation GridWorldGameEnvironment::env_step(const Action action0)
{
    auto row = current_state.row;
    auto col = current_state.col;
    float reward{0};

    Action action;
    Action rand_action = static_cast<Action>(rand_real(gen) * 20);
    if (rand_action < 4)
        action = rand_action;
    else
        action = action0;

    if (action == 0)  // move right
    {
        if (col == num_cols - 1)  // hit right wall; penalize
            reward = -1;
        else if ((row <= 1 && col == 0) || (row == 1 && col == 1))
            // hit internal wall
            reward = -1;
        else
            col++;
    }
    else if (action == 1)  // move down
    {
        if (row == num_rows - 1)  // hit bottom wall; penalize
            reward = -1;
        else
            row++;
    }
    else if (action == 2)  // move left
    {
        if (col == 0)  // hit left wall; penalize
            reward = -1;
        else
            col--;
    }
    else if (action == 3)  // move up
    {
        if (row == 0)  // hit top wall; penalize
            reward = -1;
        else
            row--;
    }
    else
    {
        throw std::out_of_range("invalid action");
    }

    // Should there be a prize? Apply criteria
    if (prize_idx == num_prizes && rand_real(gen) < prize_prob)
    {
        // select a new prize location (e.g., one of the four corners)
        prize_idx = rand_prize(gen);
    }

    // Did the agent get the prize?
    if (prize_idx < num_prizes)
    {
        if ((row == prize_positions[prize_idx].row) &&
            (col == prize_positions[prize_idx].col))
        {
            reward += 10;
            prize_idx = num_prizes;  // prize disappears
        }
    }

    // Did the monster get the agent?
    for (const auto pos : monster_positions)
    {
        if (rand_real(gen) < monster_prob)  // monster is present
        {
            if ((row == pos.row) && (col == pos.col))
            {
                if (damaged)
                    reward += -10;
                else
                    damaged = true;
                break;
            }
        }
    }

    // Does the agent get repaired?
    if ((row == repair_position.row) && (col == repair_position.col))
        damaged = false;

    current_state = { row, col, prize_idx, damaged };

    // this game is continuous; termination criteria will be some number of steps
    bool is_terminal = false;
    if (num_steps > 1000)
         is_terminal = true;
    else
        num_steps++;

    Observation observation = { reward, convert_to_linear_state(current_state), is_terminal };

    return observation;
}

void GridWorldGameEnvironment::env_cleanup()
{

}
std::string GridWorldGameEnvironment::env_message(const std::string& message)
{
    return std::string("");
}

/* Helper function to map the internal state to the linear state used by the
 * agent and environment.
 * Returns:
 *     state - the linear (integral) state
 */
State GridWorldGameEnvironment::convert_to_linear_state(const InternalState& current_state) const
{
    return ( (current_state.row * num_cols + current_state.col) * num_prizes \
            + current_state.prize_idx) * 2 + current_state.damaged;
}

