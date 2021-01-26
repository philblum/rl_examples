
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include "mountain_car_environment.hpp"

using namespace rl;
using namespace env;

/* Mountain Car Environment:
 *
 * The reward in this problem is -1 on all time steps until the car moves past its goal
 * position at the top of the mountain, which ends the episode. There are three
 * possible actions: full throttle forward (+1), full throttle reverse (-1), and zero
 * throttle (0). The car moves according to a simplified physics. Its position, x_t,
 * and velocity, v_t , are updated by:
 *     x_t+1 = bound(x_t + v_t+1)
 *     v_t+1 = bound(v_t + 0.001 * A_t - 0.0025 * cos(3x_t))
 * where the bound operation enforces -1.2 <= x_t+1 <= 0.5 and -0.07 <= v_t+1 <= 0.07.
 * In addition, when x_t+1 reaches the left bound, v_t+1 is reset to zero. When it
 * reaches the right bound, the goal is reached and the episode is terminated. Each
 * episode starts from a random position x_t in [-0.6, -0.4) and zero velocity. To
 * convert the two continuous state variables to binary features, uniform grid tilings
 * are used, with each tile covering 1/8th of the bounded distance in each dimension.
 * The feature vectors x(s, a) created by tile coding are combined linearly with the
 * parameter vector to approximate the action-value function:
 *     q_hat(s, a, w) = transp(w) * x(s,a) sum(i=1,d) w_i * x_i(s,a) for each state, s, and action, a.
 */
void MountainCarEnvironment::env_init(const EnvironmentInit params)
{
    if (std::is_empty<EnvironmentInit>::value)
        (void)params;


    gen = std::mt19937(std::random_device{}());
    rand_real = std::uniform_real_distribution<>(-0.6, -0.4);

    current_state = {};
}

/* The first method called when the episode starts; called before the
 * agent starts.
 * Returns:
 *     observation - the first state observation from the environment
 */
Observation MountainCarEnvironment::env_start()
{
    // this game is continuous; termination criteria will be some number of steps
    num_steps = 1;

    current_state = {rand_real(gen), 0};

    Observation observation = { 0, convert_to_linear_state(current_state), false };
    return observation;
}

template<class T>
constexpr const T& clamp( const T& v, const T& lo, const T& hi )
{
    assert( !(hi < lo) );
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

Observation MountainCarEnvironment::env_step(const Action action)
{
    float reward{-1};
    bool is_terminal = false;

    float position = current_state.position;
    float velocity = current_state.velocity;
    velocity = clamp<float>(velocity + 0.001 * (action - 1.0) - 0.0025 * std::cos(3.0 * position),
                          -0.07, 0.07);
    position = clamp<float>(position + velocity, -1.2, 0.5);

    if (position == -1.2)
        velocity = 0.0;
    else if (position == 0.5)
    {
        is_terminal = true;
        reward = 0.0;
    }
    current_state.position = position;
    current_state.velocity = velocity;

    Observation observation = { reward, convert_to_linear_state(current_state), is_terminal };

    return observation;
}

void MountainCarEnvironment::env_cleanup()
{

}
std::string MountainCarEnvironment::env_message(const std::string& message)
{
    return std::string("");
}

/* Helper function to map the internal state to the linear state used by the
 * agent and environment.
 * Returns:
 *     state - the linear (integral) state
 */
State MountainCarEnvironment::convert_to_linear_state(const State& current_state) const
{
    return current_state;
}
