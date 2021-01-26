
#include "pendulum_env.hpp"

#include <cassert>
#include <cmath>
#include <type_traits>

using namespace rl;
using namespace env;

/* Pendulum Environment:
 *
 * The environment consists of single pendulum that can swing 360 degrees. The
 * pendulum is actuated by applying a torque on its pivot point. The goal is to
 * get the pendulum to balance up-right from its resting position (hanging down
 * at the bottom with no velocity) and maintain it as long as possible. The
 * pendulum can move freely, subject only to gravity and the action applied by
 * the agent.
 *
 * The state is 2-dimensional, the current angle (measured from the vertical
 * upright position) in the interval [-pi, pi] and current angular velocity in
 * the interval [-2pi, 2pi]. The angular velocity is constrained in order to
 * avoid damaging the pendulum system. If the angular velocity reaches this
 * limit during simulation, the pendulum is reset to the resting position. The
 * action is the angular acceleration, with discrete values {-1, 0, 1} applied
 * to the pendulum. For more details on environment dynamics refer to the
 * original paper, Santamaría et al. (1998).
 *
 * The goal is to swing-up the pendulum and maintain its upright angle. The
 * reward is the negative absolute angle from the vertical position. Since the
 * goal is to reach and maintain a vertical position, this is a continuing task.
 * The action in this pendulum environment is not strong enough to move the
 * pendulum directly to the desired position. The agent must learn to first
 * move the pendulum away from its desired position and gain enough momentum to
 * successfully swing-up the pendulum. And even after reaching the upright
 * position the agent must learn to continually balance the pendulum in this
 * unstable position."
 */
void PendulumEnvironment::env_init(const EnvironmentInit params)
{
    if (std::is_empty<EnvironmentInit>::value)
        (void)params;

    dt = 0.05;

    last_state = {0, 0};
    last_action = 0;
}

/* The first method called when the episode starts; called before the
 * agent starts.
 * Returns:
 *     observation - the first state observation from the environment
 */
Observation PendulumEnvironment::env_start()
{
    last_state = { -pi, 0.0 };
    observation = { 0, last_state, false };

    return observation;
}

double mod(double x, double y)
{
    // Use x = (x // y) * y + (x % y) --> (x % y) = x - (x // y) * y
    // for a python equivalent
    return x - std::floor(x / y) * y;
}

/* Update the state according to the transition dynamics
 * normalize the angle so that it is always between -pi and pi.
 * If the angular velocity exceeds the bound, reset the state to the resting position
 * Compute reward according to the new state, and is_terminal should always be false
 */
Observation PendulumEnvironment::env_step(const Action action)
{
    double reward{0};
    bool is_terminal = false;

    double angle = last_state.angle;
    double velocity = last_state.velocity;

    velocity += 0.75 * ((action - 1.0) + mass * length * gravity * std::sin(angle)) /
            (mass * length * length) * dt;

    angle += velocity * dt;
    auto save_angle = angle;

    angle = mod(angle + pi, 2 * pi) - pi;  // normalize angle

    if (std::abs(velocity) > 2 * pi)  // reset if velocity is out of bounds
    {
        angle = -pi;
        velocity = 0.0;
    }

    // compute reward
    reward = -std::abs(mod(angle + pi, 2 * pi) - pi);
    last_state.angle = angle;
    last_state.velocity = velocity;

    observation = { reward, last_state, is_terminal };

    return observation;
}

void PendulumEnvironment::env_cleanup() { }
std::string PendulumEnvironment::env_message(const std::string& message)
{
    return std::string("");
}

/* Helper function to map the internal state to the linear state used by the
 * agent and environment.
 * Returns:
 *     state - the linear (integral) state
 */
State PendulumEnvironment::convert_to_linear_state(const State& current_state) const
{
    return current_state;
}
