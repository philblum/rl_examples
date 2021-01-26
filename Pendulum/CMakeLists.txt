cmake_minimum_required (VERSION 2.6)

project (Pendulum)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(
    /usr/local/boost_1_75_0
    )

set(CMAKE_MAKE_PROGRAM /usr/bin/make)
add_executable(Pendulum pendulum.cpp rl.cpp actor_critic_agent.cpp pendulum_env.cpp tc.cpp)
add_executable(PendulumTest pendulum_test.cpp rl.cpp actor_critic_agent.cpp pendulum_env.cpp tc.cpp)