cmake_minimum_required (VERSION 2.6)

project (MountainCar)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(
    /usr/include/c++/7
    /usr/include/x86_64-linux-gnu/c++/7
    /usr/include/c++/7/backward
    /usr/local/boost_1_75_0
    )

set(CMAKE_MAKE_PROGRAM /usr/bin/make)
add_executable(MountainCar mountain_car.cpp sarsa_agent.cpp mountain_car_environment.cpp rl.cpp tc.cpp)
