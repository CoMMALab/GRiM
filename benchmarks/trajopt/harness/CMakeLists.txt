# TrajOpt harness (../trajopt_harness.cpp) against the Tesseract stack built from benchmarks/external
# (see RUNNING.md: tesseract_ws, the tesseract-src env). JsonCpp::JsonCpp comes with trajopt.
cmake_minimum_required(VERSION 3.16)
project(trajopt_harness CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(tesseract REQUIRED COMPONENTS common environment geometry scene_graph collision collision_bullet kinematics_kdl)
find_package(tesseract_planning REQUIRED COMPONENTS command_language motion_planners motion_planners_simple motion_planners_trajopt)
add_executable(trajopt_harness ../trajopt_harness.cpp)
target_link_libraries(trajopt_harness tesseract::motion_planners_trajopt tesseract::motion_planners_simple
                      tesseract::motion_planners tesseract::command_language tesseract::environment JsonCpp::JsonCpp)
