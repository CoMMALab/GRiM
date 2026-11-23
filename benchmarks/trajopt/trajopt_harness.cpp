// TrajOpt (tesseract_planning / trajopt 0.35.0, built from benchmarks/external) on the shared MBM
// problems, as tesseract_planning's own motion_planners/trajopt/test/trajopt_planner_tests.cpp drives
// its planner: TrajOptMotionPlanner with TrajOptDefaultMoveProfile, TrajOptDefaultCompositeProfile
// and TrajOptOSQPSolverProfile. Beyond the defaults it sets only what the Python wheel cannot reach:
// the SQP iteration limit (opt_params.max_iter, the benchmark's budget) and the collision margin
// (both collision configs' contact margin). With TRAJOPT_PROFILE=example in the environment, the
// collision configs are instead those of tesseract_planning's examples/src/pick_and_place_example.cpp
// (its TrajOpt free-space profile): LVS_DISCRETE checks (longest valid segment 0.05), constraint
// coeff 10 + margin buffer 0.005, cost coeff 50 at 5 mm beyond the margin + buffer 0.01. The
// example's margins are relative to its 0 margin; here they sit on the benchmark margin.
//
//   trajopt_harness <urdf> <srdf> <problems.json> <first> <last> <T> <max_iter|0> <margin|-1> <out.jsonl> [iterates]
//
// With a tenth argument (any value), each row also carries "iterates": the joint values at the
// start of every SQP iteration of the timed solve (sco's own optimizer callback), and "iterate_ms": when each was taken.
//
// Problems [first, last) of the JSON mbm.load caches. Per problem: the environment and the
// straight-line seed (generateInterpolatedProgram, exactly T states) are built untimed, and so is
// the SQP problem: planner.solve's body (tesseract_planning 807c386958,
// motion_planners/trajopt/src/trajopt_motion_planner.cpp) is split at its opt->optimize(), which
// alone is timed, as every other method in bench_trajopt.py times only its solve (the first problem
// is solved once untimed first). One JSON line per problem: index, ms, successful, the trajectory.
#include <tesseract/common/profile_dictionary.h>
#include <tesseract/common/resource_locator.h>
#include <tesseract/environment/commands/add_link_command.h>
#include <tesseract/environment/environment.h>
#include <tesseract/geometry/impl/box.h>
#include <tesseract/geometry/impl/cylinder.h>
#include <tesseract/scene_graph/joint.h>
#include <tesseract/scene_graph/link.h>
#include <tesseract/command_language/composite_instruction.h>
#include <tesseract/command_language/joint_waypoint.h>
#include <tesseract/command_language/move_instruction.h>
#include <tesseract/command_language/state_waypoint.h>
#include <tesseract/motion_planners/simple/interpolation.h>
#include <tesseract/motion_planners/trajopt/profile/trajopt_default_composite_profile.h>
#include <tesseract/motion_planners/trajopt/profile/trajopt_default_move_profile.h>
#include <tesseract/motion_planners/trajopt/profile/trajopt_osqp_solver_profile.h>
#include <tesseract/motion_planners/trajopt/trajopt_motion_planner.h>
#include <tesseract/motion_planners/types.h>

#include <tesseract/common/kinematic_limits.h>
#include <tesseract/kinematics/joint_group.h>
#include <trajopt/problem_description.hpp>
#include <trajopt_common/logging.hpp>
#include <trajopt/utils.hpp>
#include <trajopt_sco/optimizers.hpp>

#include <json/json.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace tesseract::command_language;
using namespace tesseract::motion_planners;
static const std::string NS = "TrajOptMotionPlannerTask";

// TrajOptMotionPlanner::solve, line for line, split in two: construction (the problem, the
// optimizer, its callbacks and initial point), then optimize() and its result with limits enforced.
struct Solve
{
  trajopt::TrajOptProb::Ptr problem;
  sco::BasicTrustRegionSQP::Ptr opt;

  Solve(const TrajOptMotionPlanner& planner, const PlannerRequest& request)
  {
    std::shared_ptr<trajopt::ProblemConstructionInfo> pci = planner.createProblem(request);
    problem = trajopt::ConstructProblem(*pci);
    trajopt_common::gLogLevel = trajopt_common::LevelWarn;
    if (pci->opt_info.num_threads > 1)
      opt = std::make_shared<sco::BasicTrustRegionSQPMultiThreaded>(problem);
    else
      opt = std::make_shared<sco::BasicTrustRegionSQP>(problem);
    opt->setParameters(pci->opt_info);
    for (const sco::Optimizer::Callback& callback : pci->callbacks)
      opt->addCallback(callback);
    opt->initialize(trajopt::trajToDblVec(problem->GetInitTraj()));
  }

  bool optimize() const
  {
    opt->optimize();
    return opt->results().status == sco::OptStatus::OPT_CONVERGED;
  }

  tesseract::common::TrajArray traj() const
  {
    const Eigen::MatrixX2d joint_limits = problem->GetKin()->getLimits().joint_limits;
    tesseract::common::TrajArray t = trajopt::getTraj(opt->x(), problem->GetVars());
    for (Eigen::Index i = 0; i < t.rows(); i++)
      tesseract::common::enforceLimits<double>(t.row(i), joint_limits);
    return t;
  }
};

static Eigen::Isometry3d pose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
{
  Eigen::Isometry3d X = Eigen::Isometry3d::Identity();
  X.linear() = R;
  X.translation() = t;
  return X;
}

static void add_obstacle(tesseract::environment::Environment& env, const std::string& name,
                         const tesseract::geometry::Geometry::Ptr& geom, const Eigen::Isometry3d& X)
{
  tesseract::scene_graph::Link link(name);
  auto col = std::make_shared<tesseract::scene_graph::Collision>();
  col->geometry = geom;
  link.collision.push_back(col);
  tesseract::scene_graph::Joint joint(name + "_joint");
  joint.parent_link_name = "panda_link0";
  joint.child_link_name = name;
  joint.type = tesseract::scene_graph::JointType::FIXED;
  joint.parent_to_joint_origin_transform = X;
  if (!env.applyCommand(std::make_shared<tesseract::environment::AddLinkCommand>(link, joint)))
    throw std::runtime_error("could not add " + name);
}

int main(int argc, char** argv)
{
  if (argc != 10 && argc != 11)
  {
    std::cerr << "usage: trajopt_harness <urdf> <srdf> <problems.json> <first> <last> <T> <max_iter|0> <margin|-1> <out.jsonl> [iterates]\n";
    return 2;
  }
  const std::string urdf = argv[1], srdf = argv[2], problems_path = argv[3], out_path = argv[9];
  const int first = std::stoi(argv[4]), last = std::stoi(argv[5]), T = std::stoi(argv[6]), max_iter = std::stoi(argv[7]);
  const double margin = std::stod(argv[8]);

  Json::Value problems;
  std::ifstream(problems_path) >> problems;
  std::ofstream out(out_path);

  auto move_profile = std::make_shared<TrajOptDefaultMoveProfile>();
  auto composite_profile = std::make_shared<TrajOptDefaultCompositeProfile>();
  auto solver_profile = std::make_shared<TrajOptOSQPSolverProfile>();
  if (max_iter > 0)
    solver_profile->opt_params.max_iter = max_iter;
  const char* profile_env = std::getenv("TRAJOPT_PROFILE");
  // TRAJOPT_COEFF (tuning): the collision constraint's coefficient, in either profile.
  const char* coeff_env = std::getenv("TRAJOPT_COEFF");
  const double coeff = coeff_env != nullptr && *coeff_env ? std::stod(coeff_env) : 0.0;
  if (profile_env != nullptr && std::string(profile_env) == "example")
  {
    using tesseract::collision::CollisionEvaluatorType;
    const double m = margin >= 0 ? margin : 0.0;
    composite_profile->collision_constraint_config = trajopt_common::TrajOptCollisionConfig(m, coeff > 0 ? coeff : 10);
    composite_profile->collision_constraint_config.collision_check_config.type = CollisionEvaluatorType::LVS_DISCRETE;
    composite_profile->collision_constraint_config.collision_check_config.longest_valid_segment_length = 0.05;
    composite_profile->collision_constraint_config.collision_margin_buffer = 0.005;
    composite_profile->collision_cost_config = trajopt_common::TrajOptCollisionConfig(m + 0.005, 50);
    composite_profile->collision_cost_config.collision_check_config.type = CollisionEvaluatorType::LVS_DISCRETE;
    composite_profile->collision_cost_config.collision_check_config.longest_valid_segment_length = 0.05;
    composite_profile->collision_cost_config.collision_margin_buffer = 0.01;
  }
  else if (margin >= 0)
  {
    composite_profile->collision_cost_config.contact_manager_config =
        tesseract::collision::ContactManagerConfig(margin);
    composite_profile->collision_constraint_config.contact_manager_config =
        tesseract::collision::ContactManagerConfig(margin);
  }
  if (coeff > 0 && !(profile_env != nullptr && std::string(profile_env) == "example"))
    composite_profile->collision_constraint_config.collision_coeff_data = trajopt_common::CollisionCoeffData(coeff);
  std::vector<std::vector<double>> iterates;
  std::vector<double> iterate_ms;   // when each iterate was taken, from the solve's t0
  std::chrono::steady_clock::time_point t0;
  if (argc == 11)   // the joint variables come first in TrajOpt's problem: T x 7, row-major
    solver_profile->callbacks.emplace_back([&iterates, &iterate_ms, &t0, T](sco::OptProb*, sco::OptResults& r) {
      iterate_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      iterates.emplace_back(r.x.begin(), r.x.begin() + T * 7);
    });
  auto profiles = std::make_shared<tesseract::common::ProfileDictionary>();
  profiles->addProfile(NS, "DEFAULT", move_profile);
  profiles->addProfile(NS, "DEFAULT", composite_profile);
  profiles->addProfile(NS, "DEFAULT", solver_profile);
  TrajOptMotionPlanner planner(NS);
  auto locator = std::make_shared<tesseract::common::GeneralResourceLocator>();

  const std::vector<std::string> joints = { "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                                            "panda_joint5", "panda_joint6", "panda_joint7" };
  for (int i = first; i < last && i < static_cast<int>(problems.size()); ++i)
  {
    const Json::Value& p = problems[i];
    auto env = std::make_shared<tesseract::environment::Environment>();
    if (!env->init(std::filesystem::path(urdf), std::filesystem::path(srdf), locator))
      throw std::runtime_error("Tesseract could not load the robot");
    for (Json::ArrayIndex k = 0; k < p["boxes"].size(); ++k)
    {
      const Json::Value& b = p["boxes"][k];
      Eigen::Matrix3d R;   // GRiM layout: rows are the box axes; columns of R
      for (int a = 0; a < 3; ++a)
        for (int r = 0; r < 3; ++r)
          R(r, a) = b[3 + 3 * a + r].asDouble();
      add_obstacle(*env, "box" + std::to_string(k),
                   std::make_shared<tesseract::geometry::Box>(2 * b[12].asDouble(), 2 * b[13].asDouble(), 2 * b[14].asDouble()),
                   pose(R, Eigen::Vector3d(b[0].asDouble(), b[1].asDouble(), b[2].asDouble())));
    }
    for (Json::ArrayIndex k = 0; k < p["cylinders"].size(); ++k)
    {
      const Json::Value& c = p["cylinders"][k];
      Eigen::Vector3d z(c[3].asDouble(), c[4].asDouble(), c[5].asDouble());
      z.normalize();
      Eigen::Vector3d x = z.cross(std::abs(z.x()) < 0.9 ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY()).normalized();
      Eigen::Matrix3d R;
      R << x, z.cross(x), z;
      add_obstacle(*env, "cyl" + std::to_string(k),
                   std::make_shared<tesseract::geometry::Cylinder>(c[6].asDouble(), 2 * c[7].asDouble()),
                   pose(R, Eigen::Vector3d(c[0].asDouble(), c[1].asDouble(), c[2].asDouble())));
    }

    tesseract::common::ManipulatorInfo manip;
    manip.tcp_frame = "panda_hand";
    manip.manipulator = "manipulator";
    manip.working_frame = "panda_link0";
    Eigen::VectorXd qs(7), qg(7);
    for (int j = 0; j < 7; ++j)
    {
      qs[j] = p["start"][j].asDouble();
      qg[j] = p["goal"][j].asDouble();
    }
    CompositeInstruction program("DEFAULT");
    program.setManipulatorInfo(manip);
    program.push_back(MoveInstruction(JointWaypoint(joints, qs), MoveInstructionType::FREESPACE, "DEFAULT"));
    program.push_back(MoveInstruction(JointWaypoint(joints, qg), MoveInstructionType::FREESPACE, "DEFAULT"));
    // segment lengths too large to bind: min_steps alone sets the count (T states)
    CompositeInstruction seed = generateInterpolatedProgram(program, env, 1e6, 1e6, 1e6, T - 1);

    PlannerRequest request;
    request.instructions = seed;
    request.env = env;
    request.profiles = profiles;
    if (i == first)
      Solve(planner, request).optimize();   // warm-up: plugin loading and first-call costs, untimed
    const Solve solve(planner, request);   // the SQP problem, untimed
    iterates.clear();
    iterate_ms.clear();
    t0 = std::chrono::steady_clock::now();
    const bool successful = solve.optimize();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    Json::Value row;
    row["index"] = i;
    row["ms"] = ms;
    row["successful"] = successful;
    Json::Value traj(Json::arrayValue);
    const tesseract::common::TrajArray x = solve.traj();
    for (Eigen::Index k = 0; k < x.rows(); ++k)
    {
      Json::Value q(Json::arrayValue);
      for (Eigen::Index j = 0; j < x.cols(); ++j)
        q.append(x(k, j));
      traj.append(q);
    }
    row["traj"] = traj;
    if (argc == 11)
    {
      Json::Value its(Json::arrayValue);
      for (const auto& x : iterates)
      {
        Json::Value q(Json::arrayValue);
        for (double v : x)
          q.append(v);
        its.append(q);
      }
      row["iterates"] = its;
      Json::Value its_ms(Json::arrayValue);
      for (double v : iterate_ms)
        its_ms.append(v);
      row["iterate_ms"] = its_ms;
    }
    Json::StreamWriterBuilder w;
    w["indentation"] = "";
    out << Json::writeString(w, row) << "\n";
    out.flush();
  }
  return 0;
}
