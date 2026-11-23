"""Sphere-only FK builds, with no global pose buffer or placement launch."""
from . import _build


def source(robot, joints, centres, traced):
    header = _build.robot_header(robot, _build.Problem())
    if traced:
        import mantis
        from mantis import lib
        from ._mantis_ik import _model
        model = _model(robot)

        @mantis.algorithm(inputs=("q",), outputs=("centres",), name="sphere_fk", register=False)
        def sphere_fk(model, q, rows=()):
            moving = [(f, c) for f, c in rows if f is not None]
            xyz = lib.sphere_centres(model, lib.poses(model, q), moving) if moving else []
            out, k = [], 0
            for frame, centre in rows:
                if frame is None:
                    out.extend(lib.Scalar.const(model.t, float(v)) for v in centre)
                else:
                    out.extend(xyz[k:k + 3])
                    k += 3
            return out

        rows = tuple((robot.joint_names[j] if j >= 0 else None, tuple(map(float, c)))
                     for j, c in zip(joints, centres))
        header += mantis.generate(model, {sphere_fk: {"rows": rows}},
                                  variants=[mantis.Variant("thread")], name="sphere_robot")
        body = [f"float qm[{max(model.n_q, 1)}];"]
        body += [f"qm[{i}] = q[{robot.actuated_names.index(n)}];"
                 for i, n in enumerate(model.joint_names)]
        body += ["mantis::robots::sphere_robot::sphere_fk_thread<float>(qm, out);"]
    else:
        # Static ancestor closure; reuse general FK's baked tables and SE(3) math.
        needed = {int(a) for j in joints if j >= 0 for a in robot.chain(int(j))}
        body = []
        for j in robot.topo_inv:
            if int(j) not in needed:
                continue
            body += [f"float T{j}[7];", "{", "float pc[7];"]
            a = int(robot.mimic_act_idx[j] if robot.mimic_act_idx[j] >= 0 else robot.act_idx[j])
            if a < 0 and robot.mimic_off[j] == 0:
                body += [f"for (int k=0; k<7; ++k) pc[k] = kParentTf[{j}*7+k];"]
            else:
                value = f"q[{a}]" if a >= 0 else "0.0f"
                body += [f"float v = {value} * kMimicMul[{j}] + kMimicOff[{j}];",
                         "float tangent[6], delta[7];",
                         f"for (int k=0; k<6; ++k) tangent[k] = kTwists[{j}*6+k]*v;",
                         "se3_exp(tangent, delta);",
                         f"se3_compose(kParentTf+{j}*7, delta, pc);"]
            p = int(robot.parent_idx[j])
            body += ([f"se3_compose(T{p}, pc, T{j});"] if p >= 0 else
                     [f"for (int k=0; k<7; ++k) T{j}[k] = pc[k];"])
            body += ["}"]
        header += _build._constant("float", "kSphereCentres", centres)
        for s, j in enumerate(joints):
            if j < 0:
                body += [f"for (int k=0; k<3; ++k) out[{s}*3+k] = kSphereCentres[{s}*3+k];"]
            else:
                body += ["{ float v[3];",
                         f"quat_rotate(T{j}, kSphereCentres+{s}*3, v);",
                         f"for (int k=0; k<3; ++k) out[{s}*3+k] = v[k]+T{j}[4+k];", "}"]
    return (header + "\nnamespace grim::robot {\n"
            + f"constexpr int sphere_count = {len(joints)};\n"
            + "static __device__ __forceinline__ void sphere_centres(const float* q, float* out) {\n"
            + "\n".join(body) + "\n}\n}\n")


def target(robot, joints, centres, traced):
    header = source(robot, joints, centres, traced)
    so = _build.build("sphere_fk", header, robot.n_act, robot.n_joints, 0)
    return _build.register(so, ("SphereFkFfi",))[0]
