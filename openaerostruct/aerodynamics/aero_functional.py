"""
Experimental: re-implement some key components of the aero group as a single component using
OpenMDAO's functional interface, along with jax AD. This has potential to be much faster because
it eliminates passing several large matrices through OpenMDAO's layers, all of which contribute
to making the linear problem much larger.  Computing the analytic derivatives of the combined code
is really not feasible, so the jax AD provides some potential for better performance.
"""
import numpy as np

import openmdao.api as om

# TODO: These will all have to be made jax-safe.
from openaerostruct.utils.vector_algebra import add_ones_axis
from openaerostruct.utils.vector_algebra import compute_dot, compute_dot_deriv
from openaerostruct.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2
from openaerostruct.utils.vector_algebra import compute_norm, compute_norm_deriv


tol = 1e-10


def _compute_finite_vortex(r1, r2):
    r1_norm = compute_norm(r1)
    r2_norm = compute_norm(r2)

    r1_x_r2 = compute_cross(r1, r2)
    r1_d_r2 = compute_dot(r1, r2)

    num = (1.0 / r1_norm + 1.0 / r2_norm) * r1_x_r2
    den = r1_norm * r2_norm + r1_d_r2

    result = np.divide(num, den * 4 * np.pi, out=np.zeros_like(num), where=np.abs(den) > tol)

    return result


def _compute_finite_vortex_deriv1(r1, r2, r1_deriv):
    r1_norm = add_ones_axis(compute_norm(r1))
    r2_norm = add_ones_axis(compute_norm(r2))
    r1_norm_deriv = compute_norm_deriv(r1, r1_deriv)

    r1_x_r2 = add_ones_axis(compute_cross(r1, r2))
    r1_d_r2 = add_ones_axis(compute_dot(r1, r2))
    r1_x_r2_deriv = compute_cross_deriv1(r1_deriv, r2)
    r1_d_r2_deriv = compute_dot_deriv(r2, r1_deriv)

    num = (1.0 / r1_norm + 1.0 / r2_norm) * r1_x_r2
    num_deriv = (-r1_norm_deriv / r1_norm ** 2) * r1_x_r2 + (1.0 / r1_norm + 1.0 / r2_norm) * r1_x_r2_deriv

    den = r1_norm * r2_norm + r1_d_r2
    den_deriv = r1_norm_deriv * r2_norm + r1_d_r2_deriv

    result = np.divide(
        num_deriv * den - num * den_deriv, den ** 2 * 4 * np.pi, out=np.zeros_like(num), where=np.abs(den) > tol
    )

    return result


def _compute_finite_vortex_deriv2(r1, r2, r2_deriv):
    r1_norm = add_ones_axis(compute_norm(r1))
    r2_norm = add_ones_axis(compute_norm(r2))
    r2_norm_deriv = compute_norm_deriv(r2, r2_deriv)

    r1_x_r2 = add_ones_axis(compute_cross(r1, r2))
    r1_d_r2 = add_ones_axis(compute_dot(r1, r2))
    r1_x_r2_deriv = compute_cross_deriv2(r1, r2_deriv)
    r1_d_r2_deriv = compute_dot_deriv(r1, r2_deriv)

    num = (1.0 / r1_norm + 1.0 / r2_norm) * r1_x_r2
    num_deriv = (-r2_norm_deriv / r2_norm ** 2) * r1_x_r2 + (1.0 / r1_norm + 1.0 / r2_norm) * r1_x_r2_deriv

    den = r1_norm * r2_norm + r1_d_r2
    den_deriv = r1_norm * r2_norm_deriv + r1_d_r2_deriv

    result = np.divide(
        num_deriv * den - num * den_deriv, den ** 2 * 4 * np.pi, out=np.zeros_like(num), where=np.abs(den) > tol
    )

    return result


def _compute_semi_infinite_vortex(u, r):
    r_norm = compute_norm(r)
    u_x_r = compute_cross(u, r)
    u_d_r = compute_dot(u, r)

    num = u_x_r
    den = r_norm * (r_norm - u_d_r)
    return num / den / 4 / np.pi


def _compute_semi_infinite_vortex_deriv(u, r, r_deriv):
    r_norm = add_ones_axis(compute_norm(r))
    r_norm_deriv = compute_norm_deriv(r, r_deriv)

    u_x_r = add_ones_axis(compute_cross(u, r))
    u_x_r_deriv = compute_cross_deriv2(u, r_deriv)

    u_d_r = add_ones_axis(compute_dot(u, r))
    u_d_r_deriv = compute_dot_deriv(u, r_deriv)

    num = u_x_r
    num_deriv = u_x_r_deriv

    den = r_norm * (r_norm - u_d_r)
    den_deriv = r_norm_deriv * (r_norm - u_d_r) + r_norm * (r_norm_deriv - u_d_r_deriv)

    return (num_deriv * den - num * den_deriv) / den ** 2 / 4 / np.pi


class AeroFunctionalVLM(om.ExplicitComponent):
    """
    Combines 5 components.

    Parameters
    ----------
    vortex_mesh[nx, ny, 3] : numpy array
        The actual aerodynamic mesh used in VLM calculations, where we look
        at the rings of the panels instead of the panels themselves. That is,
        this mesh coincides with the quarter-chord panel line, except for the
        final row, where it lines up with the trailing edge.
    eval_name[num_eval_points, 3] : numpy array
        These are the evaluation points, either collocation or force points.
    alpha : float
        The angle of attack for the aircraft (all lifting surfaces) in degrees.
    freestream_velocities[system_size, 3] : numpy array
        The rotated freestream velocities at each evaluation point for all
        lifting surfaces. system_size is the sum of the count of all panels
        for all lifting surfaces.
    normals[nx-1, ny-1, 3] : numpy array
        The normal vector for each panel, computed as the cross of the two
        diagonals from the mesh points.

    Returns
    -------
    mtx[system_size, system_size] : numpy array
        Final fully assembled AIC matrix that is used to solve for the
        circulations.
    rhs[system_size] : numpy array
        Right-hand side of the AIC linear system, constructed from the
        freestream velocities and panel normals.
    """

    def initialize(self):
        self.options.declare("surfaces", types=list)
        self.options.declare("num_eval_points", types=int)
        self.options.declare("eval_name", types=str)
        self.options.declare("use_jax", types=bool, default=True)

    def setup(self):
        surfaces = self.options["surfaces"]
        num_eval_points = self.options["num_eval_points"]
        eval_name = self.options["eval_name"]

        system_size = 0

        # Loop through the surfaces to compute the total number of panels;
        # the system_size
        for surface in surfaces:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            system_size += (nx - 1) * (ny - 1)

        self.system_size = system_size

        self.add_input(eval_name, val=np.zeros((num_eval_points, 3)), units="m")
        self.add_input("alpha", val=1.0, units="deg")
        self.add_input("freestream_velocities", shape=(system_size, 3), units="m/s")

        self.add_output("mtx", shape=(system_size, system_size), units="1/m")
        self.add_output("rhs", shape=system_size, units="m/s")

        for surface in surfaces:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            name = surface["name"]

            ground_effect = surface.get("groundplane", False)

            vectors_name = "{}_{}_vectors".format(name, eval_name)

            # This is where we handle the symmetry in the VLM method.
            # If it's symmetric, we need to effectively mirror the mesh by
            # accounting for the ghost mesh. We do this by using an artificially
            # larger mesh here.

            if surface["symmetry"]:
                actual_ny_size = ny * 2 - 1
            else:
                actual_ny_size = ny
            if ground_effect:
                actual_nx_size = nx * 2
            else:
                actual_nx_size = nx

            self.add_input(name + "_vortex_mesh", val=np.zeros((actual_nx_size, actual_ny_size, 3)), units="m")

            normals_name = "{}_normals".format(name)

            self.add_input(normals_name, shape=(nx - 1, ny - 1, 3))

        self.mtx_n_n_3 = np.zeros((system_size, system_size, 3))
        self.normals_n_3 = np.zeros((system_size, 3))

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        surfaces = self.options["surfaces"]
        num_eval_points = self.options["num_eval_points"]
        eval_name = self.options["eval_name"]

        system_size = self.system_size

        ind_1 = 0
        ind_2 = 0

        for surface in surfaces:
            nx = surface["mesh"].shape[0]
            ny = surface["mesh"].shape[1]
            name = surface["name"]

            mesh_name = name + "_vortex_mesh"
            ground_effect = surface.get("groundplane", False)

            mesh_reshaped = np.einsum("i,jkl->ijkl", np.ones(num_eval_points), inputs[mesh_name])
            if surface["symmetry"] and ground_effect:
                eval_points_reshaped = np.einsum(
                    "il,jk->ijkl",
                    inputs[eval_name],
                    np.ones((2 * nx, 2 * ny - 1)),
                )
            elif surface["symmetry"]:
                eval_points_reshaped = np.einsum(
                    "il,jk->ijkl",
                    inputs[eval_name],
                    np.ones((nx, 2 * ny - 1)),
                )
            else:
                eval_points_reshaped = np.einsum(
                    "il,jk->ijkl",
                    inputs[eval_name],
                    np.ones((nx, ny)),
                )

            # Actually subtract the vectors.
            vectors = eval_points_reshaped - mesh_reshaped

            alpha = inputs["alpha"][0]
            cosa = np.cos(alpha * np.pi / 180.0)
            sina = np.sin(alpha * np.pi / 180.0)

            if surface["symmetry"]:
                u = np.einsum("ijk,l->ijkl", np.ones((num_eval_points, 1, 2 * (ny - 1))), np.array([cosa, 0, sina]))
            else:
                u = np.einsum("ijk,l->ijkl", np.ones((num_eval_points, 1, ny - 1)), np.array([cosa, 0, sina]))

            vel_mtx = np.zeros((system_size, nx - 1, ny - 1, 3))

            # Here, we loop through each of the vectors and compute the AIC
            # terms from the four filaments that make up a ring around a single
            # panel. Thus, we are using vortex rings to construct the AIC
            # matrix. Later, we will convert these to horseshoe vortices
            # to compute the panel forces.

            if ground_effect:
                # mirrored surface along the x mesh direction
                surfaces_to_compute = [vectors[:, :nx, :, :], vectors[:, nx:, :, :]]
                vortex_mults = [1.0, -1.0]
            else:
                surfaces_to_compute = [vectors]
                vortex_mults = [1.0]

            for i_surf, surface_to_compute in enumerate(surfaces_to_compute):
                # vortex vertices:
                #         A ----- B
                #         |       |
                #         |       |
                #         D-------C
                #
                vortex_mult = vortex_mults[i_surf]
                vert_A = surface_to_compute[:, 0:-1, 1:, :]
                vert_B = surface_to_compute[:, 0:-1, 0:-1, :]
                vert_C = surface_to_compute[:, 1:, 0:-1, :]
                vert_D = surface_to_compute[:, 1:, 1:, :]
                # front vortex
                result1 = _compute_finite_vortex(vert_A, vert_B)
                # right vortex
                result2 = _compute_finite_vortex(vert_B, vert_C)
                # rear vortex
                result3 = _compute_finite_vortex(vert_C, vert_D)
                # left vortex
                result4 = _compute_finite_vortex(vert_D, vert_A)

                # If the surface is symmetric, mirror the results and add them
                # to the vel_mtx.
                if surface["symmetry"]:
                    result = vortex_mult * (result1 + result2 + result3 + result4)
                    vel_mtx += result[:, :, : ny - 1, :]
                    vel_mtx += result[:, :, ny - 1 :, :][:, :, ::-1, :]
                else:
                    vel_mtx += vortex_mult * (result1 + result2 + result3 + result4)

                # ----------------- last row -----------------

                vert_D_last = vert_D[:, -1:, :, :]
                vert_C_last = vert_C[:, -1:, :, :]
                result1 = _compute_finite_vortex(vert_D_last, vert_C_last)
                result2 = _compute_semi_infinite_vortex(u, vert_D_last)
                result3 = _compute_semi_infinite_vortex(u, vert_C_last)

                if surface["symmetry"]:
                    res1 = result1[:, :, : ny - 1, :]
                    res1 += result1[:, :, ny - 1 :, :][:, :, ::-1, :]
                    res2 = result2[:, :, : ny - 1, :]
                    res2 += result2[:, :, ny - 1 :, :][:, :, ::-1, :]
                    res3 = result3[:, :, : ny - 1, :]
                    res3 += result3[:, :, ny - 1 :, :][:, :, ::-1, :]
                    vel_mtx[:, -1:, :, :] += vortex_mult * (res1 - res2 + res3)
                else:
                    vel_mtx[:, -1:, :, :] += vortex_mult * result1
                    vel_mtx[:, -1:, :, :] -= vortex_mult * result2
                    vel_mtx[:, -1:, :, :] += vortex_mult * result3

            name = surface["name"]
            num = (nx - 1) * (ny - 1)

            ind_2 += num

            normals_name = "{}_normals".format(name)

            # Construct the full matrix and all of the lifting surfaces
            # together
            self.mtx_n_n_3[:, ind_1:ind_2, :] = vel_mtx.reshape((system_size, num, 3))
            self.normals_n_3[ind_1:ind_2, :] = inputs[normals_name].reshape((num, 3))

            ind_1 += num

        # Actually obtain the final matrix by multiplying through with the
        # normals. Also create the rhs based on v dot n.
        outputs["mtx"] = np.einsum("ijk,ik->ij", self.mtx_n_n_3, self.normals_n_3)
        outputs["rhs"] = -np.einsum("ij,ij->i", inputs["freestream_velocities"], self.normals_n_3)


