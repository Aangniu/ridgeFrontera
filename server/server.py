import hashlib
import jinja2

import time

import misfits_kath
import scipy.interpolate as sp_int

import numpy as np
import os
import subprocess
import sys
import umbridge


def gpu_available():
    # Hard coded for now, until I find a better way to automatically check,
    # whether a GPU is available.
    # return True
    return False


def seissol_command(run_id="", ranks=4):
    if gpu_available():
        return f"not supported yet for nonlinear seissol"
    else:
        return f"mpiexec.hydra -n {ranks} -machinefile $HQ_NODE_FILE apptainer run ../seissol.sif SeisSol_Release_dskx_3_damaged-elastic {run_id}/parameters.par"


class SeisSol(umbridge.Model):
    def __init__(self, ranks):
        self.name = "SeisSol"
        # self._sleep_time = sleep_time
        self.ranks = ranks
        super().__init__("forward")

    def get_input_sizes(self, config):
        return [4]

    def get_output_sizes(self, config):
        return [100]

    def prepare_filesystem(self, parameters, config):
        param_conf_string = str((parameters, config)).encode("utf-8")
        print(param_conf_string)

        m = hashlib.md5()
        m.update(param_conf_string)
        h = m.hexdigest()
        run_id = f"simulation_{h}"
        print(run_id)

        subprocess.run(["rm", "-rf", run_id])
        subprocess.run(["mkdir", run_id])
        environment = jinja2.Environment(loader=jinja2.FileSystemLoader("."))
        mat_template = environment.get_template("Nepal_material_template.yaml")
        mat_content = mat_template.render(
            par1_gammaR=parameters[0][0],
            par2_Cd=parameters[0][1],
            par3_mu0=parameters[0][2],
            par4_lamb0=parameters[0][3]
        )
        with open(os.path.join(run_id, "Nepal_material_chain.yaml"), "w+") as mat_file:
            mat_file.write(mat_content)

        parameter_template = environment.get_template("parameters_template.par")
        parameter_content = parameter_template.render(
            output_dir=run_id,
            mesh_file=config["meshFile"]
        )
        with open(os.path.join(run_id, "parameters.par"), "w+") as parameter_file:
            parameter_file.write(parameter_content)

        return run_id

    def prepare_env(self):
        my_env = os.environ.copy()
        my_env["MV2_ENABLE_AFFINITY"] = "0"
        my_env["MV2_HOMOGENEOUS_CLUSTER"] = "1"
        my_env["MV2_SMP_USE_CMA"] = "0"
        my_env["MV2_USE_AFFINITY"] = "0"
        my_env["MV2_USE_ALIGNED_ALLOC"] = "1"
        my_env["TACC_AFFINITY_ENABLED"] = "1"
        my_env["OMP_NUM_THREADS"] = "54"
        my_env["OMP_PLACES"] = "cores(54)"
        return my_env

    def __call__(self, parameters, config):
        if not config["meshFile"]:
            config["meshFile"] = "model_0p1Hz"
        run_id = self.prepare_filesystem(parameters, config)

        # time.sleep(self._sleep_time)

        command = seissol_command(run_id, self.ranks)
        print(command)
        my_env = self.prepare_env()
        sys.stdout.flush()
        subprocess.run("cat $HQ_NODE_FILE", shell=True)
        print("reached the part for launching seissol....")
        result = subprocess.run(command, shell=True, env=my_env)
        print("passed the part for launching seissol....")
        result.check_returncode()

        n_recs = [1,2,7,16,18,19,20,21,22,23,24,44,48,49,50,51,57,59,61,64]

        m_alphas = [misfits_kath.misfit(run_id, "output", "M7.8FL34_0.1Hztopo5km_o3ga5e2c1e1_o3jan18_50s", i) for i in n_recs]

        n_timeSeries = 74

        m_timeSeries = misfits_kath.read_receiver(\
            misfits_kath.find_receiver(run_id, "M7.8FL34_0.1Hztopo5km_o3ga5e2c1e1_o3jan18_50s", n_timeSeries))

        times = np.linspace(6.0,49.999,80)

        interpolator = sp_int.interp1d(m_timeSeries["Time"], m_timeSeries['v3'])

        v_interpolated = interpolator(times)

        print(np.append(m_alphas,v_interpolated))
        return np.append(m_alphas,v_interpolated)

    def supports_evaluate(self):
        return True


if __name__ == "__main__":
    port = int(os.environ["PORT"])
    ranks = int(os.environ["RANKS"])
    print(f"Running SeisSol server with {ranks} MPI ranks on port {port}.")

    umbridge.serve_models(
        [
            SeisSol(ranks = ranks),
            # SeisSol(ranks = ranks, model_name="parameter_to_observable_map_intermediate", sleep_time=0.6),
            # SeisSol(ranks = ranks, model_name="parameter_to_observable_map_coarse", sleep_time=0.3),
        ],
        port=port,
        max_workers=100,
    )
