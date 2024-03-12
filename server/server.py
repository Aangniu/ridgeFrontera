import hashlib
import jinja2

import time

import misfits_kath as msf
import scipy.interpolate as sp_int

import numpy as np
import os
import subprocess
import sys
import umbridge

import pandas as pd

def gpu_available():
    # Hard coded for now, until I find a better way to automatically check,
    # whether a GPU is available.
    # return True
    return False


def seissol_command(run_id="", ranks=4):
    if gpu_available():
        return f"not supported yet for nonlinear seissol"
    else:
        return f"mpirun -n {ranks} -machinefile $HQ_NODE_FILE /scratch1/09840/zniu2025/seisRidge/SeisSol/build_app/SeisSol_Release_dskx_4_viscoelastic2 {run_id}/par
ameters_40s.par"
        #return f"mpiexec.hydra -n {ranks} -machinefile $HQ_NODE_FILE apptainer run ../seissol.sif SeisSol_Release_dskx_4_viscoelastic2 {run_id}/parameters_40s.par"


class SeisSol(umbridge.Model):
    def __init__(self, ranks):
        self.name = "SeisSol"
        # self._sleep_time = sleep_time
        self.ranks = ranks
        super().__init__("forward")

    def get_input_sizes(self, config):
        return [1]

    def get_output_sizes(self, config):
        return [30]

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
        mat_template = environment.get_template("Ridgecrest_material_cvms1000m_template.yaml")
        mat_content = mat_template.render(
            plastScale=parameters[0][0]
        )
        with open(os.path.join(run_id, "Ridgecrest_material_cvms1000m.yaml"), "w+") as mat_file:
            mat_file.write(mat_content)

        parameter_template = environment.get_template("parameters_40s_template.par")
        parameter_content = parameter_template.render(
            output_dir=run_id,
            mesh_file=config["meshFile"]
        )
        with open(os.path.join(run_id, "parameters_40s.par"), "w+") as parameter_file:
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
        my_env["I_MPI_SHM_HEAP_VSIZE"] = "65536"
        return my_env

    def __call__(self, parameters, config):
        if not config["meshFile"]:
            config["meshFile"] = "Ridgecrest_NewModel1_f200_topo1000_noRef_xml_UBC"
        run_id = self.prepare_filesystem(parameters, config)

        # time.sleep(self._sleep_time)

        command = seissol_command(run_id, self.ranks)
        print(command)
        my_env = self.prepare_env()
        sys.stdout.flush()
        print("reached the part for launching seissol....")
        subprocess.run("cat $HQ_NODE_FILE", shell=True)
        result = subprocess.run(command, shell=True, env=my_env)
        print("passed the part for launching seissol....")
        result.check_returncode()

        #=================MomentRate====================
        # Postprocessing to get moment rate
        data = pd.read_csv(os.path.join(run_id, "ridgecrest-energy.csv"),sep=',')

        idMoment, = np.where(data["variable"] == "seismic_moment")

        time = data["time"][idMoment]
        dt = time.iloc[1] - time.iloc[0]
        print("dt = ",dt)
        moment_d = data["measurement"][idMoment]
        mrate_data = np.gradient(moment_d,dt)

        times_interp = np.linspace(0.001,39.999,80)

        interpolator = sp_int.interp1d(time.values, mrate_data)

        mD_interpolated = interpolator(times_interp)

        ## Reference
        ref = np.load(os.path.join("ref", "MT1_Moment_rate_array.npy"))

        interpolatorRef = sp_int.interp1d(ref[:,0], ref[:,1])

        mR_interpolated = interpolatorRef(times_interp)

        print(mD_interpolated - mR_interpolated)

        m_diff = mD_interpolated - mR_interpolated

        #=================GPS====================
        prefix = 'ridgecrest'
        directory = run_id

        ref_GPS = np.load(os.path.join("ref", "dataGPSForCompare.npy"))

        rec_comps = ['v1','v2','v3']

        nRec = 10

        diffNorms = []
        for i_s in range(nRec):
            for i_c in range(3):
                sim_result = msf.read_receiver(
                    msf.find_receiver(directory,prefix,i_s+1))
                dt = sim_result['Time'][1] - sim_result['Time'][0]
                displace = np.cumsum(sim_result[ rec_comps[i_c] ])*dt

                dispRef = ref_GPS[i_c][i_s][0:21]

                # Interpoloate simulation results to compare with Ref
                times_interp = np.linspace(0.001,19.999,21)
                interpolatorGPS = sp_int.interp1d(sim_result['Time'], displace)

                dispSim_interpolated = interpolatorGPS(times_interp)

                diffGPS = dispSim_interpolated - dispRef
                diffNorms.append(np.linalg.norm(diffGPS))

        # return [m_diff.tolist()]
        return [diffNorms]

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
