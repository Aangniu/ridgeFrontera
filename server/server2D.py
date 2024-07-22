import hashlib
import jinja2

import time
from datetime import datetime

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


def seissol_command(run_id="", ranks=4, order=4):
    if gpu_available():
        return f"not supported yet for nonlinear seissol"
    else:
        #return f"mpiexec.hydra -n {ranks} -machinefile $HQ_NODE_FILE apptainer run ../seissol_auto.sif SeisSol_Release_dskx_4_viscoelastic2 {run_id}/parameters_40s.par"
        if order==4:
            return f"mpiexec.hydra -n {ranks} -machinefile $MACHINE_FILE /work2/09840/zniu2025/frontera/seisRidge/SeisSol/build_app/SeisSol_Release_dskx_4_viscoelastic2 {run_id}/parameters_40s.par"
        elif order==3:
            return f"mpiexec.hydra -n {ranks} -machinefile $MACHINE_FILE /work2/09840/zniu2025/frontera/seisRidge/SeisSol/build_O3app/SeisSol_Release_dskx_3_viscoelastic2 {run_id}/parameters_40s.par"

class SeisSol(umbridge.Model):
    def __init__(self, ranks):
        self.name = "SeisSol"
        # self._sleep_time = sleep_time
        self.ranks = ranks
        super().__init__("forward")

    def get_input_sizes(self, config):
        return [2]

    def get_output_sizes(self, config):
        return [1]

    def prepare_filesystem(self, parameters, config):
        # submission_time = time.ctime(time.time())
        submission_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        param_conf_string = str((parameters, config,submission_time)).encode("utf-8")
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
            plastScale=parameters[0][0],
            bulkFric=parameters[0][1]
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
        if not config["order"]:
            config["order"] = 4
        run_id = self.prepare_filesystem(parameters, config)

        # time.sleep(self._sleep_time)

        command = seissol_command(run_id, self.ranks, config["order"])
        print(command)
        #print("reached the part for launching seissol....")
        my_env = self.prepare_env()
        #sys.stdout.flush()
        print("reached the part for launching seissol....")
        sys.stdout.flush()
        subprocess.run("cat $MACHINE_FILE", shell=True)
        result = subprocess.run(command, shell=True, env=my_env)
        print("passed the part for launching seissol....")
        #result.check_returncode()
        print('args:', result.args)
        print('returncode:', result.returncode)
        sys.stdout.flush()

        # if result.returncode > 0:
        #   print(f"SeisSol failed with input: {parameters[0]}, return an extremely small log-likelihood (-1e4) to the client!")
        #   sys.stdout.flush()
        #   return [[-1e4]]

        errorCount = 0
        while result.returncode > 0:
          print(f"SeisSol failed with input: {parameters[0]}, re-run with the same parameters for {errorCount+1} times...")
          sys.stdout.flush()
          result = subprocess.run(command, shell=True, env=my_env)
          print('args:', result.args)
          print('returncode:', result.returncode)
          sys.stdout.flush()
          errorCount += 1
          if errorCount >= 5:
            print(f"SeisSol failed with input: {parameters[0]} for {errorCount} times, return an extremely small log-likelihood (-5e2) to the client!")
            sys.stdout.flush()
            return [[-5e2]]
          
        # Postprocessing to get moment rate

        data = pd.read_csv(os.path.join(run_id, "ridgecrest-energy.csv"),sep=',')

        idMoment, = np.where(data["variable"] == "seismic_moment")

        time = data["time"][idMoment]
        dt = time.iloc[1] - time.iloc[0]
        print("dt = ",dt)
        moment_d = data["measurement"][idMoment]
        mrate_data = np.gradient(moment_d,dt)

        times_interp = np.linspace(0.001,19.999,80)

        interpolator = sp_int.interp1d(time.values, mrate_data)

        mD_interpolated = interpolator(times_interp)

        ## Reference

        ref = np.load(os.path.join("ref", "MT1_Moment_rate_array.npy"))

        interpolatorRef = sp_int.interp1d(ref[:,0], ref[:,1])

        mR_interpolated = interpolatorRef(times_interp)

        #print(mR_interpolated,mD_interpolated)

        print(mD_interpolated - mR_interpolated)

        m_diff = mD_interpolated - mR_interpolated
        
        #=================GPS====================
        prefix = 'ridgecrest'
        directory = run_id

        ref_GPS = np.load(os.path.join("ref", "dataGPSForCompare.npy"))

        rec_comps = ['v1','v2','v3']
        
        #=================covariance=========
        cov_diag = 1e-3*np.ones(30)
        cov_diag[0] = 6e-3
        cov_diag[2] = 3e-3
        cov_diag[3] = 7e-3
        cov_diag[10] = 3e-3
        cov_diag[13] = 3e-3
        cov_diag[20] = 2e-3
        cov_diag[22] = 3e-3
        cov_diag[23] = 3e-3
        cov_diag[24] = 2e-3
        cov_diag[25] = 2e-3
        cov_diag[26] = 5e-3
        cov_diag[27] = 4e-3
        #==================================
        nRec = 10

        diffNorms = []
        log_likelihood = 0.0
        y_outSquared = 0.0
        sig_y = 1.0/np.sqrt(2.0)
        for i_c in range(3):
            for i_s in range(nRec):
                sim_result = msf.read_receiver(
                    msf.find_receiver(directory,prefix,i_s+1))
                dt = sim_result['Time'][1] - sim_result['Time'][0]
                displace = np.cumsum(sim_result[ rec_comps[i_c] ])*dt

                dispRef = ref_GPS[i_c][i_s][0:21]

                # Interpoloate simulation results to compare with Ref
                # print('Sim time series: ', sim_result['Time'])
                times_interp = np.linspace(0.001,19.99,21)
                interpolatorGPS = sp_int.interp1d(sim_result['Time'], displace)

                dispSim_interpolated = interpolatorGPS(times_interp)

                diffGPS = dispSim_interpolated - dispRef
                diffNorms.append(np.linalg.norm(diffGPS)**2)
                y_outSquared += np.linalg.norm(diffGPS)**2/cov_diag[i_c*nRec+i_s]/3/nRec
                #log_likelihood += np.linalg.norm(diffGPS)**2/cov_diag[i_c*nRec+i_s]
                print(y_outSquared)
        print(diffNorms)
        log_likelihood = 0.5*y_outSquared/sig_y**2
        return [[-log_likelihood]]
        #return [diffNorms]
        #return [m_diff.tolist()]

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
