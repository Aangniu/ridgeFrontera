# ridgeFrontera
model server on frontera for Umbridge

# How to use it?
1 Download the [input files](https://syncandshare.lrz.de/getlink/fiXv8QwkPAGPANcEXuAr4R/ridgecrest2D.zip) for execution of the Ridgecrest simulation with [SeisSol](https://github.com/SeisSol/SeisSol).

2 Rename it as "ridgecrest2D" and add place it to the home directory of this repository.

3 Pull and compile a [SeisSol](https://github.com/SeisSol/SeisSol) executable to your target cluster. The documentation for how to compile can be found [here](https://seissol.readthedocs.io/en/latest/). Save the executable to anywhere and remember the path `path/to/seissol_executable`.

3 The python script in `server/server2D.py` provides the setup for a server that expose an HTTP protocol to the [Umbridge](https://um-bridge-benchmarks.readthedocs.io/en/docs/) load balancer. Replace the `path/to/seissol_executable` to the `seissol_command()` function in this python script. This server will then take care of managing the file system for postprocessing output and send the information to Umridge.

4 Ready to go :)
