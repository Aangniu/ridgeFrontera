import argparse
import multiprocessing as mp
import time
import umbridge
import numpy as np

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("port", help="port", type=int)
        args = parser.parse_args()

        address = f"http://localhost:{args.port}"
        server_available = False
        while not server_available:
                try:
                        model = umbridge.HTTPModel(address, "forward")
                        print("Server available",address)
                        server_available = True
                except:
                        print("Server not available")
                        time.sleep(10)

        def eval_um_model(Cd):
                config = {"meshFile": "Ridgecrest_NewModel1_f200_topo1000_noRef_xml_UBC", "order": 4}
                return model([[Cd]], config)

        start_time = time.time()
        inp_np = np.linspace(0.5,3.5,31)
        inp_list = inp_np.tolist()
        with mp.Pool(1) as p:
                result = p.map(eval_um_model, [0.8])
                print(result)
        end_time = time.time()
        print(end_time - start_time)

