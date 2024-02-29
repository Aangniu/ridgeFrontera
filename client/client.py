import argparse
import multiprocessing as mp
import time
import umbridge

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
                config = {"meshFile": "Ridgecrest_NewModel1_f200_topo1000_noRef_xml_UBC"}
                return model([[Cd]], config)

        start_time = time.time()
        with mp.Pool(10) as p:
                result = p.map(eval_um_model, [1,0.5,10])
                print(result)
        end_time = time.time()
        print(end_time - start_time)

