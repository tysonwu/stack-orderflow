import argparse
import json
import os
from multiprocessing import Process

from data_pipeline.volume_data_recorder import VolumeDataRecorder


def main_ws(env, config):
    ws = VolumeDataRecorder(env, config)
    ws.run()


if __name__ == '__main__':
    '''
    Real time streaming of aggTrade data. Data are transformed and written into influx directly.
    '''
    # read argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", dest="env", help="system environment", metavar="environment", default="local")
    args = parser.parse_args()
    if args.env == 'local':
        env = 'local'
    elif args.env == 'server':
        env = 'prod'
    
    # read config
    with open(f'{os.path.dirname(__file__)}/../../../config/volume_data_recorder_config.json') as f:
        configs = json.load(f)['Config'] # list of configs for each thread


    process_list = []
    for config in configs:
        volume_recorder_p = Process(target=main_ws, args=(env, config,), daemon=True)
        process_list.append(volume_recorder_p)
        volume_recorder_p.start()

    for p in process_list:
        p.join()