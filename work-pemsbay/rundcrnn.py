from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
from load_data_dcrnn import *
from dcrnn import *
import os


# os.system("python load_data_dcrnn.py")


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        supervisor.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default="/home/cseadmin/yindu/github/data/dcrnn-data/model/dcrnn_bay.yaml", type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    print(args)
    main(args)
