#!/usr/bin/env python3

from dask.distributed import LocalCluster
from os import cpu_count, listdir, uname
from time import sleep
import json

if __name__ == '__main__':

    print(uname().nodename)
    for file in listdir():
        if file.endswith('.json') and file.startswith('config-'):
            if file.replace('.json', '').replace('config-', '') in uname().nodename.replace('-', ''):
                my_conf = json.load(open(file))
                break

    cluster = LocalCluster(**my_conf)

    print(f'Cluster started!\nDashboard: {cluster.dashboard_link}\nScheduler Address: {cluster.scheduler_comm.address}')
    for i, a_worker in enumerate(cluster.workers.values()):
        print(f'  Worker {i}:')
        print(f'    Threads: {a_worker.nthreads}')
        print(f'    Address: {a_worker.address_safe}')
        print(f'    Memory: {a_worker.memory_manager.memory_limit/(1024**3)} GiB')




    try:
        while True:
            sleep(86400)
    except KeyboardInterrupt:
        cluster.close()