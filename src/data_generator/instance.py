import random
import warnings
from typing import List

import numpy as np
from tqdm import tqdm

from src.data_generator.instance_factory import generate_deadlines
from src.data_generator.sp_factory import SPFactory
from src.data_generator.task import Task
from src.utils.file_handler.config_handler import ConfigHandler
from src.utils.file_handler.data_handler import DataHandler

Processing_time=[[[5, 9999, 4, 9999, 9999, 9999],
                  [9999, 1, 5, 9999, 3, 9999],
                  [9999, 9999, 4, 9999, 9999, 2],
                  [1, 6, 9999, 9999, 9999, 5],
                  [9999, 9999, 1, 9999, 9999, 9999],
                  [9999, 9999, 6, 3, 9999, 6]],

                 [[9999, 6, 9999, 9999, 9999, 9999],
                  [9999, 9999, 1, 9999, 9999, 9999],
                  [2, 9999, 9999, 9999, 9999, 9999],
                  [9999, 6, 9999, 6, 9999, 9999],
                  [1, 6, 9999, 9999, 9999, 5]],

                 [[9999, 6, 9999, 9999, 9999, 9999],
                  [9999, 9999, 4, 9999, 9999, 2],
                  [1, 6, 9999, 9999, 9999, 5],
                  [9999, 6, 4, 9999, 9999, 6],
                  [1, 9999, 9999, 9999, 5, 9999]],

                 [[1, 6, 9999, 9999, 9999, 5],
                  [9999, 6, 9999, 9999, 9999, 9999],
                  [9999, 9999, 1, 9999, 9999, 9999],
                  [9999, 1, 5, 9999, 3, 9999],
                  [9999, 9999, 4, 9999, 9999, 2]],

                 [[9999, 1, 5, 9999, 3, 9999],
                  [1, 6, 9999, 9999, 9999, 5],
                  [9999, 6, 9999, 9999, 9999, 9999],
                  [5, 9999, 4, 9999, 9999, 9999],
                  [9999, 6, 9999, 6, 9999, 9999],
                  [9999, 6, 4, 9999, 9999, 6]],

                 [[9999, 9999, 4, 9999, 9999, 2],
                  [2, 9999, 9999, 9999, 9999, 9999],
                  [9999, 6, 4, 9999, 9999, 6],
                  [9999, 6, 9999, 9999, 9999, 9999],
                  [1, 6, 9999, 9999, 9999, 5],
                  [3, 9999, 9999, 2, 9999, 9999]],

                 [[9999, 9999, 9999, 9999, 9999, 1],
                  [3, 9999, 9999, 2, 9999, 9999],
                  [9999, 6, 4, 9999, 9999, 6],
                  [6, 6, 9999, 9999, 1, 9999],
                  [9999, 9999, 1, 9999, 9999, 9999]],

                 [[9999, 9999, 4, 9999, 9999, 2],
                  [9999, 6, 4, 9999, 9999, 6],
                  [1, 6, 9999, 9999, 9999, 5],
                  [9999, 6, 9999, 9999, 9999, 9999],
                  [9999, 6, 9999, 6, 9999, 9999]],

                 [[9999, 9999, 9999, 9999, 9999, 1],
                  [1, 9999, 9999, 9999, 5, 9999],
                  [9999, 9999, 6, 3, 9999, 6],
                  [2, 9999, 9999, 9999, 9999, 9999],
                  [9999, 6, 4, 9999, 9999, 6],
                  [9999, 6, 9999, 6, 9999, 9999]],

                 [[9999, 9999, 4, 9999, 9999, 2],
                  [9999, 6, 4, 9999, 9999, 6],
                  [9999, 1, 5, 9999, 3, 9999],
                  [9999, 9999, 9999, 9999, 9999, 1],
                  [9999, 6, 9999, 6, 9999, 9999],
                  [3, 9999, 9999, 2, 9999, 9999]]]

num_jobs=10
num_tasks=6
num_machines=6
num_instances=10000

from multiprocessing import Process, Manager
def get_instance():
    instance=[]
    for j in range(num_jobs):
        # Generate random shuffled list of machines for job tasks
        #machines_jssp_random = random.sample(list(np.arange(num_tasks)), num_tasks)
        # pick num_tasks tasks for this job
        t=0
        for x in Processing_time[j]:
            meachine=[0 if y==9999 else 1 for y in x ]
            runtimelists=[0 if y==9999 else y for y in x ]
            runtimechoice=[x for x in runtimelists if x!=0]
            task = Task(
                job_index=j,
                task_index=t,
                machines=meachine,
                tools=[],
                deadline=0,
                done=False,
                runtime=random.choice(runtimechoice),
                runtimes=runtimelists,
                _n_machines=num_machines,
                _n_tools=0
            )
            t+=1
            instance.append(task)
        if len(Processing_time[j])!=num_tasks:
            for k in range(num_tasks-len(Processing_time[j])):
                meachine=[1 for _ in range(num_machines) ]
                runtimelists=[0 for _ in range(num_machines) ]
                task = Task(
                    job_index=j,
                    task_index=t,
                    machines=meachine,
                    tools=[],
                    deadline=0,
                    done=False,
                    runtime=0,
                    runtimes=runtimelists,
                    _n_machines=num_machines,
                    _n_tools=0
                )
                t+=1
                instance.append(task)
    return instance


def get_instances()-> List[List[Task]]:
    instances = []
    # Create and collect n instances
    for _ in range(num_instances):
        # Call instance function with currently collected arguments
        new_instance = get_instance()
        instances.append(new_instance)
    return instances



def compute_initial_instance_solution(instances: List[List[Task]],config: dict) -> List[List[Task]]:
    """
    Initializes multiple processes (optional) to generate deadlines for the raw scheduling problem instances

    :param instances: List of raw scheduling problem instances
    :param config: Data_generation config

    :return: List of scheduling problems instances with set deadlines

    """
    # Get configured number of processes
    num_processes: int = config.get('num_processes', 1)
    #num_processes: int = 16
    if num_processes > len(instances):
        num_processes = len(instances)
        warnings.warn('num_processes was set to num_instances.'
                      'The number of processes may not exceed the number of instances which need to be generated.',
                      category=RuntimeWarning)

    # Multiprocess case
    manager = Manager()
    instance_list = manager.list()
    make_span_list = manager.list()
    processes = []

    # split instances for multiprocessing
    features_dataset = np.array_split(instances, num_processes)

    for process_id in tqdm(range(num_processes), desc="Compute deadlines"):
        args = (features_dataset[process_id], instance_list, make_span_list, config)
        p = Process(target=generate_deadlines, args=args)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    return list(instance_list)


if __name__=="__main__":
    config_file_name="data_generation/fjssp/Solve_FJSP_by_DRL.yaml"
    current_config: dict = ConfigHandler.get_config(config_file_name,None)
    generated_instances: List[List[Task]]=get_instances()
    instance_list: List[List[Task]] =compute_initial_instance_solution(generated_instances,current_config)
    SPFactory.set_deadlines_to_max_deadline_per_job(instance_list, num_jobs)
    SPFactory.compute_and_set_hashes(instance_list)
    # Write resulting instance data to file
    if current_config.get('write_to_file', False):
        DataHandler.save_instances_data_file(current_config, instance_list)
    pass

















