from src.agents.test import run_episode
from src.agents.train_test_utility_functions import load_config, load_data
from src.environments.environment_loader import EnvironmentLoader
from src.utils.evaluations import EvaluationHandler


def solve():

    # create evaluation handler
    evaluation_handler = EvaluationHandler()
    config_file_path="testing/dqn/Solve_FJSP_by_DRL.yaml"
    # get config and data
    config = load_config(config_file_path,None)
    data = load_data(config)
    # create env
    environment, _ = EnvironmentLoader.load(config, data=[data[0]])
    environment.runs = 0


    # run environment episode
    run_episode(environment, model=None,heuristic_id='LTR', handler=evaluation_handler)

    image=environment.render()
    image.show()


if __name__ == '__main__':
    solve()