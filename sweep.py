import wandb
import yaml

wandb.login()

with open("config_test.yaml") as file:
    sweep_configuration = yaml.safe_load(file)


# 1: Define objective/training function
def objective(config):
    score = config.x**3 + config.y
    return score


def main():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})


# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
