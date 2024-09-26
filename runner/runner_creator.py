from runner.runner_e2e import Runner_e2e

def runner_factory(config, workdir):
    runner_config = config.runner
    if runner_config.type == "Runner_e2e":
        runner = Runner_e2e(config, workdir=workdir)
    else:
        raise f"The runner type {runner_config.type} not exist!"
    return runner