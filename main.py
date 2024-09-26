import os
import argparse
import torch
from runner.runner_creator import runner_factory
from configs import Unet3dConfig
from configs import Unet3dConfig_eval
from configs import Unet3dConfig_test
from configs import RepLKUnet3dConfig
from configs import RepLKUnet3dConfig_eval
from configs import RepLKUnet3dConfig_test
from configs import Unet3dDeepConfig
from configs import Unet3dDeepConfig_eval
from configs import Unet3dDeepConfig_test

from configs import SALKUnetConfig_D50
from configs import SALKUnetConfig_D50_test
from configs import SALKUnetConfig_D50_eval
from configs import SALKUnetConfig_D20
from configs import SALKUnetConfig_D20_test
from configs import SALKUnetConfig_D20_eval
from configs import SALKUnetConfig_D10
from configs import SALKUnetConfig_D10_test
from configs import SALKUnetConfig_D10_eval
from configs import SALKUnetConfig_D4
from configs import SALKUnetConfig_D4_test
from configs import SALKUnetConfig_D4_eval
from configs import SALKUnetConfig_D100
from configs import SALKUnetConfig_D100_eval
from configs import SALKUnetConfig_D100_test
from configs import SALKUnetConfig_RLD_test



def main(args, config):
    if not os.path.exists(args.workdir):
        os.makedirs(args.workdir)
    runner = runner_factory(config, args.workdir)
    if config.mode.mode == "train":
        runner.train()
    elif config.mode.mode == "eval":
        with torch.no_grad():
            runner.eval(args.eval_dir, args.eval_path)
    elif config.mode.mode == "test":
        with torch.no_grad():
            runner.test(args.test_dir, args.test_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--configs_name", type=str, default="SALKUnetConfig_D4_eval")
    
    parser.add_argument("--workdir", type=str, default="/home/uPET/FinalCode/SALKUnet_asset/_exp_FinD4")
    parser.add_argument("--eval_dir", type=str, default="/home/uPET/FinalCode/SALKUnet_asset/_exp_FinD4_eval")
    parser.add_argument("--eval_path", type=str, default="/home/uPET/FinalCode/SALKUnet_asset/_exp_FinD4/checkpoints/checkpoint_24.pth")
    parser.add_argument("--test_dir", type=str, default="/home/uPET/FinalCode/SALKUnet_asset/_exp_FinRLD_test")
    parser.add_argument("--test_path", type=str, default="/home/uPET/FinalCode/SALKUnet_asset/_exp_FinD100/checkpoints/checkpoint_24.pth")

    args = parser.parse_args()

    if args.configs_name == "Unet3dConfig":
        config = Unet3dConfig.getConfigs()
    elif args.configs_name == "Unet3dConfig_eval":
        config = Unet3dConfig_eval.getConfigs()
    elif args.configs_name == "Unet3dConfig_test":
        config = Unet3dConfig_test.getConfigs()
    elif args.configs_name == "RepLKUnet3dConfig":
        config = RepLKUnet3dConfig.getConfigs()
    elif args.configs_name == "RepLKUnet3dConfig_eval":
        config = RepLKUnet3dConfig_eval.getConfigs()
    elif args.configs_name == "RepLKUnet3dConfig_test":
        config = RepLKUnet3dConfig_test.getConfigs()
    elif args.configs_name == "Unet3dDeepConfig":
        config = Unet3dDeepConfig.getConfigs()
    elif args.configs_name == "Unet3dDeepConfig_eval":
        config = Unet3dDeepConfig_eval.getConfigs()
    elif args.configs_name == "Unet3dDeepConfig_test":
        config = Unet3dDeepConfig_test.getConfigs()

    elif args.configs_name == "SALKUnetConfig_D100":
        config = SALKUnetConfig_D100.getConfigs()
    elif args.configs_name == "SALKUnetConfig_D100_eval":
        config = SALKUnetConfig_D100_eval.getConfigs()
    elif args.configs_name == "SALKUnetConfig_D100_test":
        config = SALKUnetConfig_D100_test.getConfigs()
    elif args.configs_name == "SALKUnetConfig_D50":
        config = SALKUnetConfig_D50.getConfigs()
    elif args.configs_name == "SALKUnetConfig_D50_test":
        config = SALKUnetConfig_D50_test.getConfigs()
    elif args.configs_name == "SALKUnetConfig_D50_eval":
        config = SALKUnetConfig_D50_eval.getConfigs()
    elif args.configs_name == "SALKUnetConfig_D20":
        config = SALKUnetConfig_D20.getConfigs()
    elif args.configs_name == "SALKUnetConfig_D20_test":
        config = SALKUnetConfig_D20_test.getConfigs()
    elif args.configs_name == "SALKUnetConfig_D20_eval":
        config = SALKUnetConfig_D50_eval.getConfigs()
    elif args.configs_name == "SALKUnetConfig_D10":
        config = SALKUnetConfig_D10.getConfigs()
    elif args.configs_name == "SALKUnetConfig_D10_test":
        config = SALKUnetConfig_D10_test.getConfigs()
    elif args.configs_name == "SALKUnetConfig_D10_eval":
        config = SALKUnetConfig_D10_eval.getConfigs()
    elif args.configs_name == "SALKUnetConfig_D4":
        config = SALKUnetConfig_D4.getConfigs()
    elif args.configs_name == "SALKUnetConfig_D4_test":
        config = SALKUnetConfig_D4_test.getConfigs()
    elif args.configs_name == "SALKUnetConfig_D4_eval":
        config = SALKUnetConfig_D4_eval.getConfigs()
    elif args.configs_name == "SALKUnetConfig_RLD_test":
        config = SALKUnetConfig_RLD_test.getConfigs()

    
    else:
        raise f"The config name {args.config_name} not exist!"

    main(args, config)