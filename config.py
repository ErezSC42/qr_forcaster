import os
import yaml

CONFIG_FILE = "config.yml"
with open(CONFIG_FILE, "r") as fp:
    yml_config = yaml.load(fp, Loader=yaml.FullLoader)

quantiles = yml_config["model_params"]["quantiles"]
prod_model_path = os.path.join("trained_models", yml_config["best_model"])