import os
import yaml

CONFIG_FILE = "config.yml"
with open(CONFIG_FILE, "r") as fp:
    yml_config = yaml.load(fp, Loader=yaml.FullLoader)

quantiles = yml_config["model_params"]["quantiles"]

best_model_path = yml_config.get("best_model")
prod_model_path = None
if best_model_path:
    prod_model_path = os.path.join("trained_models", best_model_path)
