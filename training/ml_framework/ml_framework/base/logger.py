import os
from datetime import datetime
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
import mlflow
from ml_framework.base.schema import Config
from pathlib import Path 
import yaml 
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

def save_config(config: BaseModel, config_path: str | Path):
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="UTF-8") as f:
        yaml.dump(config.model_dump(), f)


class LoggerFactory:
    @staticmethod
    def create(config: Config):
        if config.logger.name == "mlflow":
            return LoggerFactory._mlflow(config)
        elif config.logger.name == "wandb":
            return LoggerFactory._wandb(config)
        else:
            raise ValueError(f"Unknown logger name: {config.logger.name}")

    @staticmethod
    def _mlflow(config: Config):
        project_name = config.project_name
        experiment_name = config.experiment_name
        run_name = experiment_name+"_"+datetime.now().strftime("%Y%m%d-%H%M%S")
        tracking_uri = f"file:{str(os.path.join(config.directory.log,'mlflow'))}"
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(project_name)

        if experiment is None:
            experiment_id = mlflow.create_experiment(name=project_name)
            experiment = mlflow.get_experiment(experiment_id)

        run = mlflow.start_run(
            run_name=run_name,
            experiment_id=experiment.experiment_id,
            log_system_metrics=True
        )
        logger= MLFlowLogger(
            experiment_name=project_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            run_id = run.info.run_uuid
        )
        save_config(config, Path(config.directory.log).joinpath(
            config.logger.name,
            logger.experiment_id,
            logger.run_id,
            "config.yaml"
        ))
        return logger
    
    @staticmethod
    def _wandb(config: Config):
        project_name = config.project_name
        experiment_name = config.experiment_name
        run_name = experiment_name+"_"+datetime.now().strftime("%Y%m%d-%H%M%S")
        Path(str(os.path.join(config.directory.log,'wandb'))).mkdir(exist_ok=True,parents=True)

        logger = WandbLogger(
            project=project_name,
            name=experiment_name,
            id=run_name,
            save_dir=str(os.path.join(config.directory.log))
        )
        

        save_config(config, Path(config.directory.log).joinpath(
            "wandb",
            logger.experiment._settings.sync_dir,
            "config.yaml"
        ))
        return logger