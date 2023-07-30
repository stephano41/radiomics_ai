import logging

from autorad.inference.infer_utils import get_best_run_from_experiment_name

from src.evaluation import bootstrap, log_ci2mlflow
from src.inference import get_pipeline_from_last_run, get_last_run_from_experiment_name
from src.utils.infer_utils import get_pipeline_from_best_run

logger = logging.getLogger(__name__)


def evaluate(experiment_name, X, y, rank_method='last', bs_iters=500, num_cpu=2, labels=None, bs_method='.632+', alpha=0.95):

    if rank_method=='last':
        pipeline = get_pipeline_from_last_run(experiment_name)
        run_id = get_last_run_from_experiment_name(experiment_name).run_id
    elif rank_method=='best':
        pipeline = get_pipeline_from_best_run(experiment_name)
        run_id = get_best_run_from_experiment_name(experiment_name).run_id
    else:
        raise ValueError(f"rank_method {rank_method} not implemented")

    confidence_interval = bootstrap(pipeline, X, y,
                                    method=bs_method,
                                    iters=bs_iters,
                                    num_cpu=num_cpu,
                                    labels=labels,
                                    alpha=alpha
                                    )

    logger.info(confidence_interval)
    log_ci2mlflow(confidence_interval,
                  run_id=run_id)