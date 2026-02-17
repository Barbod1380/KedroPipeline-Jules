from kedro.pipeline import Pipeline
from data_pipeline.pipelines.data_processing.pipeline import (
    create_pipeline,
    create_partial_pipeline_from_cropped,
    create_partial_pipeline_from_cropped_gilan,
    create_partial_pipeline_from_cropped_sari,
)


def register_pipelines() -> dict[str, Pipeline]:
    """
    Register all pipelines
    """
    return {
        "__default__": create_pipeline(),
        "from_cropped": create_partial_pipeline_from_cropped(),
        "from_cropped_gilan": create_partial_pipeline_from_cropped_gilan(),
        "from_cropped_sari": create_partial_pipeline_from_cropped_sari()
    }