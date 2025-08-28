from typing import Dict, Callable, Tuple

from src.noise.core import TimeSampler, NoiseProcess

##############  EXPECTED STRUCTURE  ##############
KEY_PROCESS = 'process'
KEY_SCHEDULE = 'schedule'
KEY_TIMESAMPLER = 'timesampler'

KEY_NOISE_NAME = 'name'
KEY_NOISE_PARAMS = 'params'

def build_noise_process(
        config: Dict,
        process_resolver: Callable[[str], type],
        schedule_resolver: Callable[[str], type],
        timesampler_resolver: Callable[[str], type],
        added_process_config = None
    ) -> Tuple[NoiseProcess, TimeSampler]:

    # resolve the process, schedule and time sampler from the configuration
    diff_process_class = process_resolver(config[KEY_PROCESS][KEY_NOISE_NAME])
    diff_schedule_class = schedule_resolver(config[KEY_SCHEDULE][KEY_NOISE_NAME])
    diff_timesampler_class = timesampler_resolver(config[KEY_TIMESAMPLER][KEY_NOISE_NAME])

    # setup params
    get_params = lambda key: {} if KEY_NOISE_PARAMS not in config[key] else config[key][KEY_NOISE_PARAMS]
    
    ts_params = get_params(KEY_TIMESAMPLER)
    sc_params = get_params(KEY_SCHEDULE)
    pr_params = get_params(KEY_PROCESS)

    if added_process_config is None:
        added_process_config = {}

    # setup timesampler and process (with schedule)
    timesampler = diff_timesampler_class(**ts_params)
    process = diff_process_class(
        schedule=diff_schedule_class(**sc_params),
        **pr_params,
        **added_process_config
    )

    return process, timesampler