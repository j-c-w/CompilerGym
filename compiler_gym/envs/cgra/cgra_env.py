class CgraEnv(ClientServiceCompilerEnv):
    def __init__(
        self,
        *args,
        benchmark: Optional[Union[str, Benchmark]] = None,
        datasets_site_path: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
            # Set a default benchmark for use.
            benchmark=benchmark or "cbench-v1/qsort",
            datasets=_[],
            rewards=[ ],
            derived_observation_spaces=[ ],
        )
