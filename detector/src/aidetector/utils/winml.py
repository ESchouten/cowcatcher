import logging
from importlib import import_module
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_winml_instance = None


class WinML:
    def __new__(cls, *args, **kwargs):
        global _winml_instance
        if _winml_instance is None:
            _winml_instance = super(WinML, cls).__new__(cls, *args, **kwargs)
            _winml_instance._initialized = False
        return _winml_instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._fix_winrt_runtime()
        winml = import_module("winui3.microsoft.windows.ai.machinelearning")
        bootstrap = import_module(
            "winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap"
        )

        self._win_app_sdk_handle = bootstrap.initialize(
            options=bootstrap.InitializeOptions.ON_NO_MATCH_SHOW_UI
        )
        self._win_app_sdk_handle.__enter__()
        catalog = winml.ExecutionProviderCatalog.get_default()
        self._providers = catalog.find_all_providers()
        LOGGER.info(
            "Found %d execution providers: %s",
            len(self._providers),
            [p.name for p in self._providers],
        )
        self._ep_paths: dict[str, str] = {}
        for provider in self._providers:
            LOGGER.info("Ensuring ready: %s", provider.name)
            operation = provider.ensure_ready_async()

            def on_progress(async_info, progress_info):
                LOGGER.info("Progress: %.0f%%", progress_info)

            operation.progress = on_progress
            result = operation.get()
            LOGGER.info("Result: %s", result)
            if provider.library_path == "":
                continue
            self._ep_paths[provider.name] = provider.library_path
        self._registered_eps: list[str] = []

    def __del__(self):
        self._providers = None
        self._win_app_sdk_handle.__exit__(None, None, None)

    def _fix_winrt_runtime(self):
        """
        This function removes the msvcp140.dll from the winrt-runtime package.
        So it does not cause issues with other libraries.
        """
        from importlib import metadata

        site_packages_path = Path(
            str(metadata.distribution("winrt-runtime").locate_file(""))
        )
        dll_path = site_packages_path / "winrt" / "msvcp140.dll"
        if dll_path.exists():
            dll_path.unlink()

    def register_execution_providers_to_ort(self) -> list[str]:
        import onnxruntime as ort

        for name, path in self._ep_paths.items():
            if name not in self._registered_eps:
                try:
                    LOGGER.info("Registering execution provider %s: %s", name, path)
                    ort.register_execution_provider_library(name, path)
                    self._registered_eps.append(name)
                except Exception as e:
                    LOGGER.exception(
                        "Failed to register execution provider %s: %s", name, e
                    )
        LOGGER.info("Registered execution providers: %s", self._registered_eps)
        return self._registered_eps
