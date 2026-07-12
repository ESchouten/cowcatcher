import logging
from importlib import import_module
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class WinMLRuntime:
    def __init__(self):
        self._fix_winrt_runtime()
        winml = import_module("winui3.microsoft.windows.ai.machinelearning")
        bootstrap = import_module(
            "winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap"
        )

        handle = bootstrap.initialize(
            options=bootstrap.InitializeOptions.ON_NO_MATCH_SHOW_UI
        )
        handle.__enter__()
        try:
            providers = winml.ExecutionProviderCatalog.get_default().find_all_providers()
            LOGGER.info(
                "Found %d execution providers: %s",
                len(providers),
                [provider.name for provider in providers],
            )
            ep_paths: dict[str, str] = {}
            for provider in providers:
                LOGGER.info("Ensuring ready: %s", provider.name)
                operation = provider.ensure_ready_async()

                def on_progress(_async_info, progress_info):
                    LOGGER.info("Progress: %.0f%%", progress_info)

                operation.progress = on_progress
                LOGGER.info("Result: %s", operation.get())
                if provider.library_path:
                    ep_paths[provider.name] = provider.library_path
        except Exception:
            handle.__exit__(None, None, None)
            raise

        self._win_app_sdk_handle = handle
        self._providers = providers
        self._ep_paths = ep_paths
        self._registered_eps: list[str] = []

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
