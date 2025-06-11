"""Base classes and mixins for the AutoLabeler system."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generic, TypeVar

from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

from ..config import Settings

T = TypeVar("T", bound=BaseModel)


class ConfigurableComponent(Generic[T]):
    """
    Base class for components that need configuration and storage.

    Provides standardized initialization for dataset-specific components,
    ensuring consistent storage paths and configuration management.
    """

    def __init__(
        self,
        component_type: str,
        dataset_name: str,
        settings: Settings,
        config: T | None = None,
    ):
        """
        Initialize the component with its configuration.

        Args:
            component_type: Type of component (e.g., 'labeling', 'evaluation')
            dataset_name: Unique identifier for the dataset
            settings: Application settings
            config: Component-specific Pydantic configuration model
        """
        self.component_type = component_type
        self.dataset_name = dataset_name
        self.settings = settings
        self.config = config

        # Set up storage path
        self.storage_path = Path(settings.results_dir) / component_type / dataset_name
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized {component_type} component for dataset: {dataset_name}"
        )

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the component's state."""
        return {
            "component_type": self.component_type,
            "dataset_name": self.dataset_name,
            "config": self.config.model_dump() if self.config else None,
        }


class ProgressTracker:
    """
    Mixin for tracking and resuming progress in long-running tasks.

    Handles saving and loading progress to/from a JSON file, allowing
    interrupted tasks to be resumed efficiently.
    """

    def __init__(self, progress_file: str):
        """Initialize progress tracker with a file name."""
        self._progress_file = progress_file
        self._processed_items: dict[str, Any] = {}

    def _get_progress_path(self) -> Path:
        # This can be a concrete implementation now
        if not hasattr(self, "storage_path"):
            raise AttributeError(
                "Component using ProgressTracker must have a 'storage_path' attribute."
            )
        return self.storage_path / self._progress_file

    def load_progress(self, resume_key: str) -> None:
        """Load progress from file if resume is enabled."""
        if not resume_key:
            return

        progress_path = self._get_progress_path()
        if progress_path.exists():
            try:
                with open(progress_path, "r") as f:
                    progress_data = json.load(f)
                loaded_data = progress_data.get(resume_key, {})

                if isinstance(loaded_data, list):
                    logger.warning(
                        "Old progress file format detected. Only IDs were saved, so results "
                        "cannot be recovered. Future runs will use the new format."
                    )
                    self._processed_items = {
                        str(item_id): {"autolabeler_id": str(item_id)}
                        for item_id in loaded_data
                    }
                else:
                    self._processed_items = loaded_data
                    logger.info(
                        f"Resuming progress for '{resume_key}'. "
                        f"Loaded {len(self._processed_items)} processed items."
                    )
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Could not load progress from {progress_path}: {e}")

    def save_progress(self, resume_key: str) -> None:
        """Save progress to file."""
        if not resume_key:
            return

        progress_path = self._get_progress_path()
        try:
            data = {}
            if progress_path.exists():
                with open(progress_path, "r") as f:
                    data = json.load(f)
            data[resume_key] = self._processed_items
            with open(progress_path, "w") as f:
                json.dump(data, f, indent=2)
        except (IOError, TypeError) as e:
            logger.error(f"Could not save progress to {progress_path}: {e}")

    def add_to_progress(self, item_id: Any, result: dict) -> None:
        """Add a processed item's result to the progress tracker."""
        self._processed_items[str(item_id)] = result

    def is_processed(self, item_id: Any) -> bool:
        """Check if an item has already been processed."""
        return str(item_id) in self._processed_items

    def get_processed_results(self) -> list[Any]:
        """Returns the list of already processed results."""
        return list(self._processed_items.values())


class BatchProcessor:
    """Mixin for processing items in batches with progress tracking."""

    def process_in_batches(
        self,
        items: list[Any],
        batch_size: int,
        process_func: Any,
        desc: str = "Processing batches",
        resume_key: str | None = None,
        save_interval: int | None = 10,
    ) -> list[Any]:
        """
        Process a list of items in batches with a progress bar.

        Args:
            items: List of items to process
            batch_size: Number of items per batch
            process_func: Function to apply to each batch
            desc: Description for the progress bar
            resume_key: Key for resuming progress
            save_interval: Save progress every N batches

        Returns:
            List of processed results
        """
        results = []
        if resume_key and hasattr(self, "load_progress"):
            self.load_progress(resume_key)
            if hasattr(self, "get_processed_results"):
                results = self.get_processed_results()

        # Filter out already processed items
        items_to_process = [
            item
            for item in items
            if not (
                hasattr(self, "is_processed")
                and self.is_processed(item.get("autolabeler_id"))
            )
        ]

        if not items_to_process and results:
            logger.info(
                f"All {len(results)} items were already processed. Returning cached results."
            )
            return results

        with tqdm(total=len(items_to_process), desc=desc) as pbar:
            for i in range(0, len(items_to_process), batch_size):
                batch = items_to_process[i : i + batch_size]
                batch_results = process_func(batch)
                results.extend(batch_results)

                # Update progress
                if resume_key and hasattr(self, "add_to_progress"):
                    for result_item in batch_results:
                        item_id = result_item.get("autolabeler_id")
                        self.add_to_progress(item_id, result_item)

                if (
                    resume_key
                    and save_interval
                    and (i // batch_size + 1) % save_interval == 0
                ):
                    if hasattr(self, "save_progress"):
                        self.save_progress(resume_key)

                pbar.update(len(batch))

        if resume_key and hasattr(self, "save_progress"):
            self.save_progress(resume_key)

        # De-duplicate results, just in case
        final_results = {item["autolabeler_id"]: item for item in results}
        return list(final_results.values())

    async def process_in_batches_async(
        self,
        items: list[Any],
        batch_size: int,
        process_func: Any,
        max_concurrency: int,
        desc: str = "Processing batches (async)",
    ) -> list[Any]:
        """
        Process items asynchronously in batches.

        Args:
            items: List of items to process
            batch_size: Number of items per batch
            process_func: Async function to apply to each batch
            max_concurrency: Maximum concurrent batches
            desc: Description for the progress bar

        Returns:
            List of processed results
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = []

        async def process_with_semaphore(batch: list[Any]) -> list[Any]:
            async with semaphore:
                return await process_func(batch)

        with tqdm(total=len(items), desc=desc) as pbar:
            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]
                task = asyncio.create_task(process_with_semaphore(batch))
                tasks.append(task)

            # Gather results as they complete
            all_results = []
            for future in asyncio.as_completed(tasks):
                batch_result = await future
                all_results.extend(batch_result)
                pbar.update(len(batch_result))

        return all_results
