"""Enhanced LabelingService with DSPy optimization and advanced RAG support.

This module extends the base LabelingService to support prompt optimization
and advanced RAG methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from ...config import Settings
from ..configs import LabelingConfig, DSPyOptimizationConfig, AdvancedRAGConfig
from ..knowledge import KnowledgeStore
from ..optimization import DSPyOptimizer, DSPyConfig, DSPyOptimizationResult
from ..rag import GraphRAGConfig, RAPTORConfig
from .labeling_service import LabelingService


class OptimizedLabelingService(LabelingService):
    """LabelingService with prompt optimization and advanced RAG.

    This service extends the base LabelingService to provide:
    - DSPy-based prompt optimization
    - Advanced RAG methods (GraphRAG, RAPTOR)
    - Automatic prompt caching and reuse

    Example:
        >>> # Initialize with optimization
        >>> config = LabelingConfig(use_rag=True)
        >>> dspy_config = DSPyOptimizationConfig(enabled=True)
        >>> rag_config = AdvancedRAGConfig(rag_mode='graph')
        >>> service = OptimizedLabelingService(
        ...     dataset_name='sentiment',
        ...     settings=settings,
        ...     config=config,
        ...     dspy_config=dspy_config,
        ...     rag_config=rag_config
        ... )
        >>>
        >>> # Optimize prompts first
        >>> result = service.optimize_prompts(train_df, val_df, text_col='text', label_col='label')
        >>>
        >>> # Use optimized prompts for labeling
        >>> labeled_df = service.label_dataframe(test_df, text_column='text')
    """

    def __init__(
        self,
        dataset_name: str,
        settings: Settings,
        config: LabelingConfig | None = None,
        dspy_config: DSPyOptimizationConfig | None = None,
        rag_config: AdvancedRAGConfig | None = None,
    ):
        """Initialize optimized labeling service.

        Args:
            dataset_name: Name of the dataset
            settings: Global settings
            config: Labeling configuration
            dspy_config: DSPy optimization configuration
            rag_config: Advanced RAG configuration
        """
        # Initialize base service
        super().__init__(dataset_name, settings, config)

        # Store optimization configurations
        self.dspy_config = dspy_config or DSPyOptimizationConfig()
        self.rag_config = rag_config or AdvancedRAGConfig()

        # Initialize optimizer if enabled
        self.optimizer: DSPyOptimizer | None = None
        if self.dspy_config.enabled:
            try:
                optimizer_config = DSPyConfig(
                    model_name=self.dspy_config.model_name,
                    num_candidates=self.dspy_config.num_candidates,
                    num_trials=self.dspy_config.num_trials,
                    max_bootstrapped_demos=self.dspy_config.max_bootstrapped_demos,
                    max_labeled_demos=self.dspy_config.max_labeled_demos,
                    init_temperature=self.dspy_config.init_temperature,
                    metric_threshold=self.dspy_config.metric_threshold,
                )
                self.optimizer = DSPyOptimizer(optimizer_config)
                logger.info('DSPy optimizer initialized')
            except ImportError as e:
                logger.warning(f'DSPy not available, optimization disabled: {e}')
                self.optimizer = None

        # Override knowledge store with advanced RAG if configured
        if self.rag_config.rag_mode != 'traditional':
            logger.info(f'Using advanced RAG mode: {self.rag_config.rag_mode}')
            self.knowledge_store = KnowledgeStore(
                dataset_name, settings=self.settings, rag_mode=self.rag_config.rag_mode
            )

            # Auto-build advanced indices if configured
            if self.rag_config.auto_build_on_startup:
                self._build_advanced_indices()

        # Cached optimization results
        self.optimization_result: DSPyOptimizationResult | None = None
        self.optimized_prompt_path = (
            self.storage_path / 'optimized_prompts' / 'latest.json'
        )

        # Try to load cached optimized prompts
        if self.dspy_config.cache_optimized_prompts:
            self._load_cached_prompts()

        logger.info(f'OptimizedLabelingService initialized for {dataset_name}')

    def _build_advanced_indices(self) -> None:
        """Build advanced RAG indices."""
        try:
            if self.rag_config.rag_mode == 'graph':
                graph_config = GraphRAGConfig(
                    similarity_threshold=self.rag_config.graph_similarity_threshold,
                    max_neighbors=self.rag_config.graph_max_neighbors,
                    use_communities=self.rag_config.graph_use_communities,
                    pagerank_alpha=self.rag_config.graph_pagerank_alpha,
                )
                self.knowledge_store.build_graph_rag(graph_config)
                logger.info('GraphRAG index built')

            elif self.rag_config.rag_mode == 'raptor':
                # RAPTOR requires a summarize function - we'll need the LLM client
                def summarize_cluster(texts: list[str]) -> str:
                    """Summarize a cluster of texts using LLM."""
                    combined = '\n\n'.join(texts[:10])  # Limit to first 10
                    prompt = (
                        f'Summarize the following examples in {self.rag_config.raptor_summary_length} words or less:\n\n'
                        f'{combined}'
                    )
                    client = self._get_client_for_config(self.config)
                    response = client.invoke(prompt)
                    return response.content if hasattr(response, 'content') else str(response)

                raptor_config = RAPTORConfig(
                    max_tree_depth=self.rag_config.raptor_max_tree_depth,
                    clustering_threshold=self.rag_config.raptor_clustering_threshold,
                    min_cluster_size=self.rag_config.raptor_min_cluster_size,
                    summary_length=self.rag_config.raptor_summary_length,
                    use_multi_level_retrieval=self.rag_config.raptor_use_multi_level,
                )
                self.knowledge_store.build_raptor_rag(raptor_config, summarize_cluster)
                logger.info('RAPTOR tree built')

        except Exception as e:
            logger.error(f'Failed to build advanced RAG indices: {e}')

    def optimize_prompts(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        text_column: str,
        label_column: str,
    ) -> DSPyOptimizationResult:
        """Optimize prompts using DSPy.

        Args:
            train_df: Training data
            val_df: Validation data
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            DSPyOptimizationResult with optimized prompts

        Example:
            >>> result = service.optimize_prompts(
            ...     train_df=train_data,
            ...     val_df=val_data,
            ...     text_column='text',
            ...     label_column='label'
            ... )
            >>> print(f'Validation accuracy: {result.validation_accuracy:.2%}')
        """
        if self.optimizer is None:
            raise RuntimeError(
                'DSPy optimizer not available. Install dspy-ai or enable optimization.'
            )

        logger.info('Starting prompt optimization...')

        # Run optimization
        result = self.optimizer.optimize_labeling_prompt(
            train_df=train_df,
            val_df=val_df,
            text_column=text_column,
            label_column=label_column,
        )

        # Cache result
        self.optimization_result = result

        # Save if caching enabled
        if self.dspy_config.cache_optimized_prompts:
            self.optimized_prompt_path.parent.mkdir(parents=True, exist_ok=True)
            self.optimizer.save_optimized_prompt(result, self.optimized_prompt_path)

        logger.info(
            f'Optimization complete: Train={result.train_accuracy:.2%}, '
            f'Val={result.validation_accuracy:.2%}'
        )

        return result

    def _load_cached_prompts(self) -> bool:
        """Load cached optimized prompts.

        Returns:
            True if prompts loaded successfully, False otherwise
        """
        if not self.optimized_prompt_path.exists():
            return False

        try:
            import json

            with open(self.optimized_prompt_path) as f:
                cached_data = json.load(f)

            logger.info(
                f"Loaded cached optimized prompts (val_acc={cached_data.get('validation_accuracy', 0):.2%})"
            )
            return True

        except Exception as e:
            logger.warning(f'Failed to load cached prompts: {e}')
            return False

    def label_text_with_advanced_rag(
        self,
        text: str,
        config: LabelingConfig | None = None,
        template_path: Path | None = None,
        ruleset: dict[str, Any] | None = None,
        rag_mode: str | None = None,
    ):
        """Label text using advanced RAG methods.

        Args:
            text: Text to label
            config: Labeling configuration
            template_path: Custom template path
            ruleset: Optional ruleset
            rag_mode: Override RAG mode ('traditional', 'graph', 'raptor')

        Returns:
            LabelResponse with predicted label

        Example:
            >>> # Use configured RAG mode
            >>> response = service.label_text_with_advanced_rag('example text')
            >>>
            >>> # Override to use GraphRAG
            >>> response = service.label_text_with_advanced_rag('example text', rag_mode='graph')
        """
        config = config or self.config
        template = self._load_template(template_path) if template_path else self.template

        # Prepare prompt with advanced RAG
        if config.use_rag:
            k = config.k_examples or self.settings.max_examples_per_query
            source_filter = 'human' if config.prefer_human_examples else None

            # Use advanced RAG method
            examples = self.knowledge_store.find_similar_examples_advanced(
                text=text,
                k=k,
                source_filter=source_filter,
                rag_mode=rag_mode or self.rag_config.rag_mode,
            )

            logger.debug(
                f'Advanced RAG ({rag_mode or self.rag_config.rag_mode}) retrieved {len(examples)} examples'
            )
        else:
            examples = []

        # Convert to template format
        template_examples = []
        for ex in examples:
            template_ex = {
                'page_content': ex.get('text', ''),
                'metadata': {
                    'label': ex.get('label', ''),
                    'source': ex.get('source', ''),
                    'confidence': ex.get('confidence', 1.0),
                    **ex.get('metadata', {}),
                },
            }
            template_examples.append(template_ex)

        # Render prompt
        rendered_prompt = template.render(
            text=text, examples=template_examples, ruleset=ruleset
        )

        # Track prompt
        prompt_id = self.prompt_manager.store_prompt(
            prompt_text=rendered_prompt,
            template_source=str(template.filename),
            model_name=config.model_name or self.settings.llm_model,
            examples_used=examples,
        )

        try:
            # Get prediction
            if config.use_validation:
                validator = self._get_validator_for_config(config)
                validation_rules = self._get_validation_rules(config)

                from ...models import LabelResponse

                response = validator.validate_and_retry(
                    prompt=rendered_prompt,
                    response_model=LabelResponse,
                    validation_rules=validation_rules,
                    method='function_calling',
                )
            else:
                from ...models import LabelResponse

                llm_client = self._get_client_for_config(config)
                structured_llm = llm_client.with_structured_output(
                    LabelResponse, method='function_calling'
                )
                response = structured_llm.invoke(rendered_prompt)

            self.prompt_manager.update_result(
                prompt_id, successful=True, confidence=response.confidence
            )
            return response

        except Exception as e:
            self.prompt_manager.update_result(prompt_id, successful=False)
            logger.error(f'Failed to label text: {e}')
            raise

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get statistics about optimization and advanced RAG.

        Returns:
            Dictionary with optimization and RAG statistics

        Example:
            >>> stats = service.get_optimization_stats()
            >>> print(f"Optimization enabled: {stats['dspy_enabled']}")
            >>> print(f"RAG mode: {stats['rag_mode']}")
        """
        stats = {
            'dspy_enabled': self.dspy_config.enabled,
            'dspy_config': self.dspy_config.model_dump() if self.dspy_config.enabled else None,
            'has_optimizer': self.optimizer is not None,
            'optimization_cached': self.optimization_result is not None,
            'rag_mode': self.rag_config.rag_mode,
            'rag_config': self.rag_config.model_dump(),
        }

        if self.optimization_result:
            stats['optimization_result'] = {
                'train_accuracy': self.optimization_result.train_accuracy,
                'validation_accuracy': self.optimization_result.validation_accuracy,
                'optimization_cost': self.optimization_result.optimization_cost,
                'converged': self.optimization_result.converged,
            }

        # Add knowledge store stats
        if self.knowledge_store:
            stats['knowledge_store'] = self.knowledge_store.get_stats()

        return stats
