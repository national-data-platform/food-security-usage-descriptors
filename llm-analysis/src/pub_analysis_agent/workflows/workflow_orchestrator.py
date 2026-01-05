"""
Workflow orchestrator for publication analysis.

This module defines the workflow orchestrator for managing the publication analysis pipeline.
"""

import logging
import asyncio
import uuid
from datetime import datetime, UTC
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph

from .state_models import AnalysisState

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """LangGraph state type definition."""
    state: AnalysisState
    messages: List[str]
    progress: float


class WorkflowOrchestrator:
    """
    Orchestrates the workflow for analyzing publications.
    
    This class manages the execution of the workflow steps and the state transitions
    between them using LangGraph.
    """
    
    def __init__(
        self,
        agents: Optional[Dict[str, Callable]] = None,
        enable_recovery: bool = True,
        max_retries: int = 3
    ) -> None:
        """
        Initialize the workflow orchestrator.
        
        Args:
            agents: Dictionary mapping step names to agent functions
            enable_recovery: Whether to enable workflow recovery
            max_retries: Maximum number of retries for failed steps
        """
        # Define the workflow steps in order of execution
        self.workflow_steps = [
            "classify_data_analysis",
            "validate_datasets",
            "discover_new_datasets",
            "analyze_dataset_joins",
            "extract_code_snippets",
            "verify_github_repositories",
            "generate_final_output"
        ]
        
        # Initialize agent registry
        self.agents = agents or {}
        
        # Workflow configuration
        self.enable_recovery = enable_recovery
        self.max_retries = max_retries
        
        # Build the workflow graph
        self.workflow_graph = None
        self._build_workflow_graph()
        
        logger.info("Workflow orchestrator initialized")
    
    def _build_workflow_graph(self) -> None:
        """
        Build the workflow graph using LangGraph.
        """
        # Create a new graph
        builder = StateGraph(WorkflowState)
        
        # Add nodes for each step
        for step_name in self.workflow_steps:
            builder.add_node(step_name, self._create_step_executor(step_name))
        
        # Add edges between steps
        for i in range(len(self.workflow_steps) - 1):
            current_step = self.workflow_steps[i]
            next_step = self.workflow_steps[i + 1]
            builder.add_edge(current_step, next_step)
        
        # Set the entry point
        builder.set_entry_point(self.workflow_steps[0])
        
        # Compile the graph
        self.workflow_graph = builder.compile()
        
        logger.info("Workflow graph built successfully")
    
    def _create_step_executor(self, step_name: str) -> Callable:
        """
        Create an executor function for a workflow step.
        
        Args:
            step_name: Name of the workflow step
            
        Returns:
            Step executor function
        """
        async def step_executor(state: WorkflowState) -> WorkflowState:
            """
            Execute a workflow step.
            
            Args:
                state: Current workflow state
                
            Returns:
                Updated workflow state
            """
            try:
                analysis_state = state["state"]
                
                # Log step start
                print(f"🚀 [WORKFLOW] Iniciando step: {step_name}")
                logger.info(f"Starting workflow step: {step_name}")
                
                # Update the current step in the analysis state
                analysis_state.update_step(step_name)
                
                # Get the agent function for this step
                agent_function = self.agents.get(step_name)
                
                if agent_function:
                    # Execute the agent function with retry logic
                    result = await self._execute_with_retry(
                        agent_function, 
                        analysis_state, 
                        step_name
                    )

                    # Update the state with the results
                    analysis_state = self._update_state_with_results(
                        analysis_state, 
                        step_name, 
                        result
                    )

                else:
                    # Use default step execution if no agent is registered
                    await self._default_step_execution(analysis_state, step_name)
                
                # Mark the step as completed
                analysis_state.mark_step_completed(step_name)
                
                # Log step completion
                logger.info(f"Completed workflow step: {step_name}")
                
                # Update progress
                completed_steps = len(analysis_state.completed_steps)
                total_steps = len(self.workflow_steps)
                state["progress"] = completed_steps / total_steps
                
                # Add a message about step completion
                state["messages"].append(f"Step '{step_name}' completed successfully")
                
                return state
                
            except Exception as e:
                # Log step error
                print(f"❌ [WORKFLOW] Erro no step: {step_name} - {e}")
                logger.error(f"Error in step '{step_name}': {e}")
                
                # Update the state with the error
                analysis_state = state["state"]
                analysis_state.update_step(step_name, str(e))
                
                # Add an error message
                state["messages"].append(f"Error in step '{step_name}': {e}")
                
                # If recovery is disabled, re-raise the exception
                if not self.enable_recovery:
                    raise e
                
                # Continue to the next step despite the error
                return state
        
        return step_executor
    
    async def _execute_with_retry(
        self, 
        agent_function: Callable,
        state: AnalysisState,
        step_name: str
    ) -> Any:
        """
        Execute an agent function with retry logic.
        
        Args:
            agent_function: Agent function to execute
            state: Current analysis state
            step_name: Name of the current step
            
        Returns:
            Result of the agent function
        """
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                # Execute the agent function
                if asyncio.iscoroutinefunction(agent_function):
                    result = await agent_function(state)
                else:
                    result = agent_function(state)
                return result
                
            except Exception as e:
                last_error = e
                retries += 1
                logger.warning(
                    f"Step '{step_name}' failed (attempt {retries}/{self.max_retries}): {e}"
                )
                
                # If we've reached the maximum retries, give up
                if retries > self.max_retries:
                    break
        
        # If we get here, all retries failed
        logger.error(f"Step '{step_name}' failed after {self.max_retries} attempts")
        raise last_error
    
    def _update_state_with_results(
        self, 
        state: AnalysisState, 
        step_name: str, 
        result: Any
    ) -> AnalysisState:
        """
        Update the analysis state with the results of a step.
        
        Args:
            state: Analysis state to update
            step_name: Name of the completed step
            result: Result from the agent function
        """
        # Update state based on step name and result
        if step_name == "classify_data_analysis":
            # Extract boolean from TriageResult
            if hasattr(result, 'is_data_analysis'):
                state.is_data_analysis = result.is_data_analysis
            else:
                state.is_data_analysis = result
            
        elif step_name == "identify_datasets":
            state.has_datasets = result
            
        elif step_name == "validate_datasets":
            # Add validated datasets
            if hasattr(result, 'validated_datasets'):
                for dataset in result.validated_datasets:
                    if isinstance(dataset, dict):
                        from .state_models import DatasetMention
                        dataset_obj = DatasetMention(**dataset)
                        state.add_validated_dataset(dataset_obj)
                    else:
                        state.add_validated_dataset(dataset)
                
        elif step_name == "discover_new_datasets":
            # Add newly discovered datasets from DiscoveryResult
            from ..agents.dataset_discovery_agent import DiscoveryResult
            
            if isinstance(result, DiscoveryResult):
                # Extract discovered datasets from the result object
                for discovered_dataset in result.discovered_datasets:
                    from .state_models import DatasetMention
                    # Convert DiscoveredDataset to DatasetMention
                    dataset_mention = DatasetMention(
                        name= discovered_dataset.name,
                        confidence_score_mention= discovered_dataset.confidence_score_mention,
                        confidence_score_use= discovered_dataset.confidence_score_use,
                        text_quote= discovered_dataset.text_quote,
                        context= discovered_dataset.context,
                        description= discovered_dataset.description,
                        source= discovered_dataset.source,
                        domain= discovered_dataset.domain,
                        is_known_dataset= discovered_dataset.is_known_dataset,
                        discovery_reasoning= discovered_dataset.discovery_reasoning,
                        doi= None  # DOI is only extracted during validation, not discovery
                    )
                    state.add_new_dataset(dataset_mention)
            else:
                # Fallback for direct list of datasets
                for dataset in result:
                    if isinstance(dataset, dict):
                        from .state_models import DatasetMention
                        dataset_obj = DatasetMention(**dataset)
                        state.add_new_dataset(dataset_obj)
                    else:
                        state.add_new_dataset(dataset)
                
        elif step_name == "analyze_dataset_joins":
            # Add dataset joins
            if hasattr(result, 'joins_identified'):
                for join in result.joins_identified:
                    if isinstance(join, dict):
                        from .state_models import DatasetJoin
                        join_obj = DatasetJoin(**join)
                        state.add_dataset_join(join_obj)
                    else:
                        state.add_dataset_join(join)
        
        elif step_name == "extract_code_snippets":
            if hasattr(result, 'github_repositories'):
                for repo in result.github_repositories:
                    if isinstance(repo, dict):
                        from .state_models import ExtractedGitHubRepository
                        repo_obj = ExtractedGitHubRepository(**repo)
                        state.add_extracted_github_repo(repo_obj)
                    else:
                        state.add_extracted_github_repo(repo)
            if hasattr(result, 'code_snippets'):
                for snippet in result.code_snippets:
                    if isinstance(snippet, dict):
                        from .state_models import ExtractedCodeSnippet
                        snippet_obj = ExtractedCodeSnippet(**snippet)
                        state.add_extracted_code_snippet(snippet_obj)
                    else:
                        state.add_extracted_code_snippet(snippet)
            if hasattr(result, 'external_links'):
                from .state_converters import convert_external_links_to_state
                converted_links = convert_external_links_to_state(result.external_links)
                for link_obj in converted_links:
                    state.add_extracted_link(link_obj)
        
        elif step_name == "verify_github_repositories":
            # Result should be a list of RepositoryVerificationResult or dicts
            if isinstance(result, list):
                from .state_models import RepositoryVerificationResult
                for item in result:
                    if isinstance(item, dict):
                        state.add_repo_verification_result(RepositoryVerificationResult(**item))
                    else:
                        state.add_repo_verification_result(item)
                
        elif step_name == "generate_final_output" and result is not None:
            # Set final output
            state.final_json = result

        return state
    
    async def _default_step_execution(
        self, 
        state: AnalysisState, 
        step_name: str
    ) -> None:
        """
        Default execution for steps without a registered agent.
        
        Args:
            state: Current analysis state
            step_name: Name of the current step
        """
        logger.info(f"Using default execution for step: {step_name}")
        
        # Add some default logic based on step name
        if step_name == "classify_data_analysis":
            # Default: assume it's a data analysis paper if it has GROBID content
            state.is_data_analysis = state.grobid_content is not None
        
        elif step_name == "generate_final_output":
            # Default: create a basic final output
            state.final_json = {
                "publication_id": state.publication_id,
                "is_data_analysis": state.is_data_analysis,
                "validated_datasets": len(state.validated_datasets),
                "newly_discovered_datasets": len(state.newly_discovered_datasets),
                "dataset_joins": len(state.dataset_joins),
                "processed_at": datetime.now(UTC).isoformat()
            }
    
    def register_agent(self, step_name: str, agent_function: Callable) -> None:
        """
        Register an agent function for a specific workflow step.
        
        Args:
            step_name: Name of the workflow step
            agent_function: Function to execute for this step
        """
        if step_name not in self.workflow_steps:
            raise ValueError(f"Invalid step name: {step_name}")
        
        self.agents[step_name] = agent_function
        logger.info(f"Registered agent for step: {step_name}")
    
    async def execute_workflow(
        self, 
        publication_id: str,
        initial_data: Optional[Dict[str, Any]] = None
    ) -> AnalysisState:
        """
        Execute the complete workflow for a publication.
        
        Args:
            publication_id: ID of the publication to analyze
            initial_data: Optional initial data for the workflow
            
        Returns:
            Final analysis state
        """
        try:
            # Create initial state
            workflow_id = str(uuid.uuid4())
            initial_state = AnalysisState(
                publication_id=publication_id,
                workflow_id=workflow_id
            )
            
            # Add initial data if provided
            if initial_data:
                if "fulltext" in initial_data:
                    # Convert fulltext to ensure it's serializable
                    initial_state.grobid_content = self._convert_for_serialization(initial_data["fulltext"])
                    # Extract all text content from fulltext and populate raw_text
                    initial_state.raw_text = self._extract_text_from_fulltext(initial_data["fulltext"])
                if "raw_text" in initial_data:
                    initial_state.raw_text = initial_data["raw_text"]
            
            # Create initial workflow state
            workflow_state: WorkflowState = {
                "state": initial_state,
                "messages": [],
                "progress": 0.0
            }
            
            logger.info(f"Starting workflow execution for publication: {publication_id}")
            
            # Execute the workflow
            if not self.workflow_graph:
                raise ValueError("Workflow graph not initialized")

            final_state = await self.workflow_graph.ainvoke(workflow_state)
            
            analysis_state = final_state["state"]
            logger.info(f"Workflow completed for publication: {publication_id}")
            
            return analysis_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    def get_workflow_status(self, state: AnalysisState) -> Dict[str, Any]:
        """
        Get the current status of a workflow.
        
        Args:
            state: Analysis state to check
            
        Returns:
            Status information dictionary
        """
        total_steps = len(self.workflow_steps)
        completed_steps = len(state.completed_steps)
        progress = completed_steps / total_steps if total_steps > 0 else 0.0
        
        return {
            "workflow_id": state.workflow_id,
            "publication_id": state.publication_id,
            "current_step": state.current_step,
            "completed_steps": state.completed_steps,
            "progress": progress,
            "error": state.error_message,
            "is_complete": state.is_complete()
        }
    
    def _convert_for_serialization(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable types to serializable formats."""
        import numpy as np
        import pandas as pd
        
        if isinstance(obj, dict):
            return {key: self._convert_for_serialization(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_serialization(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_for_serialization(item) for item in obj)
        elif isinstance(obj, np.ndarray):
            # Handle empty arrays and object arrays specially
            if obj.size == 0:
                return []
            elif obj.dtype == object:
                # For object arrays, convert each element recursively
                return [self._convert_for_serialization(item) for item in obj.tolist()]
            else:
                return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        elif hasattr(obj, '__dict__'):
            # Handle custom objects by converting their dict representation
            try:
                return self._convert_for_serialization(obj.__dict__)
            except:
                return str(obj)
        else:
            return obj
    
    def _extract_text_from_fulltext(self, fulltext: Any) -> str:
        """Extract all text content from fulltext structure and concatenate into a single text field."""
        import numpy as np
        
        def extract_text_recursive(obj: Any) -> List[str]:
            """Recursively extract text from nested structures."""
            texts = []
            
            if isinstance(obj, dict):
                # Look for common text fields first
                text_fields = [
                    'abstract',
                    'acknowledgement',
                    'annex',
                    'annotations',
                    'authorships',
                    'bibliography',
                    'body',
                    'funding',
                    'graphics',
                    'identifiers',
                    'keywords',
                    'content',
                    'notes',
                    'tables',
                    'title']
                processed_fields = set()
                
                for field in text_fields:
                    if field in obj and obj[field]:
                        if isinstance(obj[field], str):
                            texts.append(obj[field])
                            processed_fields.add(field)
                
                # Recursively process remaining values (skip already processed text fields)
                for key, value in obj.items():
                    if key not in processed_fields:
                        texts.extend(extract_text_recursive(value))
                    
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    texts.extend(extract_text_recursive(item))
                    
            elif isinstance(obj, np.ndarray):
                # Convert to list and process
                try:
                    for item in obj.tolist():
                        texts.extend(extract_text_recursive(item))
                except:
                    pass
                    
            elif isinstance(obj, str):
                texts.append(obj)
                
            return texts
        
        try:
            # Extract all text content
            all_texts = extract_text_recursive(fulltext)
            
            # Clean and filter texts, removing duplicates
            cleaned_texts = []
            seen_texts = set()
            
            for text in all_texts:
                if text and isinstance(text, str):
                    # Remove excessive whitespace and empty lines
                    cleaned_text = ' '.join(text.split())
                    if len(cleaned_text) > 3:  # Only include meaningful text
                        # Check for duplicates (case-insensitive)
                        text_lower = cleaned_text.lower()
                        if text_lower not in seen_texts:
                            cleaned_texts.append(cleaned_text)
                            seen_texts.add(text_lower)
            
            # Join all texts with double newlines for readability
            return '\n'.join(cleaned_texts)
            
        except Exception as e:
            logger.warning(f"Failed to extract text from fulltext: {e}")
            return "" 