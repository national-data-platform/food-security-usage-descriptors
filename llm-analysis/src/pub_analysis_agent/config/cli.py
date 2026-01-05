"""
CLI utility for configuration management and validation.

This module provides command-line tools for validating, testing, and
debugging configuration settings.
"""

import sys
import asyncio
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, UTC

import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import print as rprint

from pub_analysis_agent.agents.code_extraction_agent import CodeExtractionAgent, ExtractionConfig
from pub_analysis_agent.agents.github_repo_verification_agent import GitHubRepositoryVerificationAgent, GitHubVerificationConfig
from pub_analysis_agent.agents.dataset_join_agent import DatasetJoinAnalysisAgent, JoinAnalysisConfig
from pub_analysis_agent.agents.dataset_validation_agent import DatasetValidationAgent, ValidationConfig
from pub_analysis_agent.services.code_analysis_llm import CodeAnalysisConfig
from pub_analysis_agent.services.dataset_service import DatasetService
from pub_analysis_agent.services.link_validator import LinkValidationConfig

from .loader import ConfigurationLoader, validate_configuration_file
from .settings import Settings
from ..workflows.workflow_orchestrator import WorkflowOrchestrator
from ..services.results_service import ResultsService
from ..services.mongodb_client import MongoDBClient
from ..services.llm_service import LLMService, LLMModelConfig
from ..agents.triage_agent import TriageAgent
from ..agents.dataset_discovery_agent import DatasetDiscoveryAgent, DiscoveryConfig


console = Console()


def _convert_pandas_row_to_dict(row: pd.Series) -> Dict[str, Any]:
    """Convert a pandas Series row to a dictionary, handling numpy arrays and other non-serializable types."""
    import numpy as np
    
    def convert_value(value):
        if isinstance(value, dict):
            return {k: convert_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [convert_value(item) for item in value]
        elif isinstance(value, tuple):
            return tuple(convert_value(item) for item in value)
        elif isinstance(value, np.ndarray):
            # Handle empty arrays and object arrays specially
            if value.size == 0:
                return []
            elif value.dtype == object:
                # For object arrays, convert each element recursively
                return [convert_value(item) for item in value.tolist()]
            else:
                return value.tolist()
        elif isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif pd.isna(value):
            return None
        elif hasattr(value, '__dict__'):
            # Handle custom objects by converting their dict representation
            try:
                return convert_value(value.__dict__)
            except:
                return str(value)
        else:
            return value
    
    return {key: convert_value(value) for key, value in row.to_dict().items()}


@click.group()
def config_cli():
    """Configuration management CLI."""
    pass


@config_cli.command()
@click.option(
    "--config-path",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose output"
)
def validate(config_path: Optional[Path], verbose: bool) -> None:
    """Validate configuration file and settings."""
    try:
        # Validate configuration
        result = validate_configuration_file(config_path)
        
        if result['valid']:
            console.print("✅ Configuration is valid", style="green")
            
            if verbose and result['settings']:
                display_configuration_summary(result['settings'])
        else:
            console.print("❌ Configuration validation failed", style="red")
            console.print(f"Error: {result['error']}", style="red")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"❌ Validation error: {e}", style="red")
        sys.exit(1)


@config_cli.command()
@click.option(
    "--config-path",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file"
)
def show(config_path: Optional[Path]) -> None:
    """Display current configuration settings."""
    try:
        loader = ConfigurationLoader(config_path)
        settings = loader.create_settings()
        
        display_configuration_summary(settings)
        
    except Exception as e:
        console.print(f"❌ Error loading configuration: {e}", style="red")
        sys.exit(1)


@config_cli.command()
@click.option(
    "--config-path",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file"
)
def test_connections(config_path: Optional[Path]) -> None:
    """Test connections to configured services."""
    try:
        loader = ConfigurationLoader(config_path)
        settings = loader.create_settings()
        
        console.print("🔍 Testing service connections...\n")
        
        # Test each service
        services = [
            ("MongoDB", test_mongodb_connection, settings.database),
            ("Elasticsearch", test_elasticsearch_connection, settings.elasticsearch),
            # ("Ollama", test_ollama_connection, settings.llm),
            ("LM Studio", test_lmstudio_connection, settings.llm),
        ]
        
        results = []
        for service_name, test_func, config in services:
            result = test_func(config)
            results.append((service_name, result))
            
            status = "✅" if result['success'] else "❌"
            console.print(f"{status} {service_name}: {result['message']}")
        
        # Summary
        successful = sum(1 for _, result in results if result['success'])
        total = len(results)
        
        if successful == total:
            console.print(f"\n🎉 All {total} services are accessible!", style="green")
        else:
            console.print(f"\n⚠️  {successful}/{total} services are accessible", style="yellow")
            
    except Exception as e:
        console.print(f"❌ Error testing connections: {e}", style="red")
        sys.exit(1)


def display_configuration_summary(settings: Settings) -> None:
    """Display a formatted summary of configuration settings."""
    # Main configuration panel
    config_info = Text()
    config_info.append(f"Project: {settings.project_name}\n", style="bold blue")
    config_info.append(f"Version: {settings.version}\n")
    
    if settings.development.development_mode:
        config_info.append("Mode: Development\n", style="yellow")
    else:
        config_info.append("Mode: Production\n", style="green")
    
    console.print(Panel(config_info, title="📋 Project Information", expand=False))
    
    # Services table
    table = Table(title="🔧 Service Configuration")
    table.add_column("Service", style="cyan", no_wrap=True)
    table.add_column("Endpoint", style="magenta")
    table.add_column("Status", justify="center")
    
    # Add service rows
    services = [
        ("MongoDB", str(settings.database.connection_string), "Local" if settings.is_local_service("mongodb") else "Network"),
        ("Elasticsearch", str(settings.elasticsearch.url), "Local" if settings.is_local_service("elasticsearch") else "Network"),
        ("Ollama", str(settings.llm.ollama_base_url), "Local" if settings.is_local_service("ollama") else "Network"),
        ("LM Studio", str(settings.llm.lmstudio_base_url), "Local" if settings.is_local_service("lmstudio") else "Network"),
    ]
    
    for service, endpoint, status in services:
        status_style = "green" if status == "Local" else "blue"
        table.add_row(service, endpoint, Text(status, style=status_style))
    
    console.print(table)
    
    # Configuration details
    details_table = Table(title="⚙️ Configuration Details")
    details_table.add_column("Category", style="cyan", no_wrap=True)
    details_table.add_column("Key", style="yellow")
    details_table.add_column("Value", style="green")
    
    # Add configuration details
    config_details = [
        ("LLM", "Default Model", settings.llm.default_model),
        ("LLM", "Temperature", str(settings.llm.temperature)),
        ("LLM", "Max Tokens", str(settings.llm.max_tokens)),
        ("Processing", "Batch Size", str(settings.processing.batch_size)),
        ("Processing", "Max Concurrent", str(settings.processing.max_concurrent)),
        ("Logging", "Level", settings.logging.level),
        ("Logging", "Format", settings.logging.format),
        ("Memory", "Max Usage (MB)", str(settings.processing.max_memory_usage_mb)),
    ]
    
    for category, key, value in config_details:
        details_table.add_row(category, key, value)
    
    console.print(details_table)


def test_mongodb_connection(config) -> dict:
    """Test MongoDB connection."""
    try:
        import pymongo
        console.print(f"Testing MongoDB connection to {config.connection_string}")
        client = pymongo.MongoClient(
            config.connection_string,
            connectTimeoutMS=config.connect_timeout_ms,
            serverSelectionTimeoutMS=config.server_selection_timeout_ms
        )
        # Test connection
        client.admin.command('ping')
        client.close()
        return {'success': True, 'message': 'Connection successful'}
    except ImportError:
        return {'success': False, 'message': 'pymongo not installed'}
    except Exception as e:
        return {'success': False, 'message': f'Connection failed: {str(e)}'}


def test_elasticsearch_connection(config) -> dict:
    """Test Elasticsearch connection."""
    try:
        import httpx
        import ssl
        from pathlib import Path
        
        # Prepare headers for authentication
        headers = {}
        console.print(f"Testing Elasticsearch connection to {config.url}")
        if config.api_key:
            headers['Authorization'] = f'ApiKey {config.api_key}'
        elif config.username and config.password:
            import base64
            credentials = base64.b64encode(f"{config.username}:{config.password}".encode()).decode()
            headers['Authorization'] = f'Basic {credentials}'
        
        # Prepare SSL context if needed
        verify_ssl = True
        if config.ssl_enabled and config.verify_certs:
            if config.ca_cert_path:
                # Use the configured CA certificate path
                certificate_path = Path(config.ca_cert_path)
                if not certificate_path.is_absolute():
                    # If relative path, make it relative to project root
                    project_root = Path(__file__).parent.parent.parent.parent
                    certificate_path = project_root / config.ca_cert_path
                verify_ssl = str(certificate_path)
            else:
                # Use default SSL context
                verify_ssl = ssl.create_default_context()
        elif not config.ssl_enabled:
            verify_ssl = False
        
        # Simple health check
        with httpx.Client(timeout=5.0, verify=verify_ssl, headers=headers) as client:
            response = client.get(f"{config.url}_cluster/health")
            if response.status_code == 200:
                return {'success': True, 'message': 'Connection successful'}
            else:
                return {'success': False, 'message': f'HTTP {response.status_code}'}
                
    except ImportError:
        return {'success': False, 'message': 'httpx not installed'}
    except Exception as e:
        return {'success': False, 'message': f'Connection failed: {str(e)}'}


def test_ollama_connection(config) -> dict:
    """Test Ollama connection."""
    try:
        import httpx
        
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{config.ollama_base_url}/tags")
            if response.status_code == 200:
                return {'success': True, 'message': 'Connection successful'}
            else:
                return {'success': False, 'message': f'HTTP {response.status_code}'}
                
    except ImportError:
        return {'success': False, 'message': 'httpx not installed'}
    except Exception as e:
        return {'success': False, 'message': f'Connection failed: {str(e)}'}


def test_lmstudio_connection(config) -> dict:
    """Test LM Studio connection."""
    try:
        import httpx
        
        with httpx.Client(timeout=5.0) as client:
            console.print(f"Testing LM Studio connection to {config.lmstudio_base_url}")
            response = client.get(f"{config.lmstudio_base_url}/models")
            if response.status_code == 200:
                return {'success': True, 'message': 'Connection successful'}
            else:
                return {'success': False, 'message': f'HTTP {response.status_code}'}
                
    except ImportError:
        return {'success': False, 'message': 'httpx not installed'}
    except Exception as e:
        return {'success': False, 'message': f'Connection failed: {str(e)}'}


# Global storage for execution status
_execution_status: Dict[str, Dict[str, Any]] = {}


@config_cli.command()
@click.option(
    "--config-path",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file"
)
@click.option(
    "--parquet-file",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the parquet file containing publication data"
)
@click.option(
    "--publication-id",
    "-i",
    type=str,
    help="Specific publication ID to analyze (optional)"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose output"
)
@click.option(
    "--provider",
    "-prv",
    type=str,
    help="LLM provider",
    default="lmstudio"
)
def analyze_publication(
    config_path: Optional[Path],
    parquet_file: Path,
    publication_id: Optional[str],
    verbose: bool,
    provider: str
) -> None:
    """Start analysis of a publication from parquet file."""
    execution_id: Optional[str] = None
    try:
        # Load configuration
        loader = ConfigurationLoader(config_path)
        settings = loader.create_settings()
        
        # Validate parquet file
        if not parquet_file.exists():
            console.print(f"❌ Parquet file not found: {parquet_file}", style="red")
            sys.exit(1)
        
        # Generate execution ID
        execution_id = str(uuid.uuid4())
        
        # Initialize execution status
        _execution_status[execution_id] = {
            "status": "starting",
            "start_time": datetime.now(UTC).isoformat(),
            "parquet_file": str(parquet_file),
            "publication_id": publication_id,
            "progress": 0.0,
            "current_step": "initializing",
            "error": None
        }
        
        console.print(f"🚀 Starting analysis with execution ID: {execution_id}")
        console.print(f"📁 Parquet file: {parquet_file}")
        if publication_id:
            console.print(f"📄 Publication ID: {publication_id}")
        
        # Run the analysis asynchronously
        asyncio.run(_run_analysis_async(
            execution_id, settings, parquet_file, publication_id, verbose, provider
        ))
        
    except Exception as e:
        console.print(f"❌ Error starting analysis: {e}", style="red")
        if execution_id in _execution_status:
            _execution_status[execution_id]["status"] = "failed"
            _execution_status[execution_id]["error"] = str(e)
        sys.exit(1)


async def _run_analysis_async(
    execution_id: str,
    settings: Settings,
    parquet_file: Path,
    publication_id: Optional[str],
    verbose: bool,
    provider: str
) -> None:
    """Run the analysis asynchronously."""
    try:
        # Update status
        _execution_status[execution_id]["status"] = "loading_data"
        _execution_status[execution_id]["current_step"] = "Loading parquet data"
        _execution_status[execution_id]["progress"] = 10.0
        
        # Load parquet data
        df = pd.read_parquet(parquet_file)
        print(df.columns)
        
        if verbose:
            console.print(f"📊 Loaded {len(df)} rows from parquet file")
        
        # Filter by publication ID if specified
        if publication_id:
            if 'publication_id' not in df.columns:
                console.print("❌ Column 'publication_id' not found in parquet file", style="red")
                _execution_status[execution_id]["status"] = "failed"
                _execution_status[execution_id]["error"] = "Column 'publication_id' not found"
                return
            
            df = df[df['publication_id'] == publication_id]
            if len(df) == 0:
                console.print(f"❌ Publication ID '{publication_id}' not found in parquet file", style="red")
                _execution_status[execution_id]["status"] = "failed"
                _execution_status[execution_id]["error"] = f"Publication ID '{publication_id}' not found"
                return
        
        # Initialize services
        _execution_status[execution_id]["status"] = "initializing_services"
        _execution_status[execution_id]["current_step"] = "Initializing services"
        _execution_status[execution_id]["progress"] = 20.0
        
        # Initialize MongoDB client
        console.print("Initialize MongoDB client", style="green")
        mongodb_client = MongoDBClient(settings.database)
        mongodb_client.connect()  # connect é síncrono
        
        # Initialize results service
        console.print("Initialize results service", style="green")
        results_service = ResultsService(mongodb_client, settings.database)
        await results_service.initialize()
        
        # Initialize workflow orchestrator
        console.print("Initialize workflow orchestrator", style="green")
        workflow_orchestrator = WorkflowOrchestrator()
        

        base_url = "http://192.168.62.9:1234/v1"

        if provider == "ollama":
            base_url = "http://localhost:11434/v1"

        # Initialize and register agents
        console.print("Initialize and register agents", style="green")
        llm_config = LLMModelConfig(
            name=settings.llm.default_model,
            base_url=base_url,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            timeout=settings.llm.request_timeout
        )
        llm_service = LLMService(llm_config)
        
        triage_agent = TriageAgent(llm_service)
        workflow_orchestrator.register_agent("classify_data_analysis", triage_agent.analyze)
        
        dataset_service = DatasetService(mongodb_client)
        
        config_validate = ValidationConfig(
            confidence_threshold=1,
            fuzzy_match_threshold=80,
            max_text_length=400000,
            temperature=0.7,
            max_tokens=10000,
            context_window=400000,
            min_dataset_name_length=100,
            batch_size=10
        )

        dataset_validate_agent = DatasetValidationAgent(llm_service, dataset_service, config_validate)
        workflow_orchestrator.register_agent("validate_datasets", dataset_validate_agent.validate_datasets)
        
        config_discovery = DiscoveryConfig(
            confidence_threshold=8.0,
            max_tokens=40000,
            context_window=400000,
            max_text_length=400000,
            temperature=0.7,
            enable_unknown_dataset_discovery=False)
        dataset_discovery_agent = DatasetDiscoveryAgent(llm_service, dataset_service, config_discovery)
        workflow_orchestrator.register_agent("discover_new_datasets", dataset_discovery_agent.discover_datasets)

        config_join = JoinAnalysisConfig(
            confidence_threshold=1,
            max_text_length=40000,
            temperature=0.7,
            max_tokens=5000,
            context_window=40000,
            min_join_confidence=5.0,
            max_joins_per_call=10,
            enable_methodology_extraction=True,
            enable_challenge_documentation=True
        )
        dataset_join_agent = DatasetJoinAnalysisAgent(llm_service, dataset_service, config_join)
        workflow_orchestrator.register_agent("analyze_dataset_joins", dataset_join_agent.analyze_dataset_joins)
        
        
        config_code_extraction = ExtractionConfig(
            max_text_length=400000,
            temperature=0.7,
            max_tokens=4000,
            link_validation_timeout=5,
            min_code_snippet_length=10,
            max_code_snippet_length=2000,
            relevance_threshold=6.0,
            validate_links=True
        )
        code_analysis_config = CodeAnalysisConfig(
            temperature=0.7,
            max_tokens=4000,
            confidence_threshold=0.7,
            relevance_threshold=6.0,
            context_window=400000,
            batch_size=5
        )
        link_validation_config = LinkValidationConfig(
            timeout=5,
            max_retries=3
        )
        code_extraction_agent = CodeExtractionAgent(llm_service, config_code_extraction, code_analysis_config, link_validation_config)
        workflow_orchestrator.register_agent("extract_code_snippets", code_extraction_agent.extract_code_and_links)

        verify_config = GitHubVerificationConfig()
        github_verifier = GitHubRepositoryVerificationAgent(llm_service, None, verify_config)
        workflow_orchestrator.register_agent("verify_github_repositories", github_verifier.verify_github_repositories)

        console.print("Process each publication", style="green")
        total_publications = len(df)
        _execution_status[execution_id]["total_publications"] = total_publications
        _execution_status[execution_id]["processed_publications"] = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"Analyzing {total_publications} publication(s)...",
                total=total_publications
            )
            
            for idx, row in df.iterrows():
                try:
                    pub_id = row.get('publication_id', f"pub_{idx}")
                    
                    _execution_status[execution_id]["current_publication"] = pub_id
                    _execution_status[execution_id]["current_step"] = f"Analyzing {pub_id}"
                    _execution_status[execution_id]["progress"] = 30.0 + (idx / total_publications) * 60.0
                    
                    # Convert row to initial data and handle numpy arrays
                    initial_data = _convert_pandas_row_to_dict(row)
                    
                    # Execute workflow
                    result_state = await workflow_orchestrator.execute_workflow(
                        publication_id=pub_id,
                        initial_data=initial_data
                    )
                    
                    # Convert AnalysisState to a format that can be stored
                    # For now, just store the state as a dictionary
                    state_dict = result_state.to_dict()
                    state_dict['_id'] = pub_id  # Use publication_id as document ID
                    
                    # Store results directly in MongoDB
                    mongodb_client.results_collection.replace_one(
                        {'_id': pub_id},
                        state_dict,
                        upsert=True
                    )
                    
                    _execution_status[execution_id]["processed_publications"] += 1
                    progress.update(task, advance=1)
                    
                    if verbose:
                        console.print(f"✅ Completed analysis for {pub_id}")
                        
                except Exception as e:
                    console.print(f"⚠️ Error analyzing {pub_id}: {e}", style="yellow")
                    if verbose:
                        console.print(f"Continuing with next publication...")
        
        # Finalize
        _execution_status[execution_id]["status"] = "completed"
        _execution_status[execution_id]["current_step"] = "Analysis completed"
        _execution_status[execution_id]["progress"] = 100.0
        _execution_status[execution_id]["end_time"] = datetime.now(UTC).isoformat()
        
        console.print(f"🎉 Analysis completed successfully!")
        console.print(f"📊 Processed {_execution_status[execution_id]['processed_publications']} publication(s)")
        
    except Exception as e:
        _execution_status[execution_id]["status"] = "failed"
        _execution_status[execution_id]["error"] = str(e)
        _execution_status[execution_id]["end_time"] = datetime.now(UTC).isoformat()
        console.print(f"❌ Analysis failed: {e}", style="red")


@config_cli.command()
@click.option(
    "--execution-id",
    "-e",
    type=str,
    help="Specific execution ID to check (optional)"
)
@click.option(
    "--all",
    "-a",
    is_flag=True,
    help="Show all executions"
)
def status(execution_id: Optional[str], all: bool) -> None:
    """Show execution status."""
    try:
        if execution_id:
            if execution_id not in _execution_status:
                console.print(f"❌ Execution ID not found: {execution_id}", style="red")
                sys.exit(1)
            
            _display_execution_status(execution_id)
        elif all:
            if not _execution_status:
                console.print("📭 No executions found", style="yellow")
                return
            
            for eid in _execution_status:
                _display_execution_status(eid)
                console.print()  # Add spacing between executions
        else:
            # Show most recent execution
            if not _execution_status:
                console.print("📭 No executions found", style="yellow")
                return
            
            most_recent = max(_execution_status.keys(), 
                            key=lambda x: _execution_status[x].get("start_time", ""))
            _display_execution_status(most_recent)
            
    except Exception as e:
        console.print(f"❌ Error checking status: {e}", style="red")
        sys.exit(1)


def _display_execution_status(execution_id: str) -> None:
    """Display status for a specific execution."""
    status_info = _execution_status[execution_id]
    
    # Create status table
    table = Table(title=f"Execution Status: {execution_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    
    # Add status information
    table.add_row("Status", status_info.get("status", "unknown"))
    table.add_row("Start Time", status_info.get("start_time", "unknown"))
    table.add_row("End Time", status_info.get("end_time", "unknown"))
    table.add_row("Parquet File", status_info.get("parquet_file", "unknown"))
    table.add_row("Publication ID", status_info.get("publication_id", "all"))
    table.add_row("Current Step", status_info.get("current_step", "unknown"))
    table.add_row("Progress", f"{status_info.get('progress', 0.0):.1f}%")
    
    if "total_publications" in status_info:
        table.add_row("Total Publications", str(status_info["total_publications"]))
        table.add_row("Processed Publications", str(status_info["processed_publications"]))
    
    if status_info.get("current_publication"):
        table.add_row("Current Publication", status_info["current_publication"])
    
    if status_info.get("error"):
        table.add_row("Error", status_info["error"], style="red")
    
    console.print(table)


if __name__ == "__main__":
    config_cli() 