"""
Unit tests for CLI configuration management module.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import pandas as pd
import numpy as np
from click.testing import CliRunner

from pub_analysis_agent.config.cli import (
    _convert_pandas_row_to_dict,
    config_cli,
    validate,
    show,
    test_connections,
    display_configuration_summary,
    test_mongodb_connection,
    test_elasticsearch_connection,
    test_ollama_connection,
    test_lmstudio_connection,
    analyze_publication,
    status,
    _display_execution_status
)
from pub_analysis_agent.config.settings import Settings
from pub_analysis_agent.services.llm_service import LLMService
from pub_analysis_agent.services.mongodb_client import MongoDBClient
from pub_analysis_agent.workflows.workflow_orchestrator import WorkflowOrchestrator


class TestCLIUtilities:
    """Test CLI utility functions."""
    
    def test_convert_pandas_row_to_dict_basic_types(self):
        """Test conversion of basic pandas data types."""
        data = {
            'string': 'test',
            'integer': 42,
            'float': 3.14,
            'boolean': True,
            'none': None
        }
        row = pd.Series(data)
        result = _convert_pandas_row_to_dict(row)
        
        assert result == data
        assert isinstance(result['string'], str)
        assert isinstance(result['integer'], int)
        assert isinstance(result['float'], float)
        assert isinstance(result['boolean'], bool)
        assert result['none'] is None
    
    def test_convert_pandas_row_to_dict_numpy_types(self):
        """Test conversion of numpy data types."""
        data = {
            'numpy_int': np.int64(42),
            'numpy_float': np.float64(3.14),
            'numpy_bool': np.bool_(True),
            'numpy_array': np.array([1, 2, 3]),
            'empty_array': np.array([])
        }
        row = pd.Series(data)
        result = _convert_pandas_row_to_dict(row)
        
        assert result['numpy_int'] == 42
        assert isinstance(result['numpy_int'], int)
        assert result['numpy_float'] == 3.14
        assert isinstance(result['numpy_float'], float)
        assert result['numpy_bool'] is True
        assert isinstance(result['numpy_bool'], bool)
        assert result['numpy_array'] == [1, 2, 3]
        assert result['empty_array'] == []
    
    def test_convert_pandas_row_to_dict_complex_types(self):
        """Test conversion of complex data types."""
        data = {
            'dict': {'key': 'value'},
            'list': [1, 2, 3],
            'tuple': (1, 2, 3),
            'nested': {'list': [{'nested': 'value'}]}
        }
        row = pd.Series(data)
        result = _convert_pandas_row_to_dict(row)
        
        assert result['dict'] == {'key': 'value'}
        assert result['list'] == [1, 2, 3]
        assert result['tuple'] == (1, 2, 3)
        assert result['nested'] == {'list': [{'nested': 'value'}]}
    
    def test_convert_pandas_row_to_dict_nan_handling(self):
        """Test handling of NaN values."""
        data = {
            'nan_value': np.nan,
            'inf_value': np.inf,
            'neg_inf': -np.inf
        }
        row = pd.Series(data)
        result = _convert_pandas_row_to_dict(row)
        
        assert result['nan_value'] is None
        assert result['inf_value'] == float('inf')
        assert result['neg_inf'] == float('-inf')
    
    def test_convert_pandas_row_to_dict_custom_objects(self):
        """Test conversion of custom objects."""
        class CustomObject:
            def __init__(self, value):
                self.value = value
        
        custom_obj = CustomObject("test")
        data = {'custom': custom_obj}
        row = pd.Series(data)
        result = _convert_pandas_row_to_dict(row)
        
        assert result['custom'] == {'value': 'test'}


class TestCLICommands:
    """Test CLI command functions."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock(spec=Settings)
        settings.mongodb.connection_string = "mongodb://localhost:27017"
        settings.elasticsearch.hosts = ["localhost:9200"]
        settings.llm.provider = "openai"
        settings.llm.api_key = "test_key"
        return settings
    
    def test_config_cli_group(self, cli_runner):
        """Test that config_cli is a valid Click group."""
        result = cli_runner.invoke(config_cli, ['--help'])
        assert result.exit_code == 0
        assert "Configuration management CLI" in result.output
    
    @patch('pub_analysis_agent.config.cli.Settings')
    def test_validate_command(self, mock_settings_class, cli_runner):
        """Test validate command."""
        mock_settings = Mock()
        mock_settings.validate.return_value = True
        mock_settings_class.return_value = mock_settings
        
        result = cli_runner.invoke(config_cli, ['validate'])
        assert result.exit_code == 0
    
    @patch('pub_analysis_agent.config.cli.Settings')
    def test_show_command(self, mock_settings_class, cli_runner):
        """Test show command."""
        mock_settings = Mock()
        mock_settings_class.return_value = mock_settings
        
        result = cli_runner.invoke(config_cli, ['show'])
        assert result.exit_code == 0
    
    @patch('pub_analysis_agent.config.cli.Settings')
    def test_test_connections_command(self, mock_settings_class, cli_runner):
        """Test test-connections command."""
        mock_settings = Mock()
        mock_settings_class.return_value = mock_settings
        
        result = cli_runner.invoke(config_cli, ['test-connections'])
        assert result.exit_code == 0
    
    @patch('pub_analysis_agent.config.cli.Settings')
    def test_analyze_publication_command(self, mock_settings_class, cli_runner):
        """Test analyze-publication command."""
        mock_settings = Mock()
        mock_settings_class.return_value = mock_settings
        
        result = cli_runner.invoke(config_cli, ['analyze-publication', '--text', 'test text'])
        assert result.exit_code == 0
    
    @patch('pub_analysis_agent.config.cli.Settings')
    def test_status_command(self, mock_settings_class, cli_runner):
        """Test status command."""
        mock_settings = Mock()
        mock_settings_class.return_value = mock_settings
        
        result = cli_runner.invoke(config_cli, ['status'])
        assert result.exit_code == 0


class TestCLIFunctions:
    """Test individual CLI function implementations."""
    
    def test_display_configuration_summary(self, mock_settings):
        """Test configuration summary display."""
        with patch('pub_analysis_agent.config.cli.console.print') as mock_print:
            display_configuration_summary(mock_settings)
            mock_print.assert_called()
    
    def test_test_mongodb_connection_success(self):
        """Test successful MongoDB connection test."""
        config = Mock()
        config.mongodb.connection_string = "mongodb://localhost:27017"
        
        with patch('pub_analysis_agent.config.cli.MongoClient') as mock_mongo_class:
            mock_mongo = Mock()
            mock_mongo.admin.command.return_value = {'ok': 1}
            mock_mongo_class.return_value = mock_mongo
            
            result = test_mongodb_connection(config)
            assert result['status'] == 'success'
    
    def test_test_mongodb_connection_failure(self):
        """Test failed MongoDB connection test."""
        config = Mock()
        config.mongodb.connection_string = "mongodb://invalid:27017"
        
        with patch('pub_analysis_agent.config.cli.MongoClient', side_effect=Exception("Connection failed")):
            result = test_mongodb_connection(config)
            assert result['status'] == 'error'
    
    def test_test_elasticsearch_connection_success(self):
        """Test successful Elasticsearch connection test."""
        config = Mock()
        config.elasticsearch.hosts = ["localhost:9200"]
        config.elasticsearch.username = "user"
        config.elasticsearch.password = "pass"
        
        with patch('pub_analysis_agent.config.cli.elasticsearch.Elasticsearch') as mock_es_class:
            mock_es = Mock()
            mock_es.ping.return_value = True
            mock_es.info.return_value = {'version': {'number': '8.0.0'}}
            mock_es_class.return_value = mock_es
            
            result = test_elasticsearch_connection(config)
            assert result['status'] == 'success'
    
    def test_test_elasticsearch_connection_failure(self):
        """Test failed Elasticsearch connection test."""
        config = Mock()
        config.elasticsearch.hosts = ["localhost:9200"]
        
        with patch('pub_analysis_agent.config.cli.elasticsearch.Elasticsearch', side_effect=Exception("Connection failed")):
            result = test_elasticsearch_connection(config)
            assert result['status'] == 'error'
    
    def test_test_ollama_connection_success(self):
        """Test successful Ollama connection test."""
        config = Mock()
        config.ollama.base_url = "http://localhost:11434"
        
        with patch('pub_analysis_agent.config.cli.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'models': [{'name': 'llama2'}]}
            mock_get.return_value = mock_response
            
            result = test_ollama_connection(config)
            assert result['status'] == 'success'
    
    def test_test_lmstudio_connection_success(self):
        """Test successful LM Studio connection test."""
        config = Mock()
        config.lmstudio.base_url = "http://localhost:1234"
        
        with patch('pub_analysis_agent.config.cli.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'choices': [{'message': {'content': 'test'}}]}
            mock_post.return_value = mock_response
            
            result = test_lmstudio_connection(config)
            assert result['status'] == 'success'
    
    def test_display_execution_status(self):
        """Test execution status display."""
        execution_id = "test_execution_123"
        
        with patch('pub_analysis_agent.config.cli.console.print') as mock_print:
            _display_execution_status(execution_id)
            mock_print.assert_called()


if __name__ == '__main__':
    pytest.main([__file__])
