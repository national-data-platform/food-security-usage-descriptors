"""
Unit tests for configuration loader module.
"""

import pytest
import yaml
import json
import os
from unittest.mock import Mock, patch, mock_open, MagicMock
from pathlib import Path
from typing import Dict, Any

from pub_analysis_agent.config.loader import (
    ConfigurationLoader,
    validate_configuration_file,
    get_configuration_loader,
    load_settings_from_file
)


class TestConfigurationLoader:
    """Test ConfigurationLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Create a ConfigurationLoader instance."""
        return ConfigurationLoader()
    
    @pytest.fixture
    def sample_yaml_config(self):
        """Sample YAML configuration."""
        return {
            'app': {
                'name': 'TestApp',
                'version': '1.0.0',
                'debug': True
            },
            'database': {
                'host': 'localhost',
                'port': 27017,
                'name': 'test_db'
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 4
            }
        }
    
    def test_loader_initialization(self, loader):
        """Test ConfigurationLoader initialization."""
        assert loader is not None
        assert hasattr(loader, 'config_path')
        assert hasattr(loader, '_yaml_config')
    
    def test_load_yaml_config_success(self, loader, sample_yaml_config):
        """Test successful YAML configuration loading."""
        yaml_content = yaml.dump(sample_yaml_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('pub_analysis_agent.config.loader.yaml.safe_load') as mock_yaml_load:
                mock_yaml_load.return_value = sample_yaml_config
                
                # Mock the config_path to exist
                loader.config_path = Path('config.yaml')
                
                result = loader.load_yaml_config()
                assert result == sample_yaml_config
    
    def test_load_yaml_config_no_path(self, loader):
        """Test YAML configuration loading when no path is set."""
        loader.config_path = None
        result = loader.load_yaml_config()
        assert result == {}
    
    def test_load_yaml_config_file_not_found(self, loader):
        """Test YAML configuration loading when file doesn't exist."""
        with patch('builtins.open', side_effect=FileNotFoundError):
            loader.config_path = Path('nonexistent.yaml')
            with pytest.raises(ValueError):
                loader.load_yaml_config()
    
    def test_load_yaml_config_invalid_yaml(self, loader):
        """Test YAML configuration loading with invalid YAML content."""
        with patch('builtins.open', mock_open(read_data="invalid: yaml: content: [")):
            with patch('pub_analysis_agent.config.loader.yaml.safe_load', side_effect=yaml.YAMLError):
                loader.config_path = Path('invalid.yaml')
                with pytest.raises(ValueError):
                    loader.load_yaml_config()
    
    def test_merge_config_sources(self, loader):
        """Test merging configuration from multiple sources."""
        yaml_config = {'app': {'name': 'TestApp'}}
        loader._yaml_config = yaml_config
        
        with patch.object(loader, '_extract_env_overrides') as mock_env:
            mock_env.return_value = {'app': {'debug': True}}
            
            result = loader.merge_config_sources()
            assert 'app' in result
            assert result['app']['name'] == 'TestApp'
            assert result['app']['debug'] is True
    
    def test_extract_env_overrides(self, loader):
        """Test environment variable override extraction."""
        with patch.dict(os.environ, {'MONGODB_CONNECTION_STRING': 'mongodb://localhost:27017'}):
            overrides = loader._extract_env_overrides()
            assert 'database' in overrides
            assert overrides['database']['connection_string'] == 'mongodb://localhost:27017'
    
    def test_deep_merge(self, loader):
        """Test deep merging of configuration dictionaries."""
        base = {'app': {'name': 'App1', 'version': '1.0'}}
        override = {'app': {'name': 'App2'}, 'database': {'host': 'localhost'}}
        
        result = loader._deep_merge(base, override)
        
        assert result['app']['name'] == 'App2'  # Override
        assert result['app']['version'] == '1.0'  # Preserved
        assert result['database']['host'] == 'localhost'  # Added
    
    def test_create_settings(self, loader, sample_yaml_config):
        """Test creating Settings instance."""
        loader._yaml_config = sample_yaml_config
        
        with patch.object(loader, '_extract_env_overrides') as mock_env:
            mock_env.return_value = {}
            
            with patch('pub_analysis_agent.config.loader.Settings') as mock_settings_class:
                mock_settings = Mock()
                mock_settings_class.return_value = mock_settings
                
                result = loader.create_settings()
                assert result == mock_settings
    
    def test_validate_configuration_success(self, loader, sample_yaml_config):
        """Test successful configuration validation."""
        loader._yaml_config = sample_yaml_config
        
        with patch.object(loader, '_extract_env_overrides') as mock_env:
            mock_env.return_value = {}
            
            with patch.object(loader, 'create_settings') as mock_create:
                mock_settings = Mock()
                mock_create.return_value = mock_settings
                
                result = loader.validate_configuration()
                assert result['valid'] is True
                assert result['settings'] == mock_settings
                assert 'error' not in result
    
    def test_validate_configuration_failure(self, loader):
        """Test configuration validation failure."""
        loader._yaml_config = {}
        
        with patch.object(loader, '_extract_env_overrides') as mock_env:
            mock_env.return_value = {}
            
            with patch.object(loader, 'create_settings', side_effect=Exception("Test error")):
                result = loader.validate_configuration()
                assert result['valid'] is False
                assert result['error'] == 'Test error'
                assert result['settings'] is None


class TestConfigurationFunctions:
    """Test standalone configuration functions."""
    
    def test_validate_configuration_file_yaml_success(self):
        """Test successful YAML configuration file validation."""
        valid_yaml = """
        app:
          name: TestApp
          version: 1.0.0
        database:
          host: localhost
          port: 27017
        """
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=valid_yaml)):
                with patch('pub_analysis_agent.config.loader.yaml.safe_load') as mock_yaml_load:
                    mock_yaml_load.return_value = yaml.safe_load(valid_yaml)
                    
                    result = validate_configuration_file('config.yaml')
                    assert result['valid'] is True
    
    def test_validate_configuration_file_yaml_invalid(self):
        """Test invalid YAML configuration file validation."""
        invalid_yaml = "invalid: yaml: content: ["
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=invalid_yaml)):
                with patch('pub_analysis_agent.config.loader.yaml.safe_load', side_effect=yaml.YAMLError):
                    result = validate_configuration_file('config.yaml')
                    assert result['valid'] is False
    
    def test_get_configuration_loader(self):
        """Test getting configuration loader instance."""
        loader = get_configuration_loader()
        assert isinstance(loader, ConfigurationLoader)
    
    def test_get_configuration_loader_with_path(self):
        """Test getting configuration loader with specific path."""
        config_path = "/path/to/config"
        
        with patch('pathlib.Path.exists', return_value=True):
            loader = get_configuration_loader(config_path)
            assert isinstance(loader, ConfigurationLoader)
    
    def test_load_settings_from_file(self):
        """Test loading settings from file."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="app:\n  name: TestApp")):
                with patch('pub_analysis_agent.config.loader.yaml.safe_load') as mock_yaml_load:
                    mock_yaml_load.return_value = {'app': {'name': 'TestApp'}}
                    
                    with patch('pub_analysis_agent.config.loader.Settings') as mock_settings_class:
                        mock_settings = Mock()
                        mock_settings_class.return_value = mock_settings
                        
                        result = load_settings_from_file('config.yaml')
                        assert result == mock_settings
    
    def test_load_settings_from_file_with_path(self):
        """Test loading settings from file with specific path."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="app:\n  name: TestApp")):
                with patch('pub_analysis_agent.config.loader.yaml.safe_load') as mock_yaml_load:
                    mock_yaml_load.return_value = {'app': {'name': 'TestApp'}}
                    
                    with patch('pub_analysis_agent.config.loader.Settings') as mock_settings_class:
                        mock_settings = Mock()
                        mock_settings_class.return_value = mock_settings
                        
                        result = load_settings_from_file('/path/to/config.yaml')
                        assert result == mock_settings


if __name__ == '__main__':
    pytest.main([__file__])
