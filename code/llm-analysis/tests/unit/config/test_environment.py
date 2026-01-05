"""
Unit tests for environment configuration module.
"""

import pytest
import os
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from pub_analysis_agent.config.environment import (
    get_environment_manager,
    setup_environment,
    EnvironmentManager,
    EnvironmentType,
    ValidationLevel,
    EnvironmentVariable,
    EnvironmentSchema
)


class TestEnvironmentManager:
    """Test EnvironmentManager class."""
    
    @pytest.fixture
    def env_manager(self):
        """Create an EnvironmentManager instance."""
        return EnvironmentManager()
    
    def test_environment_manager_initialization(self, env_manager):
        """Test EnvironmentManager initialization."""
        assert env_manager is not None
        assert hasattr(env_manager, 'config')
    
    def test_load_environment_variables(self, env_manager):
        """Test loading environment variables."""
        test_env = {
            'TEST_VAR': 'test_value',
            'ANOTHER_VAR': 'another_value'
        }
        
        with patch.dict(os.environ, test_env):
            env_manager.load_environment_variables()
            assert 'TEST_VAR' in env_manager.config
            assert env_manager.config['TEST_VAR'] == 'test_value'
    
    def test_get_environment_variable_exists(self, env_manager):
        """Test getting existing environment variable."""
        with patch.dict(os.environ, {'TEST_VAR': 'test_value'}):
            result = env_manager.get_environment_variable('TEST_VAR')
            assert result == 'test_value'
    
    def test_get_environment_variable_not_exists(self, env_manager):
        """Test getting non-existing environment variable."""
        with patch.dict(os.environ, {}, clear=True):
            result = env_manager.get_environment_variable('NONEXISTENT_VAR', default='default_value')
            assert result == 'default_value'
    
    def test_set_environment_variable(self, env_manager):
        """Test setting environment variable."""
        with patch.dict(os.environ, {}, clear=True):
            env_manager.set_environment_variable('TEST_VAR', 'test_value')
            assert os.environ['TEST_VAR'] == 'test_value'
    
    def test_load_dotenv_file_success(self, env_manager):
        """Test successful loading of .env file."""
        env_content = """
        TEST_VAR=test_value
        ANOTHER_VAR=another_value
        NUMBER_VAR=42
        """
        
        with patch('builtins.open', mock_open(read_data=env_content)):
            with patch('pub_analysis_agent.config.environment.os.environ', {}):
                result = env_manager.load_dotenv_file('.env')
                assert result is True
                assert 'TEST_VAR' in env_manager.config
                assert env_manager.config['TEST_VAR'] == 'test_value'
    
    def test_load_dotenv_file_not_found(self, env_manager):
        """Test loading .env file that doesn't exist."""
        with patch('builtins.open', side_effect=FileNotFoundError):
            result = env_manager.load_dotenv_file('.env')
            assert result is False
    
    def test_validate_environment_success(self, env_manager):
        """Test successful environment validation."""
        required_vars = ['TEST_VAR', 'ANOTHER_VAR']
        test_env = {
            'TEST_VAR': 'test_value',
            'ANOTHER_VAR': 'another_value'
        }
        
        with patch.dict(os.environ, test_env):
            result = env_manager.validate_environment(required_vars)
            assert result is True
    
    def test_validate_environment_missing_vars(self, env_manager):
        """Test environment validation with missing variables."""
        required_vars = ['TEST_VAR', 'MISSING_VAR']
        test_env = {'TEST_VAR': 'test_value'}
        
        with patch.dict(os.environ, test_env):
            result = env_manager.validate_environment(required_vars)
            assert result is False


class TestEnvironmentClasses:
    """Test environment-related classes."""
    
    def test_environment_type_enum(self):
        """Test EnvironmentType enum values."""
        assert EnvironmentType.DEVELOPMENT == 'development'
        assert EnvironmentType.STAGING == 'staging'
        assert EnvironmentType.PRODUCTION == 'production'
    
    def test_validation_level_enum(self):
        """Test ValidationLevel enum values."""
        assert ValidationLevel.LOW == 'low'
        assert ValidationLevel.MEDIUM == 'medium'
        assert ValidationLevel.HIGH == 'high'
    
    def test_environment_variable_creation(self):
        """Test EnvironmentVariable creation."""
        env_var = EnvironmentVariable(
            name='TEST_VAR',
            value='test_value',
            required=True,
            description='Test variable'
        )
        
        assert env_var.name == 'TEST_VAR'
        assert env_var.value == 'test_value'
        assert env_var.required is True
        assert env_var.description == 'Test variable'
    
    def test_environment_schema_creation(self):
        """Test EnvironmentSchema creation."""
        schema = EnvironmentSchema(
            name='test_schema',
            description='Test environment schema',
            variables=[
                EnvironmentVariable(name='VAR1', value='value1', required=True),
                EnvironmentVariable(name='VAR2', value='value2', required=False)
            ]
        )
        
        assert schema.name == 'test_schema'
        assert schema.description == 'Test environment schema'
        assert len(schema.variables) == 2
        assert schema.variables[0].name == 'VAR1'
        assert schema.variables[1].name == 'VAR2'


class TestEnvironmentFunctions:
    """Test environment helper functions."""
    
    def test_get_environment_manager(self):
        """Test getting environment manager instance."""
        manager = get_environment_manager()
        assert isinstance(manager, EnvironmentManager)
    
    def test_setup_environment(self):
        """Test environment setup."""
        with patch('pub_analysis_agent.config.environment.EnvironmentManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            setup_environment()
            mock_manager.setup.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])
