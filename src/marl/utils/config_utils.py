"""
Configuration utilities for multi-agent reinforcement learning.
"""

import json
import yaml
from typing import Dict, Any, Optional
import os
from dataclasses import dataclass, asdict


@dataclass
class EnvironmentConfig:
    """Configuration for environment."""
    env_type: str = "grid_world"
    grid_size: tuple = (10, 10)
    n_agents: int = 2
    n_targets: int = 3
    max_steps: int = 100
    reward_scale: float = 1.0
    collision_penalty: float = -0.1
    target_reward: float = 10.0
    step_penalty: float = -0.01


@dataclass
class AgentConfig:
    """Configuration for agents."""
    agent_type: str = "dqn"
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 100
    hidden_dims: list = None
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128]


@dataclass
class TrainingConfig:
    """Configuration for training."""
    n_episodes: int = 1000
    max_steps_per_episode: int = 100
    eval_frequency: int = 100
    eval_episodes: int = 10
    save_frequency: int = 500
    device: str = "cpu"
    seed: int = 42


@dataclass
class AlgorithmConfig:
    """Configuration for algorithms."""
    algorithm_type: str = "independent_q_learning"
    # MADDPG specific
    tau: float = 0.005
    # MAPPO specific
    lambda_gae: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    environment: EnvironmentConfig
    agent: AgentConfig
    training: TrainingConfig
    algorithm: AlgorithmConfig
    experiment_name: str = "default_experiment"
    output_dir: str = "results"


class ConfigUtils:
    """Utility class for managing configurations."""
    
    @staticmethod
    def load_config(filepath: str) -> ExperimentConfig:
        """Load configuration from JSON or YAML file."""
        with open(filepath, 'r') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return ConfigUtils.dict_to_config(data)
    
    @staticmethod
    def save_config(config: ExperimentConfig, filepath: str) -> None:
        """Save configuration to JSON or YAML file."""
        data = ConfigUtils.config_to_dict(config)
        
        with open(filepath, 'w') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                yaml.dump(data, f, default_flow_style=False, indent=2)
            else:
                json.dump(data, f, indent=2)
    
    @staticmethod
    def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(config)
    
    @staticmethod
    def dict_to_config(data: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to configuration."""
        # Handle nested configurations
        environment = EnvironmentConfig(**data.get('environment', {}))
        agent = AgentConfig(**data.get('agent', {}))
        training = TrainingConfig(**data.get('training', {}))
        algorithm = AlgorithmConfig(**data.get('algorithm', {}))
        
        return ExperimentConfig(
            environment=environment,
            agent=agent,
            training=training,
            algorithm=algorithm,
            experiment_name=data.get('experiment_name', 'default_experiment'),
            output_dir=data.get('output_dir', 'results')
        )
    
    @staticmethod
    def create_default_config() -> ExperimentConfig:
        """Create default configuration."""
        return ExperimentConfig(
            environment=EnvironmentConfig(),
            agent=AgentConfig(),
            training=TrainingConfig(),
            algorithm=AlgorithmConfig()
        )
    
    @staticmethod
    def create_config_variants(
        base_config: ExperimentConfig,
        variants: Dict[str, Dict[str, Any]]
    ) -> Dict[str, ExperimentConfig]:
        """Create multiple configuration variants."""
        configs = {}
        
        for variant_name, variant_params in variants.items():
            # Create a copy of base config
            config_dict = ConfigUtils.config_to_dict(base_config)
            
            # Apply variant parameters
            for key, value in variant_params.items():
                if '.' in key:
                    # Handle nested parameters (e.g., 'environment.n_agents')
                    parts = key.split('.')
                    if len(parts) == 2:
                        section, param = parts
                        if section in config_dict:
                            config_dict[section][param] = value
                else:
                    config_dict[key] = value
            
            # Create new config
            configs[variant_name] = ConfigUtils.dict_to_config(config_dict)
        
        return configs
    
    @staticmethod
    def validate_config(config: ExperimentConfig) -> bool:
        """Validate configuration parameters."""
        try:
            # Validate environment config
            assert config.environment.n_agents > 0, "Number of agents must be positive"
            assert config.environment.max_steps > 0, "Max steps must be positive"
            
            # Validate agent config
            assert 0 < config.agent.learning_rate < 1, "Learning rate must be between 0 and 1"
            assert 0 < config.agent.gamma < 1, "Gamma must be between 0 and 1"
            assert config.agent.batch_size > 0, "Batch size must be positive"
            
            # Validate training config
            assert config.training.n_episodes > 0, "Number of episodes must be positive"
            assert config.training.max_steps_per_episode > 0, "Max steps per episode must be positive"
            
            return True
        except AssertionError as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    @staticmethod
    def setup_experiment_directory(config: ExperimentConfig) -> str:
        """Setup experiment directory structure."""
        # Create main experiment directory
        exp_dir = os.path.join(config.output_dir, config.experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['models', 'logs', 'plots', 'results']
        for subdir in subdirs:
            os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(exp_dir, 'config.json')
        ConfigUtils.save_config(config, config_path)
        
        return exp_dir
    
    @staticmethod
    def get_config_summary(config: ExperimentConfig) -> str:
        """Get a summary of the configuration."""
        summary = f"""
Experiment Configuration Summary
================================

Experiment Name: {config.experiment_name}
Output Directory: {config.output_dir}

Environment:
  Type: {config.environment.env_type}
  Grid Size: {config.environment.grid_size}
  Agents: {config.environment.n_agents}
  Targets: {config.environment.n_targets}
  Max Steps: {config.environment.max_steps}

Agent:
  Type: {config.agent.agent_type}
  Learning Rate: {config.agent.learning_rate}
  Gamma: {config.agent.gamma}
  Batch Size: {config.agent.batch_size}

Training:
  Episodes: {config.training.n_episodes}
  Max Steps per Episode: {config.training.max_steps_per_episode}
  Device: {config.training.device}
  Seed: {config.training.seed}

Algorithm:
  Type: {config.algorithm.algorithm_type}
        """
        
        return summary.strip()