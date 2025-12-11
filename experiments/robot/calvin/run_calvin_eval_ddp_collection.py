"""
Calvin evaluation with trajectory data collection.
Properly handles action chunks and saves embeddings for each env.step interaction.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from collections import deque, Counter
import copy
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union, Dict, List, Any
import draccus

# Setup paths
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import dependencies
import torch
from tqdm.auto import tqdm
import hydra
from omegaconf import OmegaConf
from termcolor import colored
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
from pytorch_lightning import seed_everything
from safetensors.torch import save_file, load_file

# Calvin imports
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)

logger = logging.getLogger(__name__)

# Environment setup
os.environ["NCCL_TIMEOUT"] = '0'
os.environ["FFMPEG_BINARY"] = "auto-detect"
os.environ["DISABLE_FLASH_ATTN"] = "1"
CALVIN_ROOT = os.environ.get('CALVIN_ROOT', '/home/yjh/calvin')

# ============================================================================
# Data Collection Classes
# ============================================================================

class TrajectoryCollector:
    """Manages collection of trajectory data with proper handling of action chunks."""
    
    def __init__(self, save_dir: str, model_name: str, batch_size: int = 50):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Storage for all subtask trajectories
        self.all_subtask_data = {
            'embeddings': [],  # List of step embeddings for each subtask
            'metadata': []     # Metadata for each subtask
        }
        
        # Current subtask being collected
        self.current_subtask_steps = []
        self.batch_num = 0
        self.sequence_counter = 0
        
    def start_subtask(self, subtask_name: str, sequence_id: int, subtask_idx: int):
        """Initialize collection for a new subtask."""
        self.current_subtask_steps = []
        self.current_metadata = {
            'subtask_name': subtask_name,
            'sequence_id': sequence_id,
            'subtask_idx': subtask_idx,
            'start_time': time.time()
        }
        
    def add_step(self, 
                 action: np.ndarray,
                 obs: Dict[str, Any],
                 model_output: Dict[str, Any],
                 embeddings: Dict[str, torch.Tensor]):
        """
        Add a single env.step interaction to current subtask.
        
        Args:
            action: The action that was executed
            obs: Observation after executing the action
            model_output: Raw output from model (before action extraction)
            embeddings: All embeddings from model forward pass
        """
        step_data = {
            # Raw inputs/outputs
            'action': torch.tensor(action, dtype=torch.float16),
            # 'obs_eef_pos': torch.tensor(obs.get('robot_obs', [])[:3], dtype=torch.float32),
            # 'obs_eef_quat': torch.tensor(obs.get('robot_obs', [])[3:7], dtype=torch.float32),
            # 'obs_gripper': torch.tensor(obs.get('robot_obs', [])[7:9], dtype=torch.float32),
            # "rgb_obs": torch.tensor(obs["rgb_obs"]['rgb_static'], dtype=torch.float16),
            # Embeddings - ensure all are torch tensors
            **{k: v.detach().cpu().to(torch.float16) if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float16) 
               for k, v in embeddings.items()}
        }
        
        self.current_subtask_steps.append(step_data)
        
    def end_subtask(self, success: bool, termination_reason: str = 'timeout'):
        """Finalize and store the current subtask trajectory."""
        if not self.current_subtask_steps:
            logger.warning("Ending subtask with no collected steps")
            return
            
        # Update metadata
        self.current_metadata.update({
            'success': success,
            'termination_reason': termination_reason,
            'num_steps': len(self.current_subtask_steps),
            'duration': time.time() - self.current_metadata['start_time']
        })
        
        # Store the subtask
        self.all_subtask_data['embeddings'].append(self.current_subtask_steps)
        self.all_subtask_data['metadata'].append(self.current_metadata)
        
        # Check if we should save a batch
        self.sequence_counter += 1
        if self.sequence_counter % self.batch_size == 0:
            self.save_batch()
            
    def save_batch(self):
        """Save current batch of trajectories."""
        if not self.all_subtask_data['embeddings']:
            return
            
        batch_filename = f"{self.model_name}_batch_{self.batch_num}"
        logger.info(f"Saving batch {self.batch_num} with {len(self.all_subtask_data['embeddings'])} subtasks...")
        
        self._save_trajectories(self.all_subtask_data, batch_filename)
        
        # Reset for next batch
        self.all_subtask_data = {'embeddings': [], 'metadata': []}
        self.batch_num += 1
        
    def finalize(self):
        """Save any remaining data and create master summary."""
        if self.all_subtask_data['embeddings']:
            self.save_batch()
            
        # Create master summary
        self._create_master_summary()
        
    def _save_trajectories(self, data: Dict, filename: str):
        """Save trajectory data in SafeTensors format."""
        processed_data = self._process_trajectory_data(data)
        
        # Save tensor data
        tensor_path = self.save_dir / f"{filename}.safetensors"
        tensor_data = {k: v for k, v in processed_data.items() 
                      if isinstance(v, torch.Tensor)}
        save_file(tensor_data, str(tensor_path))
        print(tensor_path)
        
        # Save metadata
        meta_path = self.save_dir / f"{filename}_metadata.json"
        metadata = {
            'subtask_mapping': processed_data.get('subtask_mapping', {}),
            'summary': {
                'total_subtasks': len(data['metadata']),
                'successful_subtasks': sum(1 for m in data['metadata'] if m['success']),
                'unique_subtasks': len(set(m['subtask_name'] for m in data['metadata'])),
                'shapes': {k: list(v.shape) for k, v in tensor_data.items()}
            }
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Saved to {tensor_path}")
        
    def _process_trajectory_data(self, data: Dict) -> Dict:
        """Process raw trajectory data into tensors."""
        result = {}
        
        # Track global indices
        current_idx = 0
        trajectory_info = {
            'start_indices': [],
            'lengths': [],
            'success': [],
            'subtask_names': [],
            'sequence_ids': [],
            'subtask_indices': []
        }
        
        # Collect all embedding keys from first step
        if data['embeddings'] and data['embeddings'][0]:
            embedding_keys = data['embeddings'][0][0].keys()
            for key in embedding_keys:
                result[key] = []
        # Process each subtask trajectory
        for traj_steps, traj_meta in zip(data['embeddings'], data['metadata']):
            if not traj_steps:
                continue
                
            # Record trajectory info
            trajectory_info['start_indices'].append(current_idx)
            trajectory_info['lengths'].append(len(traj_steps))
            trajectory_info['success'].append(1 if traj_meta['success'] else 0)
            trajectory_info['subtask_names'].append(traj_meta['subtask_name'])
            trajectory_info['sequence_ids'].append(traj_meta['sequence_id'])
            trajectory_info['subtask_indices'].append(traj_meta['subtask_idx'])
            
            # Stack embeddings for this trajectory
            for key in embedding_keys:
                if key in result:
                    # Collect all steps for this key
                    step_tensors = [step[key] for step in traj_steps if key in step]
                    if step_tensors:
                        shapes = [t.shape for t in step_tensors]
                        if len(set(shapes)) > 1:
                            logger.warning(f"Variable shapes for key {key} in trajectory, skipping.")
                            continue    
                        # Stack along time dimension
                        stacked = torch.stack(step_tensors, dim=0)
                        result[key].append(stacked)
                        
            current_idx += len(traj_steps)
            
        # Concatenate all trajectories
        for key in list(result.keys()):
            if result[key] and all(isinstance(t, torch.Tensor) for t in result[key]):
                result[key] = torch.cat(result[key], dim=0)
            else:
                del result[key]  # Remove if empty or invalid
                
        # Add trajectory metadata as tensors
        result['trajectory_start_indices'] = torch.tensor(trajectory_info['start_indices'], dtype=torch.long)
        result['trajectory_lengths'] = torch.tensor(trajectory_info['lengths'], dtype=torch.long)
        result['trajectory_success'] = torch.tensor(trajectory_info['success'], dtype=torch.long)
        result['sequence_ids'] = torch.tensor(trajectory_info['sequence_ids'], dtype=torch.long)
        result['subtask_indices'] = torch.tensor(trajectory_info['subtask_indices'], dtype=torch.long)
        
        # Create subtask name mapping
        unique_names = list(set(trajectory_info['subtask_names']))
        name_to_id = {name: i for i, name in enumerate(unique_names)}
        result['subtask_name_ids'] = torch.tensor(
            [name_to_id[name] for name in trajectory_info['subtask_names']], 
            dtype=torch.long
        )
        result['subtask_mapping'] = {
            'names': unique_names,
            'name_to_id': name_to_id,
            'id_to_name': {str(v): k for k, v in name_to_id.items()}
        }
        
        return result
        
    def _create_master_summary(self):
        """Create a master summary file listing all batches."""
        summary = {
            'model': self.model_name,
            'num_batches': self.batch_num,
            'batch_size': self.batch_size,
            'batch_files': [f"{self.model_name}_batch_{i}.safetensors" 
                          for i in range(self.batch_num)]
        }
        
        summary_path = self.save_dir / f"{self.model_name}_master_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Master summary saved to {summary_path}")

# ============================================================================
# Evaluation Functions with Data Collection
# ============================================================================

def evaluate_policy_with_collection(
    model, 
    env, 
    collector: TrajectoryCollector,
    num_sequences: int,
    ep_len: int,
    num_procs: int,
    procs_id: int,
    debug: bool = False
):
    """Evaluate policy while collecting trajectory data."""
    
    # Setup Calvin task oracle
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    
    # Get sequences for this process
    eval_sequences = get_sequences(num_sequences)
    num_seq_per_proc = num_sequences // num_procs
    start_idx = num_seq_per_proc * procs_id
    end_idx = num_seq_per_proc * (procs_id + 1)
    eval_sequences = eval_sequences[start_idx:end_idx]
    
    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, desc="Sequences", leave=False)
        
    for seq_idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
        global_seq_id = start_idx + seq_idx
        
        if debug:
            print(f"\n=== Sequence {global_seq_id}: {' -> '.join(eval_sequence)} ===")
            
        # Evaluate sequence
        result = evaluate_sequence_with_collection(
            env, model, task_oracle, collector,
            initial_state, eval_sequence, val_annotations,
            global_seq_id, ep_len, debug
        )
        results.append(result)
        
        if not debug:
            success_list = count_success(results)
            eval_sequences.set_description(
                " ".join([f"{i+1}/5: {v*100:.1f}%" for i, v in enumerate(success_list)])
            )
            
    return results

def evaluate_sequence_with_collection(
    env, model, task_oracle, collector: TrajectoryCollector,
    initial_state, eval_sequence, val_annotations,
    sequence_id: int, ep_len: int, debug: bool
):
    """Evaluate a sequence of subtasks with data collection."""
    
    # Reset environment
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    
    success_counter = 0
    
    for subtask_idx, subtask in enumerate(eval_sequence):
        if debug:
            print(f"  Subtask {subtask_idx}: {subtask}", end=" ... ")
            
        # Start collecting for this subtask
        collector.start_subtask(subtask, sequence_id, subtask_idx)
        
        # Execute subtask
        success = rollout_with_collection(
            env, model, task_oracle, collector,
            subtask, val_annotations, ep_len, debug
        )
        
        # End collection for this subtask
        collector.end_subtask(success, 'success' if success else 'timeout')
        
        if success:
            success_counter += 1
            if debug:
                print(colored("SUCCESS", "green"))
        else:
            if debug:
                print(colored("FAILED", "red"))
            break  # Stop sequence on failure
            
    return success_counter

def rollout_with_collection(
    env, model, task_oracle, collector: TrajectoryCollector,
    subtask: str, val_annotations, ep_len: int, debug: bool
):
    """
    Execute a subtask with proper handling of action chunks.
    Collects embeddings for EACH env.step interaction.
    """
    
    # Get initial observation and task
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()
    
    # Action queue for handling chunks
    action_queue = deque()
    
    for step in range(ep_len):
        # Get new action chunk if queue is empty
        if len(action_queue) == 0:
            # Model inference - returns chunk of actions
            model_output = model.step(obs, lang_annotation, step)
            
            # Extract embeddings from model output
            embeddings = model_output.get('embeddings', {})
            
            # Get action chunk and add to queue
            # if isinstance(model_output, dict) and 'actions' in model_output:
            #     actions = model_output['actions']
            # elif isinstance(model_output, list):
            #     actions = model_output
            # else:
            #     actions = [model_output]
            actions = model_output['action']    
            action_queue.extend(actions)
            
            # Store embeddings for the FIRST action in chunk
            # (We'll update for subsequent actions below)
            current_embeddings = embeddings
        
        # Get next action from queue
        action = action_queue.popleft()
        
        # Normalize gripper action
        if action[-1] < 0:
            action[-1] = -1
        else:
            action[-1] = 1
            
        # Execute action in environment
        obs, _, _, current_info = env.step(action)
        
        # CRITICAL: Collect data for THIS specific env.step
        collector.add_step(
            action=action,
            obs=obs,
            model_output={'action': action},  # Simplified for this step
            embeddings=current_embeddings
        )
        
        # For subsequent actions in chunk, we may want to update embeddings
        # based on new observations (depends on your model architecture)
        if len(action_queue) > 0 and hasattr(model, 'update_embeddings'):
            # Optional: Update embeddings with new observation
            current_embeddings = model.update_embeddings(obs, current_embeddings)
        
        # Check if subtask is completed
        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {subtask}
        )
        
        if len(current_task_info) > 0:
            return True  # Success
            
    return False  # Timeout

def extract_embeddings(model_output: Any) -> Dict[str, torch.Tensor]:
    """
    Extract embeddings from model output.
    This needs to be adapted based on your specific model architecture.
    """
    embeddings = {}
    
    if isinstance(model_output, dict):
        # Extract various embedding types based on your model
        embedding_keys = [
            'lang_embeddings', 'obs_embeddings', 'state_embeddings',
            'patch_embeddings', 'hand_patch_embeddings', 
            'hand_obs_embeddings', 'Y', 'Y_hat', 'Z'
        ]
        
        for key in embedding_keys:
            if key in model_output:
                embeddings[key] = model_output[key]
                
    # Ensure all embeddings are tensors
    for key, value in embeddings.items():
        if not isinstance(value, torch.Tensor):
            embeddings[key] = torch.tensor(value)
            
    return embeddings

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EvalConfig:
    # Model parameters
    model_family: str = "openvla"
    pretrained_checkpoint: str = "/home/yjh/.cache/huggingface/hub/models--qwbu--univla-7b-224-sft-calvin/snapshots/1bba95d6209c1eb03d87b2bdb9b71fe2420c5413/"
    my_config_path: str = "configs.json"
    load_in_8bit: bool = False                       # Load with 8-bit quantization
    load_in_4bit: bool = False                       # Load with 4-bit quantization
    center_crop: bool = False                        # Center crop? (if trained w/ random crop image aug)
    # Calvin parameters
    calvin_root: str = "/home/yjh/UniVLA/fake_dataset"
    task_suite_name: str = "calvin"                  # Task suite.
    num_sequences: int = 1000
    ep_len: int = 360
    unnorm_key: str = "calvin"
    
    # Data collection
    save_dir: str = "./trajectory_embeddings"
    batch_size: int = 50
    collect_embeddings: bool = True
    
    # Execution
    seed: int = 7
    debug: bool = False
    use_wandb: bool = False
    window_size: int = 12
    action_decoder_path: str = "/home/yjh/.cache/huggingface/hub/models--qwbu--univla-7b-224-sft-calvin/snapshots/1bba95d6209c1eb03d87b2bdb9b71fe2420c5413/action_decoder.pt"

# ============================================================================
# Main Entry Point
# ============================================================================

@draccus.wrap()
def main(cfg: EvalConfig) -> None:
    """Main evaluation function with trajectory collection."""
    
    # Set seed
    seed_everything(cfg.seed)
    
    # Initialize accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    acc = Accelerator(kwargs_handlers=[kwargs])
    device = acc.device
    
    # Load model configuration
    # with open(cfg.my_config_path, 'r') as f:
    #     model_cfg = json.load(f)
        
    # Initialize model (adapt this to your model)
    from experiments.robot.calvin.calvin_model import WrappedModel, WrappedCalvinEvaluation
    model = WrappedModel(cfg)
    model = acc.prepare(model)
    model.eval()
    
    # Setup environment
    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'],
        'depth_obs': [],
        'state_obs': ['robot_obs'],
        'actions': ['rel_actions'],
        'language': ['language']
    }
    
    from experiments.robot.calvin.calvin_env_wrapper import CalvinEnvWrapperRaw
    val_folder = Path(cfg.calvin_root) / "validation"
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    
    # Create evaluation wrapper
    eva = WrappedCalvinEvaluation(cfg, model)
    
    # Initialize trajectory collector
    model_name = "univla_abc_d"
    collector = TrajectoryCollector(
        save_dir=cfg.save_dir,
        model_name=model_name,
        batch_size=cfg.batch_size
    )
    
    # Run evaluation with collection
    results = evaluate_policy_with_collection(
        eva, env, collector,
        num_sequences=cfg.num_sequences,
        ep_len=cfg.ep_len,
        num_procs=acc.num_processes,
        procs_id=acc.process_index,
        debug=cfg.debug
    )
    
    # Finalize collection
    collector.finalize()
    
    # Compute and report metrics
    results_tensor = torch.tensor(results, dtype=torch.float, device=device)
    acc.wait_for_everyone()
    global_results = acc.gather_for_metrics(results_tensor)
    
    if acc.is_main_process:
        avg_success = global_results.mean().item()
        success_rates = count_success(global_results.tolist())
        
        print("\n" + "="*50)
        print(f"Average success: {avg_success:.3f}")
        print("Success rates by length:")
        for i, sr in enumerate(success_rates):
            print(f"  {i+1} tasks: {sr*100:.1f}%")
        print("="*50)
        
        # Save final summary
        summary_path = Path(cfg.save_dir) / f"{model_name}_final_results.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'avg_success': avg_success,
                'success_rates': success_rates,
                'total_sequences': cfg.num_sequences
            }, f, indent=2)

if __name__ == "__main__":
    main()