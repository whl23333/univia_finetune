"""
Fixed Calvin evaluation with proper trajectory boundaries and subtask information
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import copy
import random
import numpy as np
from moviepy.editor import ImageSequenceClip
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
from collections import Counter

# Import all necessary modules
import torch
from tqdm.auto import tqdm
import hydra
from omegaconf import OmegaConf
from termcolor import colored

# Import Calvin-specific modules
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)

# Import model-specific modules
import clip
import models.vision_transformer as vits
from models.gr1_version9 import GR1 
from PreProcess import PreProcess
from evaluation.calvin_evaluation import GR1CalvinEvaluation
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)
os.environ["FFMPEG_BINARY"] = "auto-detect"
CALVIN_ROOT = os.environ['CALVIN_ROOT']

# Global storage for all trajectories
all_trajectories_data = {
    'embeddings': [],  # List of dicts, one per trajectory
    'metadata': []     # List of metadata dicts
}
current_trajectory_embeddings = []
current_trajectory_id = None
current_subtask_boundaries = []
current_subtask_names = []
current_subtask_success = []


def set_seed_comprehensive(seed=42, verbose=False):
    """Set seed for maximum reproducibility across all components"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if verbose:
        print(f"Seed set to {seed}")
    
    try:
        from pytorch_lightning import seed_everything
        seed_everything(seed, workers=True)
    except ImportError:
        pass


def save_all_trajectories(all_data, save_path, model_name):
    """Save all trajectory data with proper boundaries and subtask information"""
    # Initialize processed data lists
    processed_data = {
        # Trajectory boundaries
        'trajectory_start_indices': [],
        'trajectory_lengths': [],
        # Subtask boundaries (global indices)
        'subtask_start_indices': [],
        'subtask_lengths': [],
        'subtask_success': [],
        'subtask_trajectory_id': [],
        'subtask_position_in_trajectory': [],
        # Success labels
        'trajectory_success': [],
        'trajectory_success_steps': [],
        # Embeddings will be collected per trajectory then concatenated at the end
        'Y': [],
        'Y_hat': [],
        'Z': [],
        'hand_obs_embeddings': [],
        'hand_patch_embeddings': [],
        'lang_embeddings': [],
        'obs_embeddings': [],
        'patch_embeddings': [],
        'state_embeddings': []
    }

    # For tracking subtask names
    subtask_name_to_id = {}
    subtask_names_list = []
    all_subtask_name_ids = []
    current_global_idx = 0

    # --- CORRECTION: Process each trajectory individually ---
    for traj_idx, (traj_embeddings, traj_metadata) in enumerate(zip(all_data['embeddings'], all_data['metadata'])):
        traj_length = len(traj_embeddings)

        # Handle empty trajectories gracefully
        if traj_length == 0:
            print(f"Warning: Found empty trajectory at index {traj_idx}. Skipping embedding concatenation.")
            # Still record metadata for consistency if needed, but embeddings remain empty
            # You might choose to skip recording empty trajectories entirely
            # depending on your downstream processing needs.
            # For now, let's record them with length 0 and start index at current_global_idx
            processed_data['trajectory_start_indices'].append(current_global_idx)
            processed_data['trajectory_lengths'].append(0)
            processed_data['trajectory_success'].append(0)
            processed_data['trajectory_success_steps'].append(0)
            # No embeddings to append for an empty trajectory
            # No subtasks to process for an empty trajectory
            continue # Move to the next trajectory

        # 1. Record Trajectory Metadata
        processed_data['trajectory_start_indices'].append(current_global_idx)
        processed_data['trajectory_lengths'].append(traj_length)
        processed_data['trajectory_success'].append(1 if traj_metadata.get('overall_success', False) else 0)
        processed_data['trajectory_success_steps'].append(traj_metadata.get('num_successful_steps', 0))

        # 2. Concatenate Embeddings for THIS trajectory and append to lists
        # This is the key fix: process per trajectory, not globally upfront
        traj_stacked = {}
        # Assuming all steps have the same keys
        embedding_keys = traj_embeddings[0].keys()
        for key in embedding_keys:
            # Collect tensors for this single trajectory
            tensors_list = [step[key] for step in traj_embeddings if key in step]
            if tensors_list: # Check if list is not empty before concatenating
                 # Concatenate along the step dimension (assuming dim 0)
                stacked_tensor = torch.cat(tensors_list, dim=0)
                processed_data[key].append(stacked_tensor)
            # If key is missing in all steps of this traj, we could append an empty tensor
            # or handle it based on requirements. For now, we just don't add it.

        # 3. Process Subtasks for THIS trajectory
        subtask_boundaries = traj_metadata.get('subtask_boundaries', [])
        subtask_names = traj_metadata.get('subtask_names', [])
        subtask_success = traj_metadata.get('subtask_success', [])

        # Ensure subtask data lengths match
        assert len(subtask_boundaries) == len(subtask_names) == len(subtask_success), \
            f"Mismatch in subtask data lengths for trajectory {traj_idx}"

        for subtask_idx, (subtask_name, (local_start, local_end), success) in enumerate(
            zip(subtask_names, subtask_boundaries, subtask_success)):

            # Convert local indices (within trajectory) to global indices
            global_start = current_global_idx + local_start
            global_end = current_global_idx + local_end

            processed_data['subtask_start_indices'].append(global_start)
            processed_data['subtask_lengths'].append(local_end - local_start)
            processed_data['subtask_success'].append(1 if success else 0) # Ensure int
            processed_data['subtask_trajectory_id'].append(traj_idx)
            processed_data['subtask_position_in_trajectory'].append(subtask_idx)

            # Handle subtask name mapping
            if subtask_name not in subtask_name_to_id:
                new_id = len(subtask_name_to_id) # Assign new ID
                subtask_name_to_id[subtask_name] = new_id
                subtask_names_list.append(subtask_name)
            # Append the ID corresponding to this subtask name
            all_subtask_name_ids.append(subtask_name_to_id[subtask_name])

        # 4. Update global index for the next trajectory
        current_global_idx += traj_length

    # --- CORRECTION: Finalize data structure ---
    # Concatenate all trajectory-level embeddings into single tensors
    final_data = {}
    for key in ['Y', 'Y_hat', 'Z', 'hand_obs_embeddings', 'hand_patch_embeddings',
                'lang_embeddings', 'obs_embeddings', 'patch_embeddings', 'state_embeddings']:
         # Only process keys that have data collected
        if processed_data[key] and all(isinstance(t, torch.Tensor) for t in processed_data[key]):
             # Concatenate list of trajectory tensors into one big tensor
            final_data[key] = torch.cat(processed_data[key], dim=0)
        elif processed_data[key]:
            print(f"Warning: Key '{key}' has non-tensor data in processed_data, skipping concatenation.")
        else:
             print(f"Info: Key '{key}' has no data, skipping.")

    # Convert list metadata to tensors
    # Trajectory metadata
    final_data['trajectory_start_indices'] = torch.tensor(processed_data['trajectory_start_indices'], dtype=torch.long)
    final_data['trajectory_lengths'] = torch.tensor(processed_data['trajectory_lengths'], dtype=torch.long)
    final_data['trajectory_success'] = torch.tensor(processed_data['trajectory_success'], dtype=torch.long)
    final_data['trajectory_success_steps'] = torch.tensor(processed_data['trajectory_success_steps'], dtype=torch.long)

    # Subtask metadata
    final_data['subtask_start_indices'] = torch.tensor(processed_data['subtask_start_indices'], dtype=torch.long)
    final_data['subtask_lengths'] = torch.tensor(processed_data['subtask_lengths'], dtype=torch.long)
    final_data['subtask_success'] = torch.tensor(processed_data['subtask_success'], dtype=torch.long)
    final_data['subtask_trajectory_id'] = torch.tensor(processed_data['subtask_trajectory_id'], dtype=torch.long)
    final_data['subtask_position_in_trajectory'] = torch.tensor(processed_data['subtask_position_in_trajectory'], dtype=torch.long)
    final_data['subtask_name_id'] = torch.tensor(all_subtask_name_ids, dtype=torch.long) # Convert list to tensor

    # Add subtask name mapping to final data for easy lookup
    final_data['subtask_mapping'] = {
        'subtask_names': subtask_names_list,
        'subtask_name_to_id': subtask_name_to_id,
        'id_to_subtask_name': {str(v): k for k, v in subtask_name_to_id.items()}
    }

    # --- Save files ---
    filename = f"{model_name}_all_trajectories.safetensors"
    filepath = os.path.join(save_path, filename)
    # Save main data (only tensor data)
    tensor_data = {k: v for k, v in final_data.items() if isinstance(v, torch.Tensor)}
    save_file(tensor_data, filepath)

    # Save subtask name mapping separately as JSON (since it contains non-tensor data)
    subtask_mapping_path = os.path.join(save_path, f"{model_name}_subtask_mapping.json")
    # Extract only the serializable parts for JSON
    mapping_for_json = {
        'subtask_names': final_data['subtask_mapping']['subtask_names'],
        'subtask_name_to_id': final_data['subtask_mapping']['subtask_name_to_id'],
        'id_to_subtask_name': final_data['subtask_mapping']['id_to_subtask_name']
    }
    with open(subtask_mapping_path, 'w') as f:
        json.dump(mapping_for_json, f, indent=2)

    # Save summary
    total_trajectories = len(processed_data['trajectory_lengths'])
    successful_trajectories = int(final_data['trajectory_success'].sum().item()) if 'trajectory_success' in final_data else 0
    total_steps = int(current_global_idx)
    total_subtasks = len(processed_data['subtask_start_indices'])
    successful_subtasks = int(sum(processed_data['subtask_success']))
    unique_subtask_types = len(subtask_name_to_id)

    summary = {
        'model': model_name,
        'total_trajectories': total_trajectories,
        'successful_trajectories': successful_trajectories,
        'failed_trajectories': total_trajectories - successful_trajectories,
        'total_steps': total_steps,
        'total_subtasks': total_subtasks,
        'successful_subtasks': successful_subtasks,
        'unique_subtask_types': unique_subtask_types,
        'embedding_shapes': {key: list(tensor.shape) for key, tensor in tensor_data.items()} # Use tensor_data for shapes
    }
    summary_path = os.path.join(save_path, f"{model_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved all trajectories to: {filepath}")
    print(f"Total trajectories: {summary['total_trajectories']}")
    print(f"Total steps across all trajectories: {summary['total_steps']}")
    print(f"Total subtasks: {summary['total_subtasks']}")
    print(f"Unique subtask types: {summary['unique_subtask_types']}")
    print(f"\nTrajectory lengths: {processed_data['trajectory_lengths'][:10]}..." if len(processed_data['trajectory_lengths']) > 10 else f"Trajectory lengths: {processed_data['trajectory_lengths']}")
    print(f"Trajectory start indices: {processed_data['trajectory_start_indices'][:10]}..." if len(processed_data['trajectory_start_indices']) > 10 else f"Trajectory start indices: {processed_data['trajectory_start_indices']}")

    return filepath

# Example usage (assuming all_trajectories_data is populated correctly):
# filepath = save_all_trajectories(all_trajectories_data, './output_embeddings/', 'GR1_test_model')
# Expected output for your example:
# Trajectory lengths: [50, 359, ...]
# Trajectory start indices: [0, 50, 409, ...] # 0 + 50 = 50, 50 + 359 = 409, etc.

# ... (previous imports and code remain largely the same until save_all_trajectories) ...

# Global storage for all SUBTASK "trajectories"
all_subtask_trajectories_data = {
    'embeddings': [],  # List of dicts, one per SUBTASK "trajectory"
    'metadata': []     # List of metadata dicts, one per SUBTASK "trajectory"
}

# Global variable to hold embeddings for the *current subtask* being rolled out
current_subtask_rollout_embeddings = []

def evaluate_policy_consistent(ckpt_path, model, env, eval_sr_path, eval_result_path,
                             ep_len, num_sequences, num_procs, procs_id,
                             eval_dir=None, debug=True, seed=42):
    """Evaluate policy with proper trajectory and subtask tracking, saving every 50 sequences"""
    # Use the new global structure
    global all_subtask_trajectories_data
    all_subtask_trajectories_data = {'embeddings': [], 'metadata': []} # Reset

    filename = os.path.basename(ckpt_path).replace('.pth', '')
    # Create save directory
    save_dir = os.path.join(eval_result_path, "trajectory_embeddings/")
    os.makedirs(save_dir, exist_ok=True)

    # Reset seed
    set_seed_comprehensive(seed, verbose=True)
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    eval_dir = get_log_dir(eval_dir)

    # Get sequences
    eval_sequences = get_deterministic_sequences(num_sequences, seed)

    # Distribute sequences
    num_seq_per_procs = num_sequences // num_procs
    start_idx = num_seq_per_procs * procs_id
    end_idx = num_seq_per_procs * (procs_id + 1)
    eval_sequences = eval_sequences[start_idx:end_idx]

    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    # Batch saving parameters
    BATCH_SIZE = 50
    batch_num = 0
    
    sequence_i = 0
    for initial_state, eval_sequence in tqdm(eval_sequences):
        print("--------sequence_i", sequence_i)
        sequence_seed = seed + sequence_i
        result = evaluate_sequence_consistent(
            env, model, task_oracle, initial_state, eval_sequence,
            val_annotations, debug, eval_dir, sequence_i, ep_len, sequence_seed,
            global_sequence_id=start_idx + sequence_i # Pass global sequence ID
        )
        results.append(result)

        if not debug:
            success_list = count_success(results)
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|"
            )
        
        # Check if we should save a batch
        if (sequence_i + 1) % BATCH_SIZE == 0:
            # Save current batch
            batch_filename = f"{filename}_batch_{batch_num}"
            print(f"\nSaving batch {batch_num} (sequences {sequence_i - BATCH_SIZE + 1} to {sequence_i})...")
            save_all_trajectories(all_subtask_trajectories_data, save_dir, batch_filename)
            
            # Reset for next batch
            all_subtask_trajectories_data = {'embeddings': [], 'metadata': []}
            batch_num += 1
            print(f"Batch {batch_num - 1} saved successfully. Continuing with next batch...\n")
        
        sequence_i += 1

    # Save any remaining sequences in the final batch
    if len(all_subtask_trajectories_data['embeddings']) > 0:
        batch_filename = f"{filename}_batch_{batch_num}"
        remaining_count = len(all_subtask_trajectories_data['embeddings'])
        print(f"\nSaving final batch {batch_num} ({remaining_count} remaining subtask trajectories)...")
        save_all_trajectories(all_subtask_trajectories_data, save_dir, batch_filename)
        print(f"Final batch saved successfully.\n")

    # Save evaluation summary
    print_and_save(filename, results, eval_sequences, eval_result_path, None)
    
    # Create a master summary file that lists all batch files
    master_summary = {
        'model': filename,
        'total_sequences': len(results),
        'num_batches': batch_num + 1 if len(all_subtask_trajectories_data['embeddings']) > 0 else batch_num,
        'batch_size': BATCH_SIZE,
        'batch_files': [f"{filename}_batch_{i}_all_subtask_trajectories.safetensors" 
                       for i in range(batch_num + (1 if len(all_subtask_trajectories_data['embeddings']) > 0 else 0))]
    }
    master_summary_path = os.path.join(save_dir, f"{filename}_master_summary.json")
    with open(master_summary_path, 'w') as f:
        json.dump(master_summary, f, indent=2)
    print(f"Master summary saved to: {master_summary_path}")
    
    return results

def evaluate_sequence_consistent(env, model, task_checker, initial_state, eval_sequence,
                               val_annotations, debug, eval_dir, sequence_i, ep_len, seed,
                               global_sequence_id):
    """Evaluate single sequence with subtask tracking, saving each subtask as a trajectory"""
    global all_subtask_trajectories_data # Access the global to store subtask data
    # No need for these globals anymore for sequence-level tracking:
    # global current_trajectory_embeddings, current_subtask_boundaries
    # global current_subtask_names, current_subtask_success

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    model.reset()
    success_counter = 0

    if True: # debug print block
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")

    for subtask_i, subtask in enumerate(tqdm(eval_sequence)):
        subtask_seed = seed + subtask_i * 1000

        # Record start index for this subtask (local to trajectory) - Not used for saving anymore
        # subtask_start = len(current_trajectory_embeddings) # Remove or repurpose

        success = rollout_consistent(
            env, model, task_checker, subtask, val_annotations,
            debug, eval_dir, subtask_i, sequence_i, ep_len, subtask_seed,
            global_sequence_id=global_sequence_id, subtask_in_sequence_index=subtask_i # Pass identifiers
        )

        # The rollout function now handles storing the subtask data
        # We just need to count success for the sequence result
        if success:
            success_counter += 1
        else:
            # Optional: Handle remaining subtasks in the sequence (e.g., mark them as failed if needed)
            # For now, we assume rollout_consistent handles storing the failed subtask.
            # If you want to explicitly store failed future subtasks as separate trajectories,
            # you could loop here and call a function to store empty/failure data.
            # Example (optional):
            # remaining_subtasks = eval_sequence[subtask_i + 1:]
            # for remaining_i, remaining_subtask in enumerate(remaining_subtasks):
            #     store_empty_subtask_trajectory(remaining_subtask, global_sequence_id, subtask_i + 1 + remaining_i, success=False)
            return success_counter # Stop on first failure

    return success_counter

def store_subtask_trajectory(subtask_name, subtask_embeddings, success, global_sequence_id, subtask_in_sequence_index, debug_info=None):
    """Helper function to store a completed subtask as a trajectory."""
    global all_subtask_trajectories_data

    traj_length = len(subtask_embeddings)
    # Assign a unique trajectory ID for this subtask
    # You could use a global counter or derive from sequence/subtask indices
    # Using sequence and subtask index for traceability
    subtask_traj_id = f"seq_{global_sequence_id}_subtask_{subtask_in_sequence_index}_{subtask_name}"

    # Prepare metadata for this subtask "trajectory"
    metadata = {
        'trajectory_id': subtask_traj_id, # Unique ID for this subtask instance
        'original_sequence_id': global_sequence_id,
        'subtask_in_sequence_index': subtask_in_sequence_index,
        'subtask_name': subtask_name,
        'num_successful_steps': traj_length if success else 0, # Or count actual success steps if available
        'total_steps': traj_length,
        'overall_success': success,
        # Note: subtask_boundaries concept doesn't apply directly now,
        # unless you want to define it as (0, traj_length) for this single subtask trajectory.
        # 'subtask_boundaries': [(0, traj_length)],
        # 'subtask_names': [subtask_name], # Redundant now
        # 'subtask_success': [1 if success else 0], # Redundant now
        # Add any other relevant debug info
        'debug_info': debug_info or {}
    }

    # Store the data
    all_subtask_trajectories_data['embeddings'].append(subtask_embeddings)
    all_subtask_trajectories_data['metadata'].append(metadata)
    # print(f"Stored subtask trajectory: {subtask_traj_id}, Success: {success}, Length: {traj_length}")

def rollout_consistent(env, model, task_oracle, subtask, val_annotations,
                      debug, eval_dir, subtask_i, sequence_i, ep_len, seed,
                      global_sequence_id, subtask_in_sequence_index):
    """Rollout with embedding collection, treating this rollout as a single trajectory"""
    global current_subtask_rollout_embeddings # Use the new global for current subtask data
    current_subtask_rollout_embeddings = [] # Reset for this subtask

    print(f"{subtask} ", end="")
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    if debug:
        img_dict = {
            'static': [],
            'gripper': [],
            'pred_static': [],
            'pred_gripper': [],
        }

    unfinished = 0
    for step in range(ep_len):
        print(step)
        if unfinished == 0:
            torch.manual_seed(seed + step)
            output, save_results = model.step(obs, lang_annotation)
            # Collect embeddings FOR THIS SUBTASK
            current_subtask_rollout_embeddings.append(save_results)
            action = output['action_pred']
            unfinished = action.shape[0]
        obs, _, _, current_info = env.step(action[-unfinished])
        unfinished -= 1

        if debug:
            img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
            img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))
            img_dict['pred_static'].append(copy.deepcopy(output['obs_preds'][0, -1].astype(np.uint8)))
            img_dict['pred_gripper'].append(copy.deepcopy(output['obs_hand_preds'][0, -1].astype(np.uint8)))

        # Check if current step solves the SUBTASK
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        gif_subdir = os.path.join(eval_dir, "GIF")
        os.makedirs(gif_subdir, exist_ok=True)
        
        if len(current_task_info) > 0:
            # --- SUBTASK SUCCESS ---
            if debug:
                print(colored("success", "green"), end=" ")
                for key in img_dict.keys():
                    if key == "static":
                        try:
                            clip = ImageSequenceClip(img_dict[key], fps=30)
                            clip.write_gif(os.path.join(gif_subdir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-succ.gif'), fps=30)
                        except Exception as e:
                            print(f"Error saving success GIF for {key}: {e}")

            # Store the successful subtask data as a trajectory
            store_subtask_trajectory(
                subtask_name=subtask,
                subtask_embeddings=current_subtask_rollout_embeddings,
                success=True,
                global_sequence_id=global_sequence_id,
                subtask_in_sequence_index=subtask_in_sequence_index,
                debug_info={'termination_reason': 'success'}
            )
            return True # Indicate subtask success

    # --- SUBTASK FAILURE (loop ended without success) ---
    if debug:
        print(colored("fail", "red"), end=" ")
        for key in img_dict.keys():
            if key == "static":
                try:
                    clip = ImageSequenceClip(img_dict[key], fps=30)
                    clip.write_gif(os.path.join(gif_subdir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-fail.gif'), fps=30)
                except Exception as e:
                    print(f"Error saving failure GIF for {key}: {e}")

    # Store the failed subtask data as a trajectory
    store_subtask_trajectory(
        subtask_name=subtask,
        subtask_embeddings=current_subtask_rollout_embeddings,
        success=False,
        global_sequence_id=global_sequence_id,
        subtask_in_sequence_index=subtask_in_sequence_index,
        debug_info={'termination_reason': 'timeout'}
    )
    return False # Indicate subtask failure


# --- MODIFIED save_all_trajectories ---
# This function now expects data structured per subtask.
# The logic is similar to the previous correction but iterates over subtasks instead of sequences.
def save_all_trajectories(all_data, save_path, model_name):
    """Save all SUBTASK "trajectory" data with proper boundaries and subtask information"""
    # Initialize processed data lists
    processed_data = {
        # Each "trajectory" is now a subtask
        'trajectory_start_indices': [],
        'trajectory_lengths': [],
        # Subtask metadata (now describes the single subtask this "trajectory" represents)
        'original_subtask_name': [], # Name of the subtask this trajectory is for
        'original_sequence_id': [],  # Sequence this subtask belonged to
        'subtask_in_sequence_index': [], # Index within the original sequence
        'trajectory_success': [],    # Success of this subtask "trajectory"
        'trajectory_success_steps': [], # Steps in this subtask
        # Embeddings will be collected per subtask "trajectory" then concatenated at the end
        'Y': [],
        'Y_hat': [],
        'Z': [],
        'hand_obs_embeddings': [],
        'hand_patch_embeddings': [],
        'lang_embeddings': [],
        'obs_embeddings': [],
        'patch_embeddings': [],
        'state_embeddings': []
    }

    # For tracking unique subtask *types* encountered
    subtask_name_to_id = {}
    subtask_names_list = []
    all_subtask_name_ids = [] # ID of the subtask type for each "trajectory"
    current_global_idx = 0

    # --- Process each SUBTASK "trajectory" individually ---
    for traj_idx, (traj_embeddings, traj_metadata) in enumerate(zip(all_data['embeddings'], all_data['metadata'])):
        traj_length = len(traj_embeddings)

        # Handle empty subtask trajectories gracefully (should ideally not happen)
        if traj_length == 0:
            print(f"Warning: Found empty subtask trajectory at index {traj_idx} (ID: {traj_metadata.get('trajectory_id', 'Unknown')}). Skipping embedding concatenation.")
            processed_data['trajectory_start_indices'].append(current_global_idx)
            processed_data['trajectory_lengths'].append(0)
            processed_data['trajectory_success'].append(0)
            processed_data['trajectory_success_steps'].append(0)
            # Add placeholders for subtask metadata
            processed_data['original_subtask_name'].append("empty")
            processed_data['original_sequence_id'].append(-1)
            processed_data['subtask_in_sequence_index'].append(-1)
            all_subtask_name_ids.append(-1) # Or a specific ID for 'empty'
            continue

        # 1. Record Trajectory Metadata (for this subtask)
        processed_data['trajectory_start_indices'].append(current_global_idx)
        processed_data['trajectory_lengths'].append(traj_length)
        processed_data['trajectory_success'].append(1 if traj_metadata.get('overall_success', False) else 0)
        processed_data['trajectory_success_steps'].append(traj_metadata.get('num_successful_steps', 0))

        # 2. Record Subtask-Specific Metadata
        subtask_name = traj_metadata.get('subtask_name', 'unknown')
        processed_data['original_subtask_name'].append(subtask_name)
        processed_data['original_sequence_id'].append(traj_metadata.get('original_sequence_id', -1))
        processed_data['subtask_in_sequence_index'].append(traj_metadata.get('subtask_in_sequence_index', -1))

        # 3. Handle subtask name mapping
        if subtask_name not in subtask_name_to_id:
            new_id = len(subtask_name_to_id) # Assign new ID
            subtask_name_to_id[subtask_name] = new_id
            subtask_names_list.append(subtask_name)
        all_subtask_name_ids.append(subtask_name_to_id[subtask_name])

        # 4. Concatenate Embeddings for THIS subtask "trajectory" and append to lists
        # Assuming all steps have the same keys
        if traj_embeddings: # Check if not empty
            embedding_keys = traj_embeddings[0].keys()
            for key in embedding_keys:
                # Collect tensors for this single subtask "trajectory"
                tensors_list = [step[key] for step in traj_embeddings if key in step]
                if tensors_list: # Check if list is not empty before concatenating
                    # Concatenate along the step dimension (assuming dim 0)
                    stacked_tensor = torch.cat(tensors_list, dim=0)
                    processed_data[key].append(stacked_tensor)

        # 5. Update global index for the next subtask "trajectory"
        current_global_idx += traj_length

    # --- Finalize data structure ---
    final_data = {}
    # Concatenate all subtask "trajectory"-level embeddings into single tensors
    for key in ['Y', 'Y_hat', 'Z', 'hand_obs_embeddings', 'hand_patch_embeddings',
                'lang_embeddings', 'obs_embeddings', 'patch_embeddings', 'state_embeddings']:
        # Only process keys that have data collected
        if processed_data[key] and all(isinstance(t, torch.Tensor) for t in processed_data[key]):
            # Concatenate list of subtask "trajectory" tensors into one big tensor
            final_data[key] = torch.cat(processed_data[key], dim=0)
        elif processed_data[key]:
            print(f"Warning: Key '{key}' has non-tensor data in processed_data, skipping concatenation.")
        else:
            print(f"Info: Key '{key}' has no data, skipping.")

    # Convert list metadata to tensors
    # Trajectory metadata (now subtask "trajectory" metadata)
    final_data['trajectory_start_indices'] = torch.tensor(processed_data['trajectory_start_indices'], dtype=torch.long)
    final_data['trajectory_lengths'] = torch.tensor(processed_data['trajectory_lengths'], dtype=torch.long)
    final_data['trajectory_success'] = torch.tensor(processed_data['trajectory_success'], dtype=torch.long)
    final_data['trajectory_success_steps'] = torch.tensor(processed_data['trajectory_success_steps'], dtype=torch.long)

    # Subtask-specific metadata (now describing the "trajectory")
    final_data['original_subtask_name_ids'] = torch.tensor(all_subtask_name_ids, dtype=torch.long) # Map to IDs
    final_data['original_sequence_id'] = torch.tensor(processed_data['original_sequence_id'], dtype=torch.long)
    final_data['subtask_in_sequence_index'] = torch.tensor(processed_data['subtask_in_sequence_index'], dtype=torch.long)

    # Add subtask name mapping to final data for easy lookup
    final_data['subtask_mapping'] = {
        'subtask_names': subtask_names_list,
        'subtask_name_to_id': subtask_name_to_id,
        'id_to_subtask_name': {str(v): k for k, v in subtask_name_to_id.items()}
    }
    # Also store original subtask names directly if needed (though IDs are more efficient)
    # final_data['original_subtask_name'] = processed_data['original_subtask_name'] # This would be a list, not tensor

    # --- Save files ---
    filename = f"{model_name}_all_subtask_trajectories.safetensors" # Change filename to reflect
    filepath = os.path.join(save_path, filename)

    # Save main data (only tensor data)
    tensor_data = {k: v for k, v in final_data.items() if isinstance(v, torch.Tensor)}
    save_file(tensor_data, filepath)

    # Save subtask name mapping separately as JSON
    subtask_mapping_path = os.path.join(save_path, f"{model_name}_subtask_mapping.json")
    mapping_for_json = {
        'subtask_names': final_data['subtask_mapping']['subtask_names'],
        'subtask_name_to_id': final_data['subtask_mapping']['subtask_name_to_id'],
        'id_to_subtask_name': final_data['subtask_mapping']['id_to_subtask_name']
    }
    with open(subtask_mapping_path, 'w') as f:
        json.dump(mapping_for_json, f, indent=2)

    # Save summary
    total_trajectories = len(processed_data['trajectory_lengths']) # Now counts subtasks
    successful_trajectories = int(final_data['trajectory_success'].sum().item()) if 'trajectory_success' in final_data else 0
    total_steps = int(current_global_idx)
    unique_subtask_types = len(subtask_name_to_id)

    summary = {
        'model': model_name,
        'total_subtask_trajectories': total_trajectories, # Renamed for clarity
        'successful_subtask_trajectories': successful_trajectories,
        'failed_subtask_trajectories': total_trajectories - successful_trajectories,
        'total_steps_across_all_subtasks': total_steps, # Renamed for clarity
        'unique_subtask_types': unique_subtask_types,
        'embedding_shapes': {key: list(tensor.shape) for key, tensor in tensor_data.items()}
    }
    summary_path = os.path.join(save_path, f"{model_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved all SUBTASK trajectories to: {filepath}")
    print(f"Total subtask trajectories: {summary['total_subtask_trajectories']}")
    print(f"Total steps across all subtasks: {summary['total_steps_across_all_subtasks']}")
    print(f"Unique subtask types: {summary['unique_subtask_types']}")
    print(f"\nSubtask trajectory lengths (first 10): {processed_data['trajectory_lengths'][:10]}")
    print(f"Subtask trajectory start indices (first 10): {processed_data['trajectory_start_indices'][:10]}")

    return filepath

# ... (rest of the code like load_trajectory_data, get_subtask_embeddings, etc. might need minor adjustments
#      to work with the new structure, but the core logic remains similar) ...

# Example usage (conceptual, as it's called internally):
# filepath = save_all_trajectories(all_subtask_trajectories_data, './output_embeddings/', 'GR1_test_model')
# Expected output:
# Total subtask trajectories: 150 (e.g., if you had 30 sequences * 5 subtasks avg)
# Subtask trajectory lengths: [12, 45, 8, 22, ...] (length of each individual subtask rollout)
# Subtask trajectory start indices: [0, 12, 57, 65, ...] (cumulative sum based on subtask lengths)

# === Helper functions remain the same ===

def load_trajectory_data(filepath, mapping_path=None):
    """Load trajectory data with subtask information"""
    data = load_file(filepath)
    
    if mapping_path is not None:
        with open(mapping_path, 'r') as f:
            data['subtask_mapping'] = json.load(f)
    
    return data


def get_subtask_embeddings(data, subtask_name=None, subtask_id=None, success_only=None):
    """Get embeddings for specific subtasks with detailed filtering"""
    indices = []
    metadata = {'subtask_count': 0, 'success_count': 0, 'fail_count': 0}
    
    # Convert subtask name to ID if needed
    if subtask_name is not None and 'subtask_mapping' in data:
        subtask_id = data['subtask_mapping']['subtask_name_to_id'].get(subtask_name)
        if subtask_id is None:
            print(f"Warning: Subtask '{subtask_name}' not found")
            return indices, metadata
    
    # Iterate through all subtasks
    num_subtasks = len(data['subtask_start_indices'])
    for i in range(num_subtasks):
        # Check if this is the subtask we're looking for
        if subtask_id is not None and data['subtask_name_id'][i].item() != subtask_id:
            continue
        
        # Check success filter
        is_success = data['subtask_success'][i].item() == 1
        if success_only is not None:
            if success_only and not is_success:
                continue
            if not success_only and is_success:
                continue
        
        # Get indices for this subtask
        start = data['subtask_start_indices'][i].item()
        length = data['subtask_lengths'][i].item()
        subtask_indices = list(range(start, start + length))
        indices.extend(subtask_indices)
        
        # Update metadata
        metadata['subtask_count'] += 1
        if is_success:
            metadata['success_count'] += 1
        else:
            metadata['fail_count'] += 1
    
    return indices, metadata


def get_trajectory_info(data):
    """Get information about trajectories in the dataset"""
    num_trajectories = len(data['trajectory_start_indices'])
    
    print(f"Total trajectories: {num_trajectories}")
    print(f"Trajectory lengths: {data['trajectory_lengths'].tolist()}")
    print(f"Trajectory start indices: {data['trajectory_start_indices'].tolist()}")
    
    # Show first few trajectories
    for i in range(min(5, num_trajectories)):
        start = data['trajectory_start_indices'][i].item()
        length = data['trajectory_lengths'][i].item()
        success = data['trajectory_success'][i].item()
        
        print(f"\nTrajectory {i}:")
        print(f"  Range: [{start}, {start + length})")
        print(f"  Length: {length}")
        print(f"  Success: {'Yes' if success else 'No'}")
        
        # Show subtasks in this trajectory
        subtasks_in_traj = []
        for j in range(len(data['subtask_trajectory_id'])):
            if data['subtask_trajectory_id'][j].item() == i:
                subtask_name_id = data['subtask_name_id'][j].item()
                if 'subtask_mapping' in data:
                    subtask_name = data['subtask_mapping']['id_to_subtask_name'][str(subtask_name_id)]
                else:
                    subtask_name = f"Subtask_{subtask_name_id}"
                
                subtask_success = data['subtask_success'][j].item()
                subtasks_in_traj.append((subtask_name, subtask_success))
        
        print(f"  Subtasks: {subtasks_in_traj}")


# Keep other unchanged functions...
def sort_key(string):
    """Sort function for checkpoint filenames"""
    split_string = string.split('_')
    if len(split_string) > 1:
        return int(split_string[1].split('.')[0])
    else:
        return float('inf')


def count_success(results):
    """Count success rates for different sequence lengths"""
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(filename, results, sequences, eval_result_path, epoch=None):
    """Print and save evaluation results"""
    avg_seq_len = np.mean(results)
    
    chain_sr_values = count_success(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(chain_sr_values)}

    print(f"Average successful sequence length: {avg_seq_len:.3f}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"  {i}: {sr * 100:.1f}%")

    current_epoch_data = {
        "chain_sr": chain_sr,
        "avg_seq_len": float(avg_seq_len)
    }
    
    data = {}
    data[filename] = {}
    data[filename].update(current_epoch_data)

    output_path = eval_result_path + "/" + filename + '.txt'
    print("output_path", output_path)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\nSaved results to: {output_path}")


def get_success_failure_trajectories(data):
    """Separate trajectories by success/failure"""
    success_indices = []
    failure_indices = []
    
    for i in range(len(data['trajectory_success'])):
        start_idx = data['trajectory_start_indices'][i].item()
        length = data['trajectory_lengths'][i].item()
        indices = list(range(start_idx, start_idx + length))
        
        if data['trajectory_success'][i].item() == 1:
            success_indices.extend(indices)
        else:
            failure_indices.extend(indices)
    
    return success_indices, failure_indices


def make_env_deterministic(dataset_path, subset, observation_space, device, seed=42):
    """Create environment with deterministic settings"""
    print("dataset_path--------", dataset_path)
    val_folder = Path(dataset_path) / "validation"
    from evaluation.calvin_env_wrapper_raw import CalvinEnvWrapperRaw
    
    set_seed_comprehensive(seed, verbose=True)
    
    env = CalvinEnvWrapperRaw(val_folder, subset, observation_space, device)
    
    if hasattr(env, 'seed'):
        env.seed(seed)
    
    return env


def get_deterministic_sequences(num_sequences, seed=42):
    """Get evaluation sequences in a deterministic order"""
    set_seed_comprehensive(seed)
    
    sequences = get_sequences(num_sequences)
    
    sequences_list = list(sequences)
    sequences_list.sort(key=lambda x: str(x))
    print("sequences_list", sequences_list)
    
    return sequences_list


def build_gr1_model(cfg, device):
    """Build a GR1 model from a config dictionary without loading any checkpoint."""
    model_clip, _ = clip.load(cfg['clip_backbone'], device=device)
    model_mae = vits.__dict__['vit_base'](patch_size=16, num_classes=0).to(device)
    ckpt = torch.load(cfg['mae_ckpt'], map_location=device)
    model_mae.load_state_dict(ckpt['model'], strict=False)

    model = GR1(
        model_clip,
        model_mae,
        rgb_shape=cfg['rgb_shape'],
        patch_size=cfg['patch_size'],
        state_dim=cfg['state_dim'],
        act_dim=cfg['act_dim'],
        hidden_size=cfg['embed_dim'],
        sequence_length=cfg['seq_len'],
        chunk_size=cfg['chunk_size'],
        training_target=['act_pred', 'fwd_pred', 'fwd_pred_hand'],
        img_feat_dim=cfg['img_feat_dim'],
        patch_feat_dim=cfg['patch_feat_dim'],
        lang_feat_dim=cfg['lang_feat_dim'],
        resampler_params={
            'depth': cfg['resampler_depth'],
            'dim_head': cfg['resampler_dim_head'],
            'heads': cfg['resampler_heads'],
            'num_latents': cfg['resampler_num_latents'],
            'num_media_embeds': cfg['resampler_num_media_embeds'],
        },
        without_norm_pixel_loss=cfg['without_norm_pixel_loss'],
        skip_frame=cfg.get('skip_frame', 0),
        use_hand_rgb=True,
        n_layer=cfg['n_layer'],
        n_head=cfg['n_head'],
        n_inner=4 * cfg['embed_dim'],
        activation_function=cfg['activation_function'],
        n_positions=cfg['n_positions'],
        resid_pdrop=cfg['dropout'],
        attn_pdrop=cfg['dropout'],
    ).to(device)
    return model


def load_checkpoint(model, ckpt_path, device):
    """Load checkpoint into model"""
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        missing_keys, unexpected_keys = model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
    print(f"Loaded checkpoint from {ckpt_path}")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")


def evaluate_noisy_checkpoints_consistent(config_path, ckpt_dir, output_dir, args):
    """Main evaluation function with consistent data loading"""
    os.makedirs(output_dir, exist_ok=True)
    
    set_seed_comprehensive(42, verbose=True)
    
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    acc = Accelerator(kwargs_handlers=[kwargs])
    device = acc.device
    print("device", device)

    cfg = json.load(open(config_path, 'r'))
    base_model = build_gr1_model(cfg, device)

    if cfg.get('compile_model', False):
        base_model = torch.compile(base_model)

    model = acc.prepare(base_model)
    model.eval()

    preprocessor = PreProcess(
        cfg['rgb_static_pad'],
        cfg['rgb_gripper_pad'],
        cfg['rgb_shape'],
        cfg['rgb_mean'],
        cfg['rgb_std'],
        device,
    )
    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'],
        'depth_obs': [],
        'state_obs': ['robot_obs'],
        'actions': ['rel_actions'],
        'language': ['language']
    }

    env = make_env_deterministic('./fake_dataset', cfg["subset"], observation_space, device, seed=42)
    evaluator = GR1CalvinEvaluation(model, cfg, preprocessor, device)

    results_summary = {}

    ckpt_path = ckpt_dir
    filename = os.path.basename(ckpt_path)
    
    print(f"\n=== Evaluating {filename} ===")
    
    load_checkpoint(base_model, ckpt_path, device)
    base_model.eval()
    
    if hasattr(evaluator, 'reset'):
        evaluator.reset()

    model_seed = 42
    results = evaluate_policy_consistent(
        ckpt_path,
        evaluator,
        env,
        output_dir,
        output_dir,
        cfg['ep_len'],
        cfg['num_sequences'],
        acc.num_processes,
        acc.process_index,
        output_dir,
        debug=True,
        seed=model_seed
    )

    success_tensor = torch.tensor(results, dtype=torch.float, device=device)
    mean_sr = success_tensor.mean().item()
    acc.wait_for_everyone()
    global_mean_sr = acc.gather_for_metrics(success_tensor.mean()).mean().item()

    if acc.is_main_process:
        results_summary[filename] = global_mean_sr
        print(f"[{filename}] -> Avg. Success: {global_mean_sr:.4f}")

    if acc.is_main_process:
        summary_file = os.path.join(output_dir, "noise_eval_summary.txt")
        with open(summary_file, 'w') as f:
            for name, sr in results_summary.items():
                f.write(f"{name}: {sr:.4f}\n")
        print(f"Summary written to {summary_file}")


def load_all_batches(save_dir, model_name):
    """
    Load and combine all batch files for a given model.
    
    Args:
        save_dir: Directory containing the batch files
        model_name: Model name used in the filenames
    
    Returns:
        Combined data dictionary with all trajectories
    """
    # First, load the master summary to get batch information
    master_summary_path = os.path.join(save_dir, f"{model_name}_master_summary.json")
    
    if not os.path.exists(master_summary_path):
        print(f"Master summary not found: {master_summary_path}")
        return None
    
    with open(master_summary_path, 'r') as f:
        master_summary = json.load(f)
    
    print(f"Loading {master_summary['num_batches']} batch files for model: {model_name}")
    
    # Initialize combined data structure
    combined_data = None
    all_subtask_names = []
    subtask_name_to_global_id = {}
    global_subtask_id_mapping = []  # Maps batch-specific IDs to global IDs
    
    # Load each batch file
    for batch_idx in range(master_summary['num_batches']):
        batch_filename = f"{model_name}_batch_{batch_idx}_all_subtask_trajectories.safetensors"
        batch_filepath = os.path.join(save_dir, batch_filename)
        
        # Load subtask mapping for this batch
        batch_mapping_path = os.path.join(save_dir, f"{model_name}_batch_{batch_idx}_subtask_mapping.json")
        
        if not os.path.exists(batch_filepath):
            print(f"Warning: Batch file not found: {batch_filepath}")
            continue
            
        print(f"Loading batch {batch_idx}...")
        batch_data = load_file(batch_filepath)
        
        # Load subtask mapping if exists
        batch_subtask_mapping = None
        if os.path.exists(batch_mapping_path):
            with open(batch_mapping_path, 'r') as f:
                batch_subtask_mapping = json.load(f)
        
        # If this is the first batch, initialize combined_data
        if combined_data is None:
            combined_data = {}
            for key in batch_data.keys():
                combined_data[key] = []
        
        # Calculate offset for indices based on previous batches
        index_offset = 0
        if batch_idx > 0:
            # Calculate total steps from previous batches
            for prev_batch in range(batch_idx):
                prev_batch_file = f"{model_name}_batch_{prev_batch}_all_subtask_trajectories.safetensors"
                prev_batch_path = os.path.join(save_dir, prev_batch_file)
                if os.path.exists(prev_batch_path):
                    prev_data = load_file(prev_batch_path)
                    # Get total steps from trajectory data
                    if 'trajectory_lengths' in prev_data:
                        index_offset += prev_data['trajectory_lengths'].sum().item()
        
        # Process each tensor in the batch
        for key in batch_data.keys():
            if key == 'trajectory_start_indices' or key == 'subtask_start_indices':
                # Adjust indices by adding offset
                adjusted_tensor = batch_data[key] + index_offset
                combined_data[key].append(adjusted_tensor)
            elif key == 'original_subtask_name_ids' and batch_subtask_mapping:
                # Remap subtask IDs to global IDs
                remapped_ids = []
                for local_id in batch_data[key]:
                    local_id_val = local_id.item()
                    # Find the subtask name for this local ID
                    local_name = batch_subtask_mapping['id_to_subtask_name'].get(str(local_id_val), f"unknown_{local_id_val}")
                    
                    # Get or create global ID for this subtask name
                    if local_name not in subtask_name_to_global_id:
                        global_id = len(subtask_name_to_global_id)
                        subtask_name_to_global_id[local_name] = global_id
                        all_subtask_names.append(local_name)
                    
                    global_id = subtask_name_to_global_id[local_name]
                    remapped_ids.append(global_id)
                
                remapped_tensor = torch.tensor(remapped_ids, dtype=torch.long)
                combined_data[key].append(remapped_tensor)
            else:
                # For all other tensors, just append
                combined_data[key].append(batch_data[key])
    
    # Concatenate all tensors
    print("Concatenating all batch data...")
    final_data = {}
    for key in combined_data.keys():
        if len(combined_data[key]) > 0:
            final_data[key] = torch.cat(combined_data[key], dim=0)
        else:
            print(f"Warning: No data found for key '{key}'")
    
    # Add combined subtask mapping
    final_data['subtask_mapping'] = {
        'subtask_names': all_subtask_names,
        'subtask_name_to_id': subtask_name_to_global_id,
        'id_to_subtask_name': {str(v): k for k, v in subtask_name_to_global_id.items()}
    }
    
    # Print summary
    total_trajectories = len(final_data.get('trajectory_lengths', []))
    total_steps = final_data.get('trajectory_lengths', torch.tensor([])).sum().item()
    print(f"\nCombined data summary:")
    print(f"  Total trajectories: {total_trajectories}")
    print(f"  Total steps: {total_steps}")
    print(f"  Unique subtask types: {len(all_subtask_names)}")
    
    return final_data


# Example usage:
# combined_data = load_all_batches('./trajectory_embeddings/', 'GR1_9')
# 
# # Now you can use the combined data as if it were a single file:
# indices, meta = get_subtask_embeddings(combined_data, subtask_name="push_red_button")
# print(f"Found {meta['subtask_count']} push_red_button instances across all batches")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs.json',
                        help='Path to the JSON config file.')
    parser.add_argument('--ckpt_dir', type=str, default='/home/eciel/VLA/GR1-Training/eva_result2/model_ckp/GR1_9.pth',
                        help='Directory containing noisy checkpoints.')
    parser.add_argument('--output_dir', type=str, default='eval_trained_results/',
                        help='Directory to store final summary of results.')
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--main_process_port', type=int)
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args()

    evaluate_noisy_checkpoints_consistent(args.config, args.ckpt_dir, args.output_dir, args)
    

if __name__ == "__main__":
    main()


"""
=== USAGE EXAMPLES ===

1. Load and check trajectory structure:
```python
data = load_trajectory_data(
    'GR1_9_all_trajectories.safetensors',
    'GR1_9_subtask_mapping.json'
)

# Check trajectory information
get_trajectory_info(data)
```

2. Get embeddings for specific subtask:
```python
# Get all "push_red_button" attempts
indices, meta = get_subtask_embeddings(data, subtask_name="push_red_button")
print(f"Found {meta['subtask_count']} instances")

# Get only successful ones
success_idx, _ = get_subtask_embeddings(data, subtask_name="push_red_button", success_only=True)
X_success = data['lang_embeddings'][success_idx]
Y_success = data['Y'][success_idx]
```

3. Analyze I(X,Y) by subtask success/failure:
```python
for subtask_name in data['subtask_mapping']['subtask_names']:
    success_idx, success_meta = get_subtask_embeddings(data, subtask_name, success_only=True)
    fail_idx, fail_meta = get_subtask_embeddings(data, subtask_name, success_only=False)
    
    if success_idx and fail_idx:
        # Your MI computation
        print(f"{subtask_name}: {success_meta['success_count']} success, {fail_meta['fail_count']} fail")
```
"""