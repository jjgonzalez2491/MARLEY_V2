import json
import numpy as np
import pandas as pd
import os

def detect_and_average_overlaps(matrix):
    df = pd.DataFrame(matrix)
    df.columns = ['iteration'] + [f'result_{i}' for i in range(1, matrix.shape[1])]
    averaged_df = df.groupby('iteration').mean().reset_index()
    averaged_df = averaged_df.sort_values('iteration')
    return averaged_df.values


def quick_overlap_fix(matrix):
    return detect_and_average_overlaps(matrix)


def find_values(key, dictionary):
    values = []

    def _find_values(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key:
                    values.append(v)
                elif isinstance(v, (dict, list)):
                    _find_values(v)
        elif isinstance(obj, list):
            for item in obj:
                _find_values(item)

    _find_values(dictionary)
    return values

# NEW: extract nested key path directly
def get_nested_value(d, path):
    """
    Example:
    path = ['env_runners','agent_episode_returns_mean','Agent_p_0']
    """
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur

string = '250526_CM_CfD_runs'

agent_g = 16

scenarios = {    
    79: (6,  673, 'CM_CfD_', 1, 0, 100, 80, 0.50, False, 2, True),     
    80: (6,  674, 'CM_CfD_', 1, 0, 100, 80, 0.50, False, 2, True),           
    81: (6,  675, 'CM_CfD_', 1, 0, 100, 80, 0.50, False, 2, True),           
    82: (6,  676, 'CM_CfD_', 1, 1, 100, 80, 0.50, False, 2, True),           
    83: (6,  677, 'CM_CfD_', 1, 1, 100, 80, 0.50, False, 2, True),            
    84: (6,  678, 'CM_CfD_', 1, 1, 100, 80, 0.50, False, 2, True),                                                                                    
}

run_numbers = np.array([673, 674, 675, 676, 677, 678])

checkpoint_runs = [
    "80_CM_CfD_80", "81_CM_CfD_81", "82_CM_CfD_82",   # H + IPPO
    "83_CM_CfD_83", "84_CM_CfD_84", "85_CM_CfD_85",   # H + MAPPO
]

key_numbers = np.array([80, 81, 82, 83, 84, 85])

method = ["MAPPO", "MAPPO", "MAPPO", "MAPPO", "MAPPO", "MAPPO"]

run_number_total = 6

# ... (keep detect_and_average_overlaps and quick_overlap_fix as they are) ...

def get_nested_value(d, path):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur

def find_root_folders(root_folder):
    json_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename == "result.json":
                file_path = os.path.join(dirpath, filename)
                json_files.append(file_path)
    json_files.sort(key=lambda x: os.path.getctime(x))
    return json_files

# Setup agent IDs for rewards and entropy
agent_policy_ids = [f"policy_Agent_g_{i}" for i in range(agent_g)] + ["policy_Agent_p_0"]

for run_number_tmp in range(run_number_total):

    run_number = run_numbers[run_number_tmp]
    key_number = key_numbers[run_number_tmp]
    method_tmp = method[run_number_tmp]

    root_folder = f'/work/cmcc/jg24923/CM_EoM_ss_battery_{run_number}/{method_tmp}'
    file_names = find_root_folders(root_folder)
    
    if not file_names:
        print(f"No files found for run {run_number}")
        continue

    # Containers for different metric types
    reward_rows = []
    entropy_rows = []
    system_rows = []

    for file_name in file_names:
        with open(file_name, 'r') as f:
            for line in f:
                if not line.strip(): continue
                row_data = json.loads(line)
                
                # Use the specific timestep metric you identified
                step_val = get_nested_value(row_data, ['env_runners', 'num_module_steps_sampled_lifetime', 'policy_Agent_p_0'])
                if step_val is None: continue 

                # 1. AGENT REWARDS
                rew_dict = {'timesteps': step_val}
                for p_id in agent_policy_ids:
                    val = get_nested_value(row_data, ['env_runners', 'module_episode_returns_mean', p_id])
                    rew_dict[p_id.replace('policy_', '')] = val if val is not None else np.nan
                reward_rows.append(rew_dict)

                # 2. AGENT ENTROPY
                ent_dict = {'timesteps': step_val}
                for p_id in agent_policy_ids:
                    val = get_nested_value(row_data, ['learners', p_id, 'entropy'])
                    ent_dict[p_id.replace('policy_', '')] = val if val is not None else np.nan
                entropy_rows.append(ent_dict)

                # 3. SYSTEM METRICS (Global)# 3. SYSTEM METRICS (Global & Per-Agent Critic Metrics)
                sys_dict = {
                    'timesteps': step_val,
                    'vf_explained_var_global': get_nested_value(row_data, ['learners', 'shared_critic', 'vf_explained_var']),
                    'episode_return_mean': get_nested_value(row_data, ['env_runners', 'episode_return_mean']),
                    'episode_return_min': get_nested_value(row_data, ['env_runners', 'episode_return_min']),
                    'episode_return_max': get_nested_value(row_data, ['env_runners', 'episode_return_max']),
                }

                # Dynamically add vf_explained_var for ALL agents (Agent_g_0 to Agent_g_15 and Agent_p_0)
                for p_id in agent_policy_ids:
                    # Construct the internal RLlib key name, e.g., 'vf_explained_var_policy_Agent_g_0'
                    rllib_vf_key = f'vf_explained_var_{p_id}'
                    
                    # Construct a clean CSV column name, e.g., 'vf_explained_var_Agent_g_0'
                    clean_column_name = f'vf_explained_var_{p_id.replace("policy_", "")}'
                    
                    val = get_nested_value(row_data, ['learners', 'shared_critic', rllib_vf_key])
                    sys_dict[clean_column_name] = val if val is not None else np.nan

                system_rows.append(sys_dict)

    # Processing and Saving
    metrics_to_save = {
        'agent_mean': reward_rows,
        'agent_entropy': entropy_rows,
        'system_performance': system_rows
    }

    if not os.path.exists(string):
        os.makedirs(string)

    for prefix, rows in metrics_to_save.items():
        if rows:
            df = pd.DataFrame(rows)
            # Apply your overlap logic (averaging duplicate timesteps)
            cleaned_values = quick_overlap_fix(df.values)
            final_df = pd.DataFrame(cleaned_values, columns=df.columns)
            
            output_path = f'{string}/{prefix}_{key_number}.csv'
            final_df.to_csv(output_path, index=False)
            print(f"Saved {prefix} for run {key_number}")