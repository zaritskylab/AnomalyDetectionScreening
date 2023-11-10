import os
import pickle
import pandas as pd


def save_list_to_txt(my_list, file_path):
    # os.path.join(dir, file_path)
    try:
        with open(file_path, 'w') as txt_file:
            for item in my_list:
                txt_file.write(str(item) + '\n')
        print(f"List saved to {file_path} successfully.")
    except Exception as e:
        print(f"Error: {e}")

def load_list_from_txt(file_path):
    # os.path.join(dir, file_path)
    try:
        with open(file_path, 'r') as txt_file:
            my_list = txt_file.read().splitlines()
        print(f"List loaded from {file_path} successfully.")
        return my_list
    except Exception as e:
        print(f"Error: {e}")
        return None


def load_dict_pickles_and_concatenate(directory) -> dict: 
    # Initialize an empty dictionary to store the concatenated data.
    concatenated_dict = {}

    # List all files in the directory.
    files = os.listdir(directory)

    # Loop through each file in the directory.
    for filename in files:
        # Check if the file is a pickle file (ends with .pkl).
        if filename.endswith(".pickle"):
            file_path = os.path.join(directory, filename)
            
            # Load the pickle file and merge it into the concatenated_dict.
            with open(file_path, 'rb') as file:
                data = pickle.load(file)

                if isinstance(data, dict):
                    concatenated_dict.update(data)
                # else:
                    # concatenated_dict = pd.concat([concatenated_dict, data], axis=0)

    return concatenated_dict

def load_df_pickles_and_concatenate(directory) -> pd.DataFrame: 
    # Initialize an empty dictionary to store the concatenated data.
    df = pd.DataFrame([])

    # List all files in the directory.
    files = os.listdir(directory)

    # Loop through each file in the directory.
    for filename in files:
        # Check if the file is a pickle file (ends with .pkl).
        if filename.endswith(".pickle"):
            file_path = os.path.join(directory, filename)
            
            # Load the pickle file and merge it into the concatenated_dict.
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                df = pd.concat([df, data], axis=0)

    return df

