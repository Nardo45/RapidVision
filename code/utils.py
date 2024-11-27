# Import necessary modules for file operations, system information, and JSON parsing
import os, sys, json

from inspect import stack


# Function to extract JSON data from a file and convert it into a Python dictionary
def extract_json_2_dict(path_2_json):
    """
    Extracts data from a JSON file and converts it into a Python dictionary.

    Parameters:
    path_2_json (str): The file path to the JSON file to be read.

    Returns:
    dict: A dictionary representation of the JSON data extracted from the file.

    Raises:
    FileNotFoundError: If the specified JSON file does not exist.
    json.JSONDecodeError: If the file content is not valid JSON.
    """
    try:
        # Open the JSON file in read mode
        with open(path_2_json, 'r') as file_json:
            return json.load(file_json)
    except FileNotFoundError:
        print(f"Error: The file '{path_2_json}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{path_2_json}' contains invalid JSON.")

    return None

def save_2_json(path_2_json, new_data):
    data = {}
    if os.path.exists(path_2_json) and os.path.getsize(path_2_json) > 0:
        try:
            with open(path_2_json, 'r') as data_file:
                data = json.load(data_file)
        except json.JSONDecodeError:
            print(f"Error: The file '{path_2_json}' contains invalid JSON.")

    data.update(new_data)

    with open(path_2_json, 'w') as data_file:
        json.dump(data, data_file, indent=4)


def absolute_path(parent_folder, file_or_folder_name=None, search_folder_name=None):
    # Get the directory of the file that called this function
    caller_directory = os.path.dirname(stack()[1].filename)

    while os.path.basename(caller_directory) != parent_folder:
        caller_directory = os.path.dirname(caller_directory)
    
    # Check if running under PyInstaller, and use the _MEIPASS directory if available
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        file_path = os.path.join(sys._MEIPASS, file_or_folder_name or '')
        if os.path.exists(file_path):
            return file_path

    if not file_or_folder_name and not search_folder_name:
        return caller_directory
    
    for root, dirs, files in os.walk(caller_directory):
        if search_folder_name and search_folder_name in dirs:
            if file_or_folder_name:
                file_path = os.path.join(root, search_folder_name, file_or_folder_name)
                if os.path.exists(file_path):
                    return file_path
            else:
                return os.path.join(root, search_folder_name)
        elif file_or_folder_name and file_or_folder_name in files:
            return os.path.join(root, file_or_folder_name)
        
    raise FileNotFoundError(f"File or folder '{file_or_folder_name}' not found.")