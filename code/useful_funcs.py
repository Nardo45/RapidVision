# Import necessary modules for file operations, system information, and JSON parsing
import os, sys, json, inspect


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
            # Read the file content
            file_content = file_json.read()

            # Parse the JSON content into a Python dictionary
            data = json.loads(file_content)

    except FileNotFoundError:
        print(f"Error: The file '{path_2_json}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{path_2_json}' contains invalid JSON.")
        return None

    # Return the extracted Python dictionary
    return data

def save_2_json(path_2_json, new_data):
    if os.path.exists(path_2_json) or os.path.getsize(path_2_json) > 0:
        with open(path_2_json, 'r') as data_file:
                try:
                    data = json.load(data_file)
                except json.JSONDecodeError:
                    # Start fresh if the data is corrupted
                    data = {}
    else:
        # If the file does't exist or is empty, start with an empty dict
        data = {}

    data.update(new_data)

    # Write the data back to the JSON file (formatted)
    with open(path_2_json, 'w') as data_file:
        json.dump(data, data_file, indent=4)

def absolute_path(parent_folder, file_or_folder_name=None, search_folder_name=None):
    # Get the directory of the file that called this function
    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename
    caller_directory = os.path.dirname(caller_file)

    while os.path.basename(caller_directory) != parent_folder:
        caller_directory = os.path.dirname(caller_directory)
    
    # Check if running under PyInstaller, and use the _MEIPASS directory if available
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        file_path = os.path.join(sys._MEIPASS, file_or_folder_name)
        if os.path.exists(file_path):
            return file_path

    # If both file and folder are specified, search for the folder first, then the file inside it
    if search_folder_name and file_or_folder_name:
        for root, dirs, _ in os.walk(caller_directory):
            if search_folder_name in dirs:
                search_folder_path = os.path.join(root, search_folder_name)
                file_path = os.path.join(search_folder_path, file_or_folder_name)
                if os.path.exists(file_path):
                    return file_path
        raise FileNotFoundError(f"Folder '{search_folder_name}' or file '{file_or_folder_name}' not found.")

    # If only a folder is specified, search for the folder under the parent directory
    elif search_folder_name and not file_or_folder_name:
        for root, dirs, _ in os.walk(caller_directory):
            if search_folder_name in dirs:
                return os.path.join(root, search_folder_name)
        raise FileNotFoundError(f"Folder '{search_folder_name}' not found.")

    # If only a file is specified, search for the file under the parent directory
    elif file_or_folder_name and not search_folder_name:
        for root, _, files in os.walk(caller_directory):
            if file_or_folder_name in files:
                return os.path.join(root, file_or_folder_name)
        raise FileNotFoundError(f"File '{file_or_folder_name}' not found.")

    # If neither folder nor file is specified, raise an error
    else:
        raise ValueError("You must specify either a file or a folder to search for.")