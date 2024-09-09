import os, sys

def absolute_path(parent_folder, file_or_folder_name, search_within_folders=False, search_folder_name=None):
    """
    Locate the absolute path of a file or folder by searching within the current working directory, 
    or within a specified parent folder or subdirectories. The function supports running in both 
    normal execution and when bundled via PyInstaller.

    Parameters:
    -----------
    parent_folder : str
        The name of the parent folder to search up to. If the parent folder is reached, the search stops.
    file_or_folder_name : str
        The name of the file or folder to search for.
    search_within_folders : bool, optional
        If True, the function searches within the subdirectories of the current working directory (default is False).
    search_folder_name : str, optional
        If `search_within_folders` is True, this specifies the specific subfolder name to search in (default is None).

    Returns:
    --------
    str
        The absolute path of the file or folder if found.

    Raises:
    -------
    FileNotFoundError
        If the specified file or folder cannot be found in the current directory or any of its parent directories.

    Notes:
    ------
    - When using PyInstaller, the `sys._MEIPASS` attribute contains the temporary folder where PyInstaller 
      unpacks the bundled app. This function handles such cases and checks there first.
    - The function traverses up from the current directory toward the root directory or the specified 
      `parent_folder`, looking for the target file or folder.
    """

    # Check if running under PyInstaller, and use the _MEIPASS directory if available
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        file_path = os.path.join(sys._MEIPASS, file_or_folder_name)
        if os.path.exists(file_path):
            return file_path

    # Start searching in the current working directory
    current_directory = os.getcwd()

    # If searching within subdirectories, locate the file inside the specified subfolder
    if search_within_folders:
        # List all directories in the current working directory
        dirs_in_parent_folder = [d for d in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, d))]
        
        # If search_folder_name is provided, search inside that folder, otherwise return None
        search_dir = search_folder_name if search_folder_name in dirs_in_parent_folder else None
        
        # Construct the file path inside the target subfolder
        file_path = os.path.join(current_directory, search_dir, file_or_folder_name)
        
        # Return the file path if it exists
        if os.path.exists(file_path):
            return file_path
        else:
            raise FileNotFoundError(f"File '{file_or_folder_name}' not found in current directory.")

    # Traverse up the directory tree until the parent folder is reached or the root directory
    while current_directory != 'c:\\' and current_directory != 'c:/' and os.path.basename(current_directory) != parent_folder:
        # Check for the file in the current directory
        file_path = os.path.join(current_directory, file_or_folder_name)
        
        # Return the path if the file is found
        if os.path.exists(file_path):
            return file_path

        # Move up to the parent directory
        current_directory = os.path.dirname(current_directory)

    # If the file wasn't found in the current directory or its parent directories, raise an exception
    raise FileNotFoundError(f"File '{file_or_folder_name}' not found in current directory or its parent directories.")
