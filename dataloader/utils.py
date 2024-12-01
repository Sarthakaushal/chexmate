import os
import json
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

def setup_kaggle_credentials(api_key, api_username):
    """
    Sets up Kaggle credentials using provided API key and username.
    
    Args:
        api_key (str): Your Kaggle API key
        api_username (str): Your Kaggle username
    """
    # Create .kaggle directory if it doesn't exist
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Create kaggle.json with credentials
    kaggle_cred = {
        "username": api_username,
        "key": api_key
    }
    
    # Save credentials to kaggle.json
    cred_path = os.path.join(kaggle_dir, 'kaggle.json')
    with open(cred_path, 'w') as f:
        json.dump(kaggle_cred, f)
    
    # Set appropriate permissions
    os.chmod(cred_path, 0o600)

def download_dataset(dataset_name, save_path='./data', api_key=None, api_username=None):
    """
    Downloads a dataset from Kaggle if it doesn't already exist locally.
    
    Args:
        dataset_name (str): Name of the Kaggle dataset in format 'owner/dataset-name'
        save_path (str): Local directory to save the dataset
        api_key (str, optional): Kaggle API key. If provided, will set up credentials
        api_username (str, optional): Kaggle username. Required if api_key is provided
    """
    # Set up credentials if provided
    if api_key and api_username:
        setup_kaggle_credentials(api_key, api_username)
    
    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Check if dataset already exists
    dataset_folder = dataset_name.split('/')[-1]
    if not os.path.exists(os.path.join(save_path, dataset_folder)):
        print(f"Downloading {dataset_name} to {save_path}...")
        try:
            api.dataset_download_files(
                dataset_name,
                path=save_path,
                unzip=True
            )
            print("Download completed successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
    else:
        print(f"Dataset already exists in {save_path}")
# sample usage

# def main():
#     """Example usage of dataset download function"""
#     # Example 1: Using environment variables for authentication
#     dataset_name = "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"
#     save_path = "/home/sk4858/chexmate/data/global"
#     download_dataset(dataset_name, save_path)
    
#     # # Example 2: Explicitly providing credentials
#     # api_key = "your_kaggle_api_key"
#     # api_username = "your_kaggle_username" 
#     # dataset_name = "paultimothymooney/chest-xray-pneumonia"
#     # save_path = "./data"
#     # download_dataset(
#     #     dataset_name,
#     #     save_path,
#     #     api_key=api_key,
#     #     api_username=api_username
#     # )

# if __name__ == "__main__":
#     main()
