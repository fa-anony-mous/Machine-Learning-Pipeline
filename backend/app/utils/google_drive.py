import requests
import os

def download_file_from_google_drive(file_id, destination):
    """
    Download a file from Google Drive using its file ID
    Handles large files that might show a confirmation page
    """
    # Define URLs
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()

    # First request to get cookies and confirmation code if needed
    response = session.get(url, stream=True)
    
    # Check if there's a download warning (for large files)
    # Google Drive shows a confirmation page for large files
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            # Add confirmation code to URL
            url = f"{url}&confirm={value}"
            break
    
    # Second request with confirmation if needed
    response = session.get(url, stream=True)
    response.raise_for_status()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Save the file
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)
    
    return destination