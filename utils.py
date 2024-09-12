import os
import zipfile
import requests

def remove_first_last_line(text):
        lines = text.splitlines()
        # Remove first and last line
        return "\n".join(lines[1:-1]) if len(lines) > 2 else ""
    
def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)      

def getUrl(url):
    """ Given a URL it returns its body """
    response = requests.get(url)
    return response.json()



def process_zip_files_to_faiss(folder_path):
    # Ensure proper error handling for file operations
    all_texts = []  # Store all texts to embed later
    file_paths = [] # Store file paths for reference

    # Step 1: Extract README.md from the zip files and gather the texts
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.zip'):
            zip_file_path = os.path.join(folder_path, file_name)
            print(f"Processing: {zip_file_path}")

            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    # Extract all files to a temporary directory
                    temp_extract_path = os.path.join(folder_path, file_name.replace('.zip', ''))
                    os.makedirs(temp_extract_path, exist_ok=True)

                    # Extract all files into the temp directory
                    zip_ref.extractall(temp_extract_path)

                    # Search for README.md within the extracted files
                    for root, _, files in os.walk(temp_extract_path):
                        if 'README.md' in files:
                            readme_path = os.path.join(root, 'README.md')
                            print(f"Found README.md in {readme_path}")

                            # Read the content of README.md
                            with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                all_texts.append(content)
                                file_paths.append(readme_path)

                            # Optionally, remove the README.md file from the temp directory after processing
                            os.remove(readme_path)

                    # Clean up the temp extraction directory
                    for root, dirs, files in os.walk(temp_extract_path, topdown=False):
                        for name in files:
                            os.remove(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                    os.rmdir(temp_extract_path)

            except zipfile.BadZipFile:
                print(f"Skipping invalid zip file: {zip_file_path}")
            except Exception as e:
                print(f"Error processing {zip_file_path}: {e}")

    return all_texts, file_paths