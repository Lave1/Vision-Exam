import requests
from bs4 import BeautifulSoup
import os
import shutil
import time
import logging

# Set the query and the number of images you want to download
query = "sedia"
num_images = 1000

# Set the download directory for images
download_directory = "C:/Users/AlessandroLaVeglia/repo_git/Vision-Exam/immagini"

# Ensure the download directory exists; create it if it doesn't
if not os.path.exists(download_directory):
    os.makedirs(download_directory)

# Create a logger for error messages
logging.basicConfig(filename='error.log', level=logging.ERROR)

# Function to download and save an image
def download_image(image_url, file_path):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            # Check the content type to ensure it's an image
            content_type = response.headers.get("content-type")
            if content_type.startswith("image"):
                with open(file_path, "wb") as file:
                    file.write(response.content)
            else:
                logging.error(f"Invalid content type for URL: {image_url}")
        else:
            logging.error(f"Failed to download image from URL: {image_url}")
    except Exception as e:
        logging.error(f"Error downloading image: {str(e)}")

# Construct the search URL for Bing Image Search
url = f"https://www.bing.com/images/search?q={query}"

# Initialize variables for pagination
start = 1
results_per_page = 50  # Bing typically shows 50 images per page

# Download the images and save them to the specified directory
while len(os.listdir(download_directory)) < num_images:
    try:
        # Send an HTTP GET request for the current page
        response = requests.get(url + f'&first={start}')
        if response.status_code != 200:
            logging.error(f"Failed to retrieve search results from page {start / results_per_page}.")
            break

        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")

        # Find and extract image URLs using the 'data-src' attribute
        image_elements = soup.find_all("img", class_="mimg")
        image_urls = [img.get("data-src") for img in image_elements if img.get("data-src")]

        # Download images from the current page
        for i, image_url in enumerate(image_urls):
            if len(os.listdir(download_directory)) >= num_images:
                break
            file_path = os.path.join(download_directory, f"{query}_{start + i}.jpg")
            download_image(image_url, file_path)

        # Increment the start value for the next page
        start += results_per_page

        # Add a delay to avoid being blocked
        time.sleep(2)

    except Exception as e:
        logging.error(f"Error processing page {start / results_per_page}: {str(e)}")
        break

print(f"Downloaded {len(os.listdir(download_directory))} images to {download_directory}.")

# per pulire la cartella immagini


# folder_to_clear = "C:/Users/AlessandroLaVeglia/repo_git/Vision-Exam/immagini"
# # Ensure the folder exists
# if os.path.exists(folder_to_clear):
#     # List all files and subfolders in the folder
#     folder_contents = os.listdir(folder_to_clear)

#     # Loop through each item in the folder
#     for item in folder_contents:
#         item_path = os.path.join(folder_to_clear, item)

#         # Check if the item is a file and delete it
#         if os.path.isfile(item_path):
#             os.remove(item_path)
#         # If it's a subfolder, use shutil.rmtree to delete it and its contents
#         elif os.path.isdir(item_path):
#             shutil.rmtree(item_path)
    
#     print(f"Folder '{folder_to_clear}' has been cleared.")
# else:
#     print(f"Folder '{folder_to_clear}' does not exist.")