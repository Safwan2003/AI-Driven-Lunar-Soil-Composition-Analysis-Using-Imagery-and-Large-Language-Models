import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import argparse
import concurrent.futures

def get_image_links(url):
    """Fetches image URLs from the Planetary Society index pages."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.lower().endswith(('.png', '.jpg')):
                full_url = urljoin(url, href)
                links.append(full_url)
        return links
    except Exception as e:
        print(f"Error fetching links from {url}: {e}")
        return []

def download_image(url, save_dir):
    """Helper function to download a single image."""
    filename = os.path.basename(url)
    save_path = os.path.join(save_dir, filename)
    
    if os.path.exists(save_path):
        return f"Skipped (exists): {filename}"
        
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return f"Downloaded: {filename}"
        else:
            return f"Failed: {url} (Status {response.status_code})"
    except Exception as e:
        return f"Error {url}: {str(e)}"

def download_images_parallel(image_urls, save_dir, max_workers=10, limit=None):
    """Downloads images in parallel."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"Starting download of up to {limit if limit else 'all'} images to {save_dir} with {max_workers} workers...")
    
    urls_to_download = image_urls[:limit] if limit else image_urls
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(download_image, url, save_dir): url for url in urls_to_download}
        
        count = 0
        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()
            # print(result) # Optional: comment out to reduce noise
            count += 1
            if count % 10 == 0:
                print(f"Processed {count}/{len(urls_to_download)}")
                
    print(f"Completed. {len(urls_to_download)} images processed.")

def main():
    parser = argparse.ArgumentParser(description="Download lunar soil images.")
    parser.add_argument('--limit', type=int, default=2000, help='Number of images to download per category')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel download threads')
    args = parser.parse_args()

    # Determine base directory (project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    pcam_dir = os.path.join(data_dir, 'pcam')
    tcam_dir = os.path.join(data_dir, 'tcam')

    # URLs from the Planetary Society article
    pcam_index_url = 'http://planetary.s3.amazonaws.com/data/change3/pcam.html'
    tcam_index_url = 'http://planetary.s3.amazonaws.com/data/change3/tcam.html'

    print(f"Saving data to: {data_dir}")

    print("Fetching PCAM links...")
    pcam_links = get_image_links(pcam_index_url)
    print(f"Found {len(pcam_links)} PCAM images.")
    if pcam_links:
        download_images_parallel(pcam_links, pcam_dir, max_workers=args.workers, limit=args.limit)

    print("Fetching TCAM links...")
    tcam_links = get_image_links(tcam_index_url)
    print(f"Found {len(tcam_links)} TCAM images.")
    if tcam_links:
        download_images_parallel(tcam_links, tcam_dir, max_workers=args.workers, limit=args.limit)

if __name__ == "__main__":
    main()
