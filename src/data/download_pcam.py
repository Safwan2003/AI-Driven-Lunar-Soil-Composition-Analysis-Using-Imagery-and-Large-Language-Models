"""
Chang'e 3 PCAM Dataset Downloader
Downloads color-corrected lunar surface images from Planetary Society S3 bucket.
"""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import pandas as pd
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PCAMDownloader:
    """Downloads and organizes Chang'e 3 PCAM imagery."""
    
    BASE_URL = "https://planetary.s3.amazonaws.com/data/change3/pcam.html"
    IMAGE_BASE = "https://planetary.s3.amazonaws.com/data/change3/"
    
    def __init__(self, output_dir: str = "data/pcam"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = []
    
    def scrape_image_urls(self) -> List[Dict[str, str]]:
        """
        Parse the PCAM HTML index to extract PNG image URLs.
        Downloads ONLY Left camera (PCAML) to avoid stereo pair duplicates.
        
        Returns:
            List of dicts with keys: url, filename, observation, timestamp, camera
        """
        logger.info(f"Scraping image list from {self.BASE_URL}")
        response = requests.get(self.BASE_URL, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all links - PNG files are in /pcam/png/ directory
        image_data = []
        seen_files = set()  # Avoid duplicates
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # PNG files are in the /png/ subdirectory
            # Filter for PNG images from Left camera only (PCAML)
            # This avoids stereo pair duplicates (PCAMR is right camera)
            if '/png/' in href and href.endswith('.png') and 'PCAML' in href:
                filename = href.split('/')[-1]
                
                # Skip if already seen
                if filename in seen_files:
                    continue
                seen_files.add(filename)
                
                # Extract metadata from filename
                # Examples:
                # - PCAML-C-001_SCI_N_20131224185403_0006_A_2C.png (color)
                # - PCAML-Q-001_SCI_N_20131224185403_0006_A_2A.png (grayscale panchromatic)
                try:
                    parts = filename.replace('.png', '').split('_')
                    
                    # Determine camera type from middle section
                    camera_type = 'Full Color' if '-C-' in filename else 'Panchromatic'
                    
                    # Get timestamp and observation ID from filename parts
                    timestamp = parts[4] if len(parts) > 4 else 'unknown'
                    observation = parts[5] if len(parts) > 5 else 'unknown'
                    
                    # Version (2A=grayscale, 2B=grayscale, 2C=color demosaiced)
                    version = parts[-1] if len(parts) > 0 else 'unknown'
                    
                    image_data.append({
                        'url': self.IMAGE_BASE + href if not href.startswith('http') else href,
                        'filename': filename,
                        'observation': observation,
                        'timestamp': timestamp,
                        'camera': 'Left',
                        'type': camera_type,
                        'version': version
                    })
                except (IndexError, ValueError) as e:
                    logger.warning(f"Could not parse metadata from {filename}: {e}")
        
        logger.info(f"Found {len(image_data)} unique Left camera PNG images")
        return image_data

    
    def download_image(self, image_info: Dict[str, str]) -> Dict[str, str]:
        """Download a single image."""
        url = image_info['url']
        filename = image_info['filename']
        filepath = self.output_dir / filename
        
        # Skip if already downloaded
        if filepath.exists():
            logger.debug(f"Skipping {filename} (already exists)")
            return {**image_info, 'status': 'skipped', 'path': str(filepath)}
        
        try:
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.debug(f"Downloaded {filename}")
            return {**image_info, 'status': 'success', 'path': str(filepath)}
        
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return {**image_info, 'status': 'failed', 'error': str(e)}
    
    def download_all(self, max_images: int = None, max_workers: int = 6) -> pd.DataFrame:
        """
        Download all PCAM images in parallel.
        
        Args:
            max_images: Limit number of images (for testing). None = download all.
            max_workers: Number of parallel download threads.
        
        Returns:
            DataFrame with download metadata
        """
        image_list = self.scrape_image_urls()
        
        if max_images:
            image_list = image_list[:max_images]
            logger.info(f"Limiting to {max_images} images for testing")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.download_image, img): img for img in image_list}
            
            with tqdm(total=len(image_list), desc="Downloading PCAM images") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
        
        # Create metadata DataFrame
        df = pd.DataFrame(results)
        
        # Save metadata
        metadata_path = self.output_dir / "metadata.csv"
        df.to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Print summary
        success_count = (df['status'] == 'success').sum()
        skipped_count = (df['status'] == 'skipped').sum()
        failed_count = (df['status'] == 'failed').sum()
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Download Summary:")
        logger.info(f"  Success: {success_count}")
        logger.info(f"  Skipped: {skipped_count}")
        logger.info(f"  Failed: {failed_count}")
        logger.info(f"{'='*50}\n")
        
        return df


def main():
    """CLI interface for PCAM downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Chang'e 3 PCAM lunar images")
    parser.add_argument('--output', '-o', default='data/pcam', help='Output directory')
    parser.add_argument('--limit', '-l', type=int, help='Limit number of images (for testing)')
    parser.add_argument('--threads', '-t', type=int, default=6, help='Number of download threads')
    
    args = parser.parse_args()
    
    downloader = PCAMDownloader(output_dir=args.output)
    df = downloader.download_all(max_images=args.limit, max_workers=args.threads)
    
    print(f"\nDataset ready at: {args.output}")
    print(f"Total images: {len(df)}")


if __name__ == "__main__":
    main()
