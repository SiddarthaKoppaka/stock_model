"""
Entry point for data scraping.

Usage:
    python scripts/run_scrape.py
"""

import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.data.scraper import scrape_nifty500_data


def main():
    # Load config
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    setup_logger(
        log_dir=Path(config['paths']['root']) / config['paths']['logs'],
        log_level='INFO'
    )

    # Set seed
    set_seed(config['seed'])

    # Run scraping
    print("=" * 80)
    print("Starting Nifty 500 Data Scraping")
    print("=" * 80)

    scrape_nifty500_data(config)

    print("\nScraping completed!")


if __name__ == "__main__":
    main()
