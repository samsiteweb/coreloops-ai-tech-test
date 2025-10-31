"""Data loader for GCS transaction files."""

import os
import logging
import requests
import pandas as pd
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataLoader:
    
    def __init__(self, base_url: str, local_data_dir: str):
        self.base_url = base_url.rstrip('/')
        self.local_data_dir = Path(local_data_dir)
        self.local_data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, remote_path: str, local_path: Path, force: bool = False) -> bool:
        if local_path.exists() and not force:
            logger.debug(f"File already exists: {local_path}")
            return True
        
        url = f"{self.base_url}/{remote_path}"
        
        try:
            logger.info(f"Downloading: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save file
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded: {local_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def list_available_dates(
        self, 
        start_date: str = "2024-10-01", 
        end_date: Optional[str] = None,
        max_attempts: int = 10
    ) -> List[str]:
        # Probe for available files since we can't list bucket contents
        available_dates = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            # Default to checking up to today + 7 days
            end = datetime.now() + timedelta(days=7)
        
        consecutive_failures = 0
        
        while current_date <= end and consecutive_failures < max_attempts:
            date_str = current_date.strftime("%Y-%m-%d")
            url = f"{self.base_url}/data/{date_str}.csv"
            
            try:
                response = requests.head(url, timeout=10)
                if response.status_code == 200:
                    available_dates.append(date_str)
                    consecutive_failures = 0
                    logger.debug(f"Found file for date: {date_str}")
                else:
                    consecutive_failures += 1
            except requests.exceptions.RequestException:
                consecutive_failures += 1
            
            current_date += timedelta(days=1)
        
        logger.info(f"Found {len(available_dates)} available date files")
        return available_dates
    
    def download_transactions(
        self, 
        dates: Optional[List[str]] = None,
        start_date: str = "2024-10-01",
        end_date: Optional[str] = None
    ) -> List[Path]:
        if dates is None:
            dates = self.list_available_dates(start_date, end_date)
        
        downloaded_files = []
        
        for date in dates:
            remote_path = f"data/{date}.csv"
            local_path = self.local_data_dir / f"{date}.csv"
            
            if self.download_file(remote_path, local_path):
                downloaded_files.append(local_path)
        
        logger.info(f"Downloaded {len(downloaded_files)}/{len(dates)} files")
        return downloaded_files
    
    def download_fx_rates(self, fx_rates_file: str = "fx_rates.csv") -> Optional[Path]:
        local_path = self.local_data_dir / fx_rates_file
        
        if self.download_file(fx_rates_file, local_path):
            return local_path
        return None
    
    def load_transactions(self, file_paths: List[Path]) -> pd.DataFrame:
        if not file_paths:
            logger.warning("No files to load")
            return pd.DataFrame()
        
        dfs = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                # Extract date from filename (format: YYYY-MM-DD.csv)
                date_str = file_path.stem  # Gets filename without extension
                df['file_date'] = date_str
                dfs.append(df)
                logger.debug(f"Loaded {len(df)} rows from {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        if not dfs:
            logger.error("No files loaded")
            return pd.DataFrame()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded total of {len(combined_df):,} transactions from {len(dfs)} files")
        
        return combined_df
    
    def load_fx_rates(self, fx_rates_path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(fx_rates_path)
            logger.info(f"Loaded {len(df)} FX rate records")
            return df
        except Exception as e:
            logger.error(f"Failed to load FX rates from {fx_rates_path}: {e}")
            return pd.DataFrame()


def load_all_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = DataLoader(
        base_url=config['data']['gcs_base_url'],
        local_data_dir=config['data']['local_data_dir']
    )
    
    # Download FX rates
    fx_path = loader.download_fx_rates(config['data']['fx_rates_file'])
    if fx_path is None:
        raise RuntimeError("Failed to download FX rates")
    
    # Download transactions
    transaction_files = loader.download_transactions()
    if not transaction_files:
        raise RuntimeError("Failed to download any transaction files")
    
    # Load data
    transactions_df = loader.load_transactions(transaction_files)
    fx_rates_df = loader.load_fx_rates(fx_path)
    
    return transactions_df, fx_rates_df




