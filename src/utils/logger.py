import logging
import os

def setup_logger(logging_dir: str):
    """Setup logging configuration."""
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{logging_dir}/pipeline.log"),
            logging.StreamHandler()
        ],
    )