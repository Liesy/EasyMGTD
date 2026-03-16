import os
from dotenv import load_dotenv

# Automatically load environment variables from .env file when the package is imported
load_dotenv()

from .auto import AutoDetector, ModelBasedDetector, AutoExperiment
