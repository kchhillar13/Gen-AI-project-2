"""
Root-level entry point for Streamlit Cloud.
Streamlit Cloud is configured to run 'app.py' at the project root.
This file simply delegates execution to src/app.py so the new
folder structure works without changing any Cloud settings.
"""
import runpy
import sys
from pathlib import Path

# Add src/ to the Python path so imports inside src/app.py resolve correctly
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

# Execute src/app.py as the main script
runpy.run_path(str(SRC_DIR / "app.py"), run_name="__main__")
