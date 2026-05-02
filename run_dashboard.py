"""Launch the Streamlit dashboard."""
import os
import subprocess
import sys

if __name__ == "__main__":
    port = os.getenv("PORT", "8501")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py",
         "--server.port", port,
         "--browser.gatherUsageStats", "false"],
        check=True,
    )
