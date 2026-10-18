"""ONE command: run the entire pipeline, then deploy the API + dashboard.

    python runall.py                 # pipeline + API on http://127.0.0.1:8000
    python runall.py --no-api        # pipeline only
    python runall.py --skip-pipeline # API only (reuse existing outputs)
    python runall.py --open --port 9000
"""
import argparse
import threading
import webbrowser

from src.pipeline import run_pipeline

BANNER = "\033[96m\n=== Climate Trend Analyzer ===\033[0m"


def main():
    ap = argparse.ArgumentParser(description="Run the full Climate Trend Analyzer pipeline and deploy the API.")
    ap.add_argument("--skip-pipeline", action="store_true", help="skip analysis, only start the API")
    ap.add_argument("--no-api", action="store_true", help="run the analysis only")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true", help="open the dashboard in your browser")
    a = ap.parse_args()

    print(BANNER)
    if not a.skip_pipeline:
        run_pipeline()
    if a.no_api:
        return

    import uvicorn  # imported late so --no-api works without it
    url = f"http://{'localhost' if a.host == '0.0.0.0' else a.host}:{a.port}"
    print(f"\n\033[92mAPI live:\033[0m {url}  |  docs: {url}/docs  |  Ctrl+C to stop\n")
    if a.open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("api.app:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
