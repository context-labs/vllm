# 4090
python3 -m venv .venv
source .venv/bin/activate
pip install jinja2
export MAX_JOBS=6
sudo apt install ninja-build
pip install -e .
# For running tests:
pip install -r requirements/test.txt
pip install pytest
pip install pytest_asyncio