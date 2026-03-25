from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WEB_SERVER = ROOT / "src" / "marl" / "visualization" / "web_server.py"
spec = spec_from_file_location("web_server_entry", WEB_SERVER)
module = module_from_spec(spec)
spec.loader.exec_module(module)
app = module.app
