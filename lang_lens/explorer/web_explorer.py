from ..lens import Lens
from .explorer import Explorer
import os
import dill
import streamlit.web.cli as stcli
import sys

class WebExplorer(Explorer):
    def __init__(self, lens: Lens):
        self.lens = lens
        self.inspections = []

    def inspect(self, vec):
        inspection = self.lens.inspect(vec)
        self.inspections.append(inspection)

    def launch(self):
        app_dir = os.path.join(os.path.dirname(__file__), "../explorer_app")
        app_path = os.path.join(app_dir, "main.py")

        pickle_path = os.path.join(app_dir, "explorer_state.pkl")
        with open(pickle_path, "wb") as f:
            dill.dump(self, f)

        # Run streamlit in the same process (blocking call)
        sys.argv = ["streamlit", "run", app_path, "--", pickle_path]
        stcli.main()
