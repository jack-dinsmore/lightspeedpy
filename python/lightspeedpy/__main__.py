import numpy as np
import os, subprocess

if __name__ == "__main__":
    print("""
Lightspeed has many tools. You can call them with the pattern

python -m lightspeedpy.TOOL_NAME ARGUMENTS

or import them as modules in python code with the code

import lightspeedpy.TOOL_NAME

Below is a list of the available tools. For more information on each, run

python -m lightspeedpy.TOOL_NAME -h
""")
    module_dir = os.path.dirname(__file__)
    modules = []
    for f in os.listdir(module_dir):
        if f == "__pycache__": continue
        if f == "template": continue
        if os.path.isdir(os.path.join(module_dir, f)):
            modules.append(f)
    modules = np.sort(modules)

    second_col_length = max([len(m) for m in modules]) + 4
    for m in modules:
        # Run command and capture output as a string
        result = subprocess.run(["python3", "-m", f"lightspeedpy.{m}", "-h"], capture_output=True, text=True)
        output = result.stdout
        start = output.find("\n\n")+1
        stop = output[start:].find("\n\n")+start
        description = output[start+1:stop]
        print(m, " " * (second_col_length - len(m)), description)