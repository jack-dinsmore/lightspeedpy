import numpy as np
import os

if __name__ == "__main__":
    print("Lightspeed has many callable modules.")

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
        init_file = os.path.join(module_dir, m, "__init__.py")
        if not os.path.exists(init_file): continue

        description = ""
        with open(init_file, 'r') as f:
            for line in f.readlines():
                if not line.lower().startswith("# description: "): continue
                description = line[len("# description: "):]
                if description.endswith("\n"): description=description[:-1]
                break
        print(m, " " * (second_col_length - len(m)), description)