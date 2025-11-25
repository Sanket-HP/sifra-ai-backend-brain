import os

folders = [
    "utils",
    "core",
    "core/hdp_fusionnet",
    "core/hds_unity",
    "tasks",
    "data",
    "api",
    "ui",
    "config"
]

for f in folders:
    path = os.path.join(f, "__init__.py")
    if not os.path.exists(path):
        open(path, "w").close()
        print("Created:", path)
    else:
        print("Exists:", path)
print("Initialization files creation complete.")