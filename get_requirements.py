import os
import ast
import pkgutil
import sys
import subprocess
from pathlib import Path
import importlib.metadata

# Get the set of standard library modules
def get_stdlib_modules():
    return set(sys.builtin_module_names).union(
        {name for _, name, ispkg in pkgutil.iter_modules() if not ispkg}
    )

# Recursively collect all Python files from a folder
def get_all_python_files(folder_path):
    return list(Path(folder_path).rglob("*.py"))

# Extract top-level imports from a Python file using AST
def extract_imports_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            node = ast.parse(f.read(), filename=str(filepath))
        except Exception as e:
            print(f"Skipping {filepath} due to parse error: {e}")
            return set()

    imports = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for alias in n.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                imports.add(n.module.split('.')[0])
    return imports

# Determine which packages are third-party
def filter_third_party_modules(modules, stdlib_modules):
    third_party = set()
    for mod in modules:
        if mod in stdlib_modules:
            continue
        try:
            dist = importlib.metadata.distribution(mod)
            third_party.add(dist.metadata["Name"])
        except importlib.metadata.PackageNotFoundError:
            pass  # Could not find a distribution for this module
    return third_party

# Write to requirements.txt
def write_requirements_file(packages, output_file="requirements.txt"):
    with open(output_file, "w") as f:
        for package in sorted(packages):
            try:
                version = importlib.metadata.version(package)
                f.write(f"{package}=={version}\n")
            except importlib.metadata.PackageNotFoundError:
                f.write(f"{package}\n")

def main(target_folder):
    print(f"📁 Scanning folder: {target_folder}")
    py_files = get_all_python_files(target_folder)
    all_imports = set()

    for file in py_files:
        all_imports |= extract_imports_from_file(file)

    stdlib_modules = get_stdlib_modules()
    third_party = filter_third_party_modules(all_imports, stdlib_modules)

    print(f"✅ Found {len(third_party)} third-party packages.")
    write_requirements_file(third_party)
    print("📦 requirements.txt generated!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate requirements.txt from imports.")
    parser.add_argument("folder", help="Path to the folder to scan")
    args = parser.parse_args()
    main(args.folder)
