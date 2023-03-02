import shutil
import subprocess

import tomli
import tomli_w


def copy_simple():
    # had to change the env name for environment.yml so we shouldn't overwrite
    # shutil.copy2('environment.yml', 'imprint/environment.yml')
    #
    shutil.copy2(".gitignore", "imprint/.gitignore")
    shutil.copy2(".gitleaks.toml", "imprint/.gitleaks.toml")
    shutil.copy2(".vscode/settings.json", "imprint/.vscode/settings.json")
    shutil.copy2("setup.cfg", "imprint/setup.cfg")
    shutil.copy2("requirements-jax-cuda.txt", "imprint/requirements-jax-cuda.txt")


def transfer_pyproject():
    with open("pyproject.toml", "rb") as f:
        confirm_pp = tomli.load(f)

    with open("imprint/pyproject.toml", "rb") as f:
        imprint_pp = tomli.load(f)

    def copy_entry(name):
        keys = name.split(".")
        imprint_dict = imprint_pp
        confirm_dict = confirm_pp
        for k in keys[:-1]:
            if k not in imprint_dict:
                imprint_dict[k] = {}
            imprint_dict = imprint_dict[k]
            confirm_dict = confirm_dict[k]
        imprint_dict[keys[-1]] = confirm_dict[keys[-1]]

    copy_entry("tool.poetry.dependencies")
    copy_entry("tool.poetry.group.test")
    copy_entry("tool.poetry.group.dev")

    with open("imprint/pyproject.toml", "wb") as f:
        tomli_w.dump(imprint_pp, f)

    subprocess.call("poetry lock", shell=True, cwd="imprint")


def main():
    copy_simple()
    transfer_pyproject()


if __name__ == "__main__":
    main()
