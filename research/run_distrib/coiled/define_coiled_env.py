import subprocess

reqs = subprocess.run(
    "poetry export --without-hashes", stdout=subprocess.PIPE, shell=True
).stdout.decode("utf-8")
reqs = reqs.split("\n")
reqs = [r.split(";")[0][:-1] for r in reqs if "jax" not in r]
with open("coiled-requirements.txt", "r") as f:
    reqs.extend([L.strip() for L in f.readlines()])
reqs = [r for r in reqs if len(r) > 0]
pip_installs = "\n".join([f"    - {r}" for r in reqs])
environment_yml = f"""# THIS IS AUTOGENERATED BY build_coiled_env.py  DO NOT MODIFY.
# THIS IS AUTOGENERATED BY build_coiled_env.py  DO NOT MODIFY.
# THIS IS AUTOGENERATED BY build_coiled_env.py  DO NOT MODIFY.
# THIS IS AUTOGENERATED BY build_coiled_env.py  DO NOT MODIFY.
name: confirm
channels:
  - conda-forge
  - nvidia
dependencies:
  - python=3.10
  - nvidia:cuda-toolkit=11.8
  - conda-forge:cudnn=8.4
  - pip
  - pip:
{pip_installs}
"""
print(environment_yml)
with open("env-coiled.yml", "w") as f:
    f.write(environment_yml)
