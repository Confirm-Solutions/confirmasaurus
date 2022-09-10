import subprocess

for i in range(4, 17):
    print(i, i + 1)
    size = 1000000
    begin = i * size
    end = (i + 1) * size
    cmd = f"python research/berry/run4d.py --begin {begin} --end {end}"

    print(cmd)
    print(cmd)
    print(cmd)

    subprocess.call(cmd, shell=True)
