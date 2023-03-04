Line Profiler is great and comes installed with our development environment. `kernprof -lbv filename.py`, add `@profile` to your functions.

py-spy is also useful and does stack sampling profiling: `sudo py-spy record --format speedscope -o prof.out -- python bench.py`

This version was useful for profiling multi-threaded code that uses asyncio: `sudo py-spy record --rate 1000 --nonblocking --threads --idle --format speedscope -o prof.out -- python bench.py`

Open https://www.speedscope.app and load the profile results.