import builtins

if "profile" not in builtins.__dict__:
    builtins.__dict__["profile"] = lambda x: x
