import builtins
import pyclbr

builtins.__dict__["profile"] = lambda x: x
