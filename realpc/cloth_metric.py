"""Optional deterministic native acceleration of the same colored projection."""
import ctypes,hashlib,subprocess,os
from pathlib import Path
import numpy as np
_LIB=None

def project(y,edges,rest,inv,limit,sweeps):
    global _LIB
    if _LIB is False:return None
    if _LIB is None:
        source=Path(__file__).with_suffix('.cpp');key=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        cache=Path.home()/'.cache/realpc';cache.mkdir(parents=True,exist_ok=True)
        target=cache/f'cloth_metric_{key}.so'
        if not target.exists():
            temp=target.with_suffix(f'.{os.getpid()}.so')
            try:
                subprocess.run(['g++','-O3','-ffp-contract=off','-shared','-fPIC',str(source),'-o',str(temp)],check=True,capture_output=True)
            except (OSError,subprocess.CalledProcessError):
                _LIB=False
                return None  # Identical NumPy implementation remains available.
            os.replace(temp,target)
        _LIB=ctypes.CDLL(str(target));p=ctypes.c_void_p
        _LIB.project_metric.argtypes=[p,p,p,p,ctypes.c_int64,ctypes.c_double,ctypes.c_int];_LIB.project_metric.restype=None
    y=np.ascontiguousarray(y,dtype=np.float64).copy();edges=np.ascontiguousarray(edges,dtype=np.int64);rest=np.ascontiguousarray(rest,dtype=np.float64);inv=np.ascontiguousarray(inv,dtype=np.float64)
    _LIB.project_metric(y.ctypes.data,edges.ctypes.data,rest.ctypes.data,inv.ctypes.data,len(edges),limit,sweeps)
    return y
