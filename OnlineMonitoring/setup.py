# with open("/sys/firmware/devicetree/base/serial-number", "r") as f:
#     serial = str(f.read().rstrip("\x00"))

# with open("./odin/core.py", "w") as f:
#     f.write("serial_num = " + serial)


from distutils.core import setup
from Cython.Build import cythonize

pyfiles = [
            'demo/wave.py',
           'demo/ctrl_light.py',
           'demo/det_infer.py',
           'demo/main.py',
           'demo/label.py',
           'demo/compare.py',
           'demo/postprocess.py',
           'demo/source_rc.py',
           'demo/tcp.py',
           'demo/window.py',
           'demo/ocr_utils.py']
           

for py in pyfiles:
    setup(ext_modules = cythonize(py))

