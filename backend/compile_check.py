import py_compile
import sys
import traceback

try:
    py_compile.compile('d:/app_file/translat/backend/main.py', doraise=True)
    print("SUCCESS")
except Exception:
    with open("compile_error_v2.txt", "w") as f:
        traceback.print_exc(file=f)
    print("FAILURE")
