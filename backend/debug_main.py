import sys
import traceback

with open("status.txt", "w", encoding="utf-8") as f:
    f.write("INIT: Debug Script Started\n")

try:
    import main
    with open("status.txt", "a", encoding="utf-8") as f:
        f.write("SUCCESS: Main Imported\n")
except Exception as e:
    with open("status.txt", "a", encoding="utf-8") as f:
        f.write("ERROR: Import Failed\n")
        f.write(str(e) + "\n")
        f.write(traceback.format_exc())
