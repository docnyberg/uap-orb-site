import csv, sys
path = sys.argv[1]
with open(path, newline="", encoding="utf-8", errors="replace") as f:
    r = csv.reader(f)
    for row in r:
        # print one row per line, joined with pipes => stable & readable
        print(" | ".join(row))