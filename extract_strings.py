import sys
import re

def extract_strings(filename, min_len=4):
    try:
        with open(filename, "rb") as f:
            data = f.read()
            # Find sequences of printable characters
            strings = re.findall(b"[\x20-\x7E]{" + str(min_len).encode() + b",}", data)
            for s in strings:
                print(s.decode("ascii"))
    except Exception as e:
        print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_strings(sys.argv[1])
