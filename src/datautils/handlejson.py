import ijson
from pathlib import Path
import sys

def count_json_array(filename: str) -> int:
    count = 0
    with Path.open(filename, "rb") as f:
        parser = ijson.items(f, "item")     # Stream parse the array, counting each item
        for _ in parser:
            count += 1
    return count

def main():
    filename = sys.argv[1]
    print(count_json_array(filename))

if __name__ == "__main__":
    main()

# maybe delete this
