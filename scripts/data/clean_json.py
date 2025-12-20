from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["nq", "webquestions", "triviaqa"], required=True)
    parser.add_argument("--split", type=str, choices=["train", "dev", "test"], required=True)
    return parser.parse_args()

all_chars = set()

def clean(text: str) -> str:
    """Clean text by replacing unicode characters."""
    # global all_chars
    # all_chars |= set(text)

    return (
        text.replace("\u000a", " ")     # line feed
            .replace("\u00a0", " ")     # nbsp (no-break space)
            .replace("\u00ad", "")      # soft hyphen (for syllables)
            .replace("\u00bd", "1/2")   # vulgar fraction one half
            .replace("\u00be", "3/4")   # vulgar fraction three quarters
            .replace("\u02bb", "")      # modifier letter turned comma (as in, Hawaii)
            .replace("\uff03", "#")     # fullwidth number sign
            .replace("\u2009", " ")     # thin space
            .replace("\u200a", " ")     # hair space
            .replace("\u200b", "")      # zwsp (zero-width space)
            .replace("\u200e", "")      # lrm (left-to-right mark)
            .replace("\u2044", "/")     # fraction slash
            .replace("\u2011", "-")     # non-breaking hyphen
            .replace("\u2013", "-")     # en dash
            .replace("\u2212", "-")     # minus sign
            .replace("\u2018", "'")     # left single quotation mark
            .replace("\u2019", "'")     # right single quotation mark
            .replace("\u201c", '"')     # left double quotation mark
            .replace("\u201d", '"')     # right double quotation mark
            .replace("\u2022", "")      # bullet point
            .replace("\ufffc", "")      # object replacement character
            .strip()                    # remove leading/trailing whitespace
    )

def main() -> None:
    args = parse_args()
    df_path = Path("data") / args.dataset / f"{args.dataset}-{args.split}.jsonl"
    raw_df = pd.read_json(df_path, lines=True)
    raw_df["question"] = raw_df["question"].apply(clean)
    raw_df["answers"] = raw_df["answers"].apply(lambda x: [clean(a) for a in x])
    raw_df.to_json(df_path, orient="records", lines=True)

    # import unicodedata

    # for c in sorted(all_chars):
    #     print(f"Character: {repr(c)} | Unicode: U+{ord(c):04X} | Name: {unicodedata.name(c, 'UNKNOWN')}")

if __name__ == "__main__":
    main()
