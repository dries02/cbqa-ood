# import unicodedata
from argparse import ArgumentParser, Namespace
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import regex as re


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["nq", "webquestions", "triviaqa"], required=True)
    parser.add_argument("--split", type=str, choices=["train", "dev", "test"], required=True)
    return parser.parse_args()


def decode_escaped_utf8(text: str) -> str:
    return re.sub(
        r"\\x([0-9a-fA-F]{2})",
        lambda m: bytes.fromhex(m.group(1)).decode("utf-8", errors="ignore"),
        text)


# all_chars = set()
pattern = re.compile(r"\[\d+\]")       # Wikipedia reference tokens
# pattern2 = re.compile(r"\(.*?\)")      # usually clarification or disambiguation etc

def clean(text: str) -> str:
    """Clean text by replacing unicode characters."""
    # global all_chars
    # all_chars |= set(text)
    text = text.encode("utf-16", "surrogatepass").decode("utf-16")      # fix rendering
    text = unquote(text)
    text = decode_escaped_utf8(text)

    return " ".join(
        re.sub(pattern, "", text)
            .replace("\u000a", " ")     # line feed
            .replace("\u00a0", " ")     # nbsp (no-break space)
            .replace("\u00a9", "")      # copyright
            .replace("\u00ad", "")      # soft hyphen (for syllables)
            .replace("\u00ae", "")      # registered trademark
            .replace("\u00bd", "1/2")   # vulgar fraction one half
            .replace("\u00be", "3/4")   # vulgar fraction three quarters
            .replace("\u017f", "s")     # long s to s
            .replace("\u02bb", "")      # modifier letter turned comma (as in, Hawaii)
            .replace("\u2009", " ")     # thin space
            .replace("\u200a", " ")     # hair space
            .replace("\u200b", "")      # zwsp (zero-width space)
            .replace("\u200e", "")      # lrm (left-to-right mark)
            .replace("\u2011", "-")     # non-breaking hyphen
            .replace("\u2013", "-")     # en dash
            .replace("\u2018", "'")     # left single quotation mark
            .replace("\u2019", "'")     # right single quotation mark
            .replace("\u201c", "")      # left double quotation mark         -- removed!
            .replace("\u201d", "")      # right double quotation mark        -- removed!
            .replace("\u2122", "")      # trademark
            .replace('"', "")           # remove quotes!
            .replace("`", "'")          # replace backticks
            .replace("\u2022", "")      # bullet point
            .replace("\u2044", "/")     # fraction slash
            .replace("\u2212", "-")     # minus sign
            .replace("\uff03", "#")     # fullwidth number sign
            .replace("\ufffc", "")      # object replacement character
            .split(),                   # normalize whitespace
    )


# def clean_experimental(raw_answers: list[str]) -> list[str]:
#     answers = []
#     seen = set()

#     for raw_answer in raw_answers:
#         if not any(c.isascii() and (c.isalpha() or c.isdigit()) for c in raw_answer):  # foreign language?
#             continue

#         answer = re.sub(pattern2, "", raw_answer)       # remove "(xxx)", Wikipedia artefact
#         answer = answer.replace("0's", "0s")            # probably ungrammatical decade
#         answer = answer.replace("\\", " ")              # no backslashes, likely latex artefact

#         if "UN/LOCODE" in answer:
#             continue                                # United Nations Code for Trade and Transport Locations

#         if "ATCvet code" in answer or "ATC code" in answer:
#             continue                                # pharmacy code

#         if re.match(r"ISO \d+", answer):            # skip ISO country codes (redirect pages)
#             continue

#         answer = answer.strip("'").strip()
#         answer = " ".join(answer.split())           # normalize whitespace

#         if answer and answer.lower() not in seen:
#             seen.add(answer.lower())
#             answers.append(answer)                                  # no duplicates

#     if not answers:
#         msg = f"There are no valid answers... {raw_answers}"
#         raise ValueError(msg)

#     return answers


def main() -> None:
    args = parse_args()
    raw_df_path = Path("data") / args.dataset / f"{args.dataset}-{args.split}.jsonl"
    raw_df = pd.read_json(raw_df_path, lines=True)
    raw_df["question"] = raw_df["question"].apply(clean)
    raw_df["answers"] = raw_df["answers"].apply(lambda x: [clean(a) for a in x])

    # if args.dataset == "triviaqa" and args.split == "train":
    #     raw_df["answers"] = raw_df["answers"].apply(clean_experimental)

    clean_df_path = Path("data") / args.dataset / f"{args.dataset}-{args.split}-clean.jsonl"
    raw_df.to_json(clean_df_path, orient="records", lines=True)

    # for c in sorted(all_chars):
    #     print(f"Character: {repr(c)} | Unicode: U+{ord(c):04X} | Name: {unicodedata.name(c, 'UNKNOWN')}")

if __name__ == "__main__":
    main()
