import os
import json
import re
import typer
from typing import List, Tuple

app = typer.Typer()


def normalize_text(text: str) -> str:
    return text.replace('\r\n', '\n').replace('\r', '\n')


def extract_qa_entries(text: str) -> Tuple[List[Tuple[str, str]], int]:
    """
    Extracts valid (question, answer) pairs from free-form text.
    Handles irregular separators and skips incomplete blocks.
    """
    entries = []
    invalid_count = 0
    lines = normalize_text(text).splitlines()

    question = ""
    answer_lines = []
    is_reading_answer = False

    def is_separator(line: str) -> bool:
        return re.fullmatch(r"[-=]{3,}", line.strip()) is not None

    def is_question(line: str) -> bool:
        return (
            "؟" in line or
            line.strip().endswith("?") or
            line.strip().lower().startswith("سوال")
        )

    def flush_entry():
        nonlocal question, answer_lines, invalid_count
        if question and answer_lines:
            # Remove separators from answers
            clean_answer = "\n".join(
                l for l in answer_lines if not is_separator(l)
            ).strip()
            if clean_answer:
                entries.append((question.strip(), clean_answer))
            else:
                invalid_count += 1
        elif question and not answer_lines:
            invalid_count += 1
        elif not question and answer_lines:
            invalid_count += 1
        question = ""
        answer_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if is_question(stripped):
            flush_entry()
            question = stripped
            is_reading_answer = True
            continue

        if is_separator(stripped):
            continue  # ignore standalone separators in answers

        if is_reading_answer:
            answer_lines.append(stripped)

    flush_entry()  # handle last block

    return entries, invalid_count



def convert_to_jsonl(entries: List[Tuple[str, str]], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for question, answer in entries:
            json_obj = {
                "messages": [
                    {"role": "system", "content": "تو یک کد ریویور هستی. سوال برنامه‌نویس رو بررسی کن."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")


@app.command()
def process(folder: str):
    """
    Read all .txt files in the folder and extract Q&A pairs to fine_tune_data.jsonl
    """
    if not os.path.isdir(folder):
        typer.secho(f"❌ Folder not found: {folder}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    txt_files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    if not txt_files:
        typer.secho("❌ No .txt files found in the folder.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    all_entries = []

    for filename in txt_files:
        filepath = os.path.join(folder, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            qa_entries, parse_invalid_count = extract_qa_entries(text)
            valid_entries = []

            for idx, (question, answer) in enumerate(qa_entries, 1):
                if not question.strip() or not answer.strip():
                    typer.secho(
                        f"⚠️ Skipped invalid Q&A in file '{filename}', entry #{idx} (missing question or answer)",
                        fg=typer.colors.YELLOW
                    )
                    continue
                valid_entries.append((question, answer))

            if valid_entries:
                typer.secho(f"✅ {len(valid_entries)} valid Q&A pairs in '{filename}'", fg=typer.colors.GREEN)
                all_entries.extend(valid_entries)
            else:
                typer.secho(f"⚠️ No valid Q&A pairs found in '{filename}'", fg=typer.colors.YELLOW)

            if parse_invalid_count:
                typer.secho(
                    f"⚠️ Skipped {parse_invalid_count} incomplete Q&A blocks in '{filename}' (missing question or answer)",
                    fg=typer.colors.MAGENTA
                )

        except UnicodeDecodeError:
            typer.secho(f"❌ Encoding error in file: {filename}", fg=typer.colors.RED)
        except Exception as e:
            typer.secho(f"❌ Error processing file '{filename}': {e}", fg=typer.colors.RED)

    if not all_entries:
        typer.secho("❌ No Q&A entries found in any files.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    output_path = os.path.join(folder, "fine_tune_data.jsonl")
    convert_to_jsonl(all_entries, output_path)

    typer.secho(f"🎉 All Q&A entries written to: {output_path}", fg=typer.colors.BLUE)
    typer.secho(f"🔢 Total Q&A pairs: {len(all_entries)}", fg=typer.colors.CYAN)



