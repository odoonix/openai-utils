import os
import json
import re
import typer


app = typer.Typer()

def _normalize_text(text):
    return text.replace('\r\n', '\n').replace('\r', '\n')


def _extract_qa_entries(text):
    entries = []
    lines = _normalize_text(text).splitlines()

    question = ""
    answer_lines = []
    is_reading_answer = False

    for line in lines:
        stripped = line.strip()

        # skip empty lines
        if not stripped:
            continue

        # check for separator line
        if re.fullmatch(r"[-=]{5,}", stripped):
            if question and answer_lines:
                entries.append((question.strip(), "\n".join(answer_lines).strip()))
                question = ""
                answer_lines = []
                is_reading_answer = False
            continue

        # if line looks like a question
        if (
            not is_reading_answer and
            ("؟" in stripped or stripped.endswith("?") or stripped.lower().startswith("سوال"))
        ):
            question = stripped
            is_reading_answer = True
            continue

        # otherwise, this is part of the answer
        if is_reading_answer:
            answer_lines.append(stripped)

    # handle last block
    if question and answer_lines:
        entries.append((question.strip(), "\n".join(answer_lines).strip()))

    return entries



def convert_to_jsonl(entries, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for question, answer in entries:
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": "تو یک کد ریویور هستی. سوال برنامه‌نویس رو بررسی کن."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            }, ensure_ascii=False) + "\n")



@app.command()
def process(filepath: str):
    """Read a text file containing Q&A pairs and convert it to JSONL format for fine-tuning."""
    try:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"❌ File not found: {filepath}")

        folder = os.path.dirname(filepath)
        # filename = os.path.basename(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        qa_entries = _extract_qa_entries(text)

        if not qa_entries:
            typer.secho("❌ No questions or answers found.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        output_path = os.path.join(folder, "fine_tune_data.jsonl")
        convert_to_jsonl(qa_entries, output_path)

        typer.secho(f"✅ JSONL file successfully saved to:\n{output_path}", fg=typer.colors.GREEN)
        typer.secho(f"🔢 Total question-answer pairs: {len(qa_entries)}", fg=typer.colors.BLUE)

    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
    except UnicodeDecodeError:
        typer.secho("❌ Encoding error: Make sure the file is saved with UTF-8 encoding.", fg=typer.colors.RED)
    except Exception as e:
        typer.secho(f"❌ Unexpected error: {e}", fg=typer.colors.RED)




