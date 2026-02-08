import os
import re

TARGET_EXTENSIONS = {".py"}
IGNORE_DIRS = {".git", "venv", "__pycache__", ".idea", ".ipynb_checkpoints"}


def get_indentation(line):
    return len(line) - len(line.lstrip())


def is_block_start(line):
    # Controlla se la riga finisce con : (ignorando commenti)
    content = line.split("#")[0].strip()
    return content.endswith(":")


def should_remove(line):
    stripped = line.strip()
    # Rimuovi print(...) solo se è semplice (non multilinea)
    if stripped.startswith("print(") and stripped.endswith(")"):
        return True
    # Rimuovi commenti interi, ma ignora shebang (#!) o encoding
    if (
        stripped.startswith("#")
        and not stripped.startswith("#!")
        and "coding:" not in stripped
    ):
        return True
    return False


def clean_file_safe(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 1. Identifica le righe da rimuovere
    to_remove = [False] * len(lines)
    for i, line in enumerate(lines):
        if should_remove(line):
            to_remove[i] = True

    # 2. Ricostruisci il file inserendo 'pass' se necessario
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Se la riga NON è da rimuovere, la teniamo
        if not to_remove[i]:
            # Rimuovi emoji
            clean_line = re.sub(r"[^\x00-\x7F]+", "", line)
            new_lines.append(clean_line)

            # Se questa riga apre un blocco (finisce con :), controlliamo se il blocco rimarrà vuoto
            if is_block_start(clean_line):
                current_indent = get_indentation(clean_line)

                # Guarda avanti per vedere se c'è codice valido nel blocco
                j = i + 1
                block_has_content = False
                while j < len(lines):
                    next_line = lines[j]
                    # Salta righe vuote
                    if not next_line.strip():
                        j += 1
                        continue

                    next_indent = get_indentation(next_line)
                    # Se l'indentazione torna indietro, il blocco è finito
                    if next_indent <= current_indent:
                        break

                    # Se troviamo una riga CHE NON verrà rimossa, il blocco è salvo
                    if not to_remove[j]:
                        block_has_content = True
                        break

                    j += 1

                # Se il blocco è vuoto (tutto rimosso), inseriamo 'pass'
                if not block_has_content:
                    pass_indent = current_indent + 4
                    new_lines.append(" " * pass_indent + "pass\n")

        i += 1

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"✨ Pulito (Safe): {filepath}")


def main():
    root_dir = os.getcwd()
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in files:
            if os.path.splitext(file)[1] in TARGET_EXTENSIONS:
                if "clean_code" in file:
                    continue
                clean_file_safe(os.path.join(root, file))


if __name__ == "__main__":
    confirm = input(
        "⚠️  Script SAFE: Rimuove print/commenti e aggiunge 'pass' dove serve. Procedere? (s/n): "
    )
    if confirm.lower() == "s":
        main()
    else:
        print("Annullato.")
