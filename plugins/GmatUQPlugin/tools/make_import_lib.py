"""Build an MSVC import library (.lib) from a DLL's export table.

GMAT's Windows release ships its DLLs without import libraries, which a
plugin needs at link time.  This script lists the exports with dumpbin,
writes a module-definition file and turns it into an import library with
lib.exe.  Both tools come with Visual Studio; they are looked up next to the
linker CMake found.

Usage: make_import_lib.py <path to .dll> <output folder> [<path to link.exe>]
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

_EXPORT = re.compile(r"^\s*\d+\s+[0-9A-Fa-f]+\s+(?:[0-9A-Fa-f]{8}\s+)?(\S+)")


def find_tool(name: str, linker: str | None) -> str:
    if linker:
        candidate = Path(linker).with_name(name + ".exe")
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name}.exe not found; run from a Visual Studio developer environment")
    return found


def exported_names(dumpbin: str, dll: Path) -> list[str]:
    output = subprocess.run([dumpbin, "/nologo", "/exports", str(dll)], capture_output=True, text=True, check=True).stdout
    names = []
    in_table = False
    for line in output.splitlines():
        if re.match(r"^\s*ordinal\s+hint\s+RVA\s+name", line):
            in_table = True
            continue
        if in_table:
            if line.strip().startswith("Summary"):
                break
            match = _EXPORT.match(line)
            if match:
                names.append(match.group(1))
    if not names:
        raise SystemExit(f"no exports found in {dll}")
    return names


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    dll = Path(argv[1])
    out = Path(argv[2])
    linker = argv[3] if len(argv) > 3 else None
    out.mkdir(parents=True, exist_ok=True)
    names = exported_names(find_tool("dumpbin", linker), dll)
    definition = out / (dll.stem + ".def")
    definition.write_text("LIBRARY " + dll.name + "\nEXPORTS\n" + "".join(f"    {name}\n" for name in names), encoding="ascii")
    subprocess.run(
        [find_tool("lib", linker), "/nologo", f"/def:{definition}", f"/out:{out / (dll.stem + '.lib')}", "/machine:x64"],
        check=True,
    )
    print(f"{dll.name}: {len(names)} exports -> {out / (dll.stem + '.lib')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
