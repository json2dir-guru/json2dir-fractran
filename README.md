# json2dir-fractran

json2dir in FRACTRAN: `json2dir.fractran` (2647 fractions, compiled by `tools/fracc.py` from the register-machine source `json2dir.rm`) checks UTF-8, parses and validates the JSON document and builds the shell script; the POSIX sh launcher `json2dir` is pure transport — it encodes the input bytes into the starting number `2^E` (E = the bytes in base 256 under a leading 1), runs an unmodified FRACTRAN interpreter ([pimlu/fractran](https://github.com/pimlu/fractran)'s cycle-detecting Lean version), reads the output bytes back from the exponent of 3 (base-1000 digits), and runs the script only if it ends with `# json2dir: end`.

    python3 tools/fracc.py json2dir.rm > json2dir.fractran
    JSON2DIR_FRACTRAN=/path/to/fractran ./json2dir < tree.json
