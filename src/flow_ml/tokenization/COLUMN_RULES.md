# JCL column rules — normative spec for the custom pre-tokenizer

This document is the spec **both** `flow_ml.tokenization.jcl_tokenizer`
(Python, training-time) and flow-studio's
`crates/flow-model-runtime/src/backends/bert_classifier.rs` (Rust,
inference-time) must implement identically. A fixture-based round-trip
test in the Rust backend asserts byte-identical output on ~20 hand-authored
JCL strings.

## Goal

JCL is column-sensitive. A standard BPE tokenizer chops on whitespace +
frequency, oblivious to JCL's column-sensitive grammar:

- **Columns 1-2**: must be `//` for any JCL statement (or `/*` for a
  delimiter), else it's invalid.
- **Columns 3-10**: name field (statement label).
- **Columns 12+**: operation + operands.
- **Column 72**: continuation marker — if non-blank, this statement
  continues on the next line.
- **Columns 73-80**: sequence numbers, traditionally ignored by the JCL
  parser.

The pre-tokenizer normalises these rules into special-token markers the
BPE then learns alongside the JCL content.

## The seven rules

Applied in order, per input line:

### Rule 1: tab expansion

Tabs in input are expanded to spaces (one tab = 4 spaces, left-aligned).
JCL conventionally has no tabs, but pasted samples sometimes do.

### Rule 2: column 73-80 strip

Anything in columns 73+ is dropped before tokenisation. (Column index is
1-based per the JCL convention, 0-based in code.)

```
//STEP1   EXEC PGM=IEFBR14                                              0001234
                                                                        ^ col 73
```

becomes

```
//STEP1   EXEC PGM=IEFBR14
```

Lines shorter than 72 columns pass through unchanged.

### Rule 3: continuation marker

If column 72 is non-blank, emit a `<CONT>` token at end-of-line **after**
the column-72 character. The character itself is preserved (it carries
information — typically `X` or `&`).

```
//STEP1   EXEC PGM=IEBGENER,PARM=('FIRSTPART',                          X
```

becomes (logically — actual tokens emitted by the BPE differ)

```
//STEP1   EXEC PGM=IEBGENER,PARM=('FIRSTPART',                          X <CONT>
```

### Rule 4: line start marker

Emit a `<COL1>` token at the start of every non-empty line, **before**
any other tokens on that line. Lets the model learn statement boundaries
even when BPE breaks `//STEP1` into multiple subword tokens.

### Rule 5: blank line preservation

Truly empty lines (all whitespace) are dropped. JCL ignores them.

### Rule 6: line terminator

Replace `\r\n` with `\n` (Windows line endings). Trailing `\n` on the
last line is optional; the pre-tokenizer doesn't enforce.

### Rule 7: encoding

Input is decoded as UTF-8 by both implementations. Non-UTF-8 bytes raise
an error in Python (`UnicodeDecodeError`) and `InferenceError::Backend`
in Rust. JCL is conventionally ASCII; non-ASCII content is treated as
suspicious and surfaced via the error path.

## Special tokens

| Token | Purpose | BPE vocab index |
|---|---|---|
| `<COL1>` | Line start marker (Rule 4) | reserved at BPE training time |
| `<CONT>` | Continuation marker after column-72 (Rule 3) | reserved at BPE training time |
| `<PAD>` | Padding token | reserved |
| `<UNK>` | Out-of-vocab fallback | reserved |
| `<CLS>` | BERT CLS token | reserved |
| `<SEP>` | BERT SEP token | reserved |

Special tokens are added to the BPE vocab via the
`tokenizers::AddedToken` API in Python; the Rust side enumerates them by
the same names.

## Pseudocode

```python
def pre_tokenize(text: str) -> str:
    lines = text.replace("\r\n", "\n").split("\n")
    out_lines = []
    for line in lines:
        line = line.expandtabs(4)
        if not line.strip():
            continue  # Rule 5
        # Rule 3: continuation marker.
        cont = (len(line) > 71 and line[71] != " ")
        # Rule 2: strip cols 73+.
        line = line[:72]
        # Rule 4: line start marker.
        out_lines.append(f"<COL1> {line}" + (" <CONT>" if cont else ""))
    return "\n".join(out_lines)
```

The Rust impl mirrors this byte-for-byte.

## Fixture set

`/Users/nedal/TECH/FLOW/flow-ml/src/flow_ml/tokenization/fixtures/jcl_pretokenize_fixtures.json`
holds ~20 input/expected pairs. Both Python tests and the Rust backend's
unit tests load this file and assert identical output. Adding a new edge
case means appending one fixture; the implementations stay locked.
