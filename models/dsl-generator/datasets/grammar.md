# Flow DSL grammar reference

Canonical reference for the DSL emitted by the Flow Studio runtime
(`crates/flow-dsl/src/serializer.rs`) and consumed by `crates/flow-dsl/src/parser.rs`.
The training set in `samples/seed_samples.jsonl` should produce DSL that round-trips
through the parser without modification.

## Header

```
flow "<name>" v<version>
```

- `<name>` is a quoted string (escapes: `\"`, `\\`, `\n`, `\t`, `\r`).
- `<version>` is a semver-shaped string; defaults to `1.0.0` if omitted.

Example:

```
flow "JCL Validate-Submit" v1.0.0
```

## Nodes

```
<id>[<kind>: "<label>"] {
  field1: value1
  field2: value2
}
```

- `<id>` is a slug (`[a-z0-9-]+`); used as the identifier in edges.
- `<kind>` is one of `action`, `ai`, `cloud_ai`, `utility`.
- `<label>` is the human-readable display name (quoted string).
- The body is optional. Values may be quoted strings, numbers, booleans, or `null`.

Field reference per kind:

| kind     | required fields | optional fields                                                            |
|----------|-----------------|----------------------------------------------------------------------------|
| action   | `adapter`       | `actionId`, `connectionId`, `command`, `args`, `cwd`, `timeoutMs`           |
| ai       | `modelId`       | `thresholdHigh`, `thresholdLow`, `input`                                    |
| cloud_ai | `provider`, `modelId` | `prompt`, `maxTokens`, `temperature`, `auditContent`                  |
| utility  | `utilityId`     | -                                                                            |

`adapter` for `action` kind is one of `mock`, `zowe`, `zosmf`, `ssh`, `shell`.
`provider` for `cloud_ai` is one of `claude`, `openai`, `gemini`.

## Edges

```
<source>[.<outcome>] --> <target> [: "<label>"] [when <condition>]
```

- `<outcome>` is `pass` (alias `success`), `fail` (alias `failure`), or omitted (`always`).
- The label and `when` clause are optional and rarely used.

Example:

```
validate-jcl.pass --> submit-jcl
validate-jcl.fail --> notify
submit-jcl --> archive
```

## Canonical document

```
flow "JCL Validate-Submit" v1.0.0

validate-jcl[ai: "Validate JCL"] {
  modelId: "jcl-validator-stub:v0"
  thresholdHigh: 0.9
  thresholdLow: 0.6
}

submit-jcl[action: "Submit JCL"] {
  adapter: "zowe"
  actionId: "submit-jcl"
}

notify[utility: "Notify on-call"] {
  utilityId: "send-email"
}

validate-jcl.pass --> submit-jcl
validate-jcl.fail --> notify
submit-jcl --> notify
```

## Generation rules

The fine-tuned model must:

1. Always begin with a `flow "<name>" v<version>` header.
2. Use exactly one of the four supported kinds per node, with valid required fields.
3. Reference only ids declared in the same document from edges.
4. Emit a single canonical document per response, no commentary outside the JSON envelope.
5. Wrap the result as `{"dsl": "<dsl text>"}` per the response schema.
