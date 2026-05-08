# Spool Interpreter Label Taxonomy

Generic z/OS spool categories:
- dataset_resolution_failure
- allocation_failure
- permission_or_security_failure
- jcl_syntax_failure
- utility_parameter_failure
- execution_abend
- scheduler_or_environment_issue
- other

Smart/RESTART and Smart/RRSAF specific categories. The full DCA / SQLCODE /
abend reason mapping lives in `flow-starter/docs/smart-restart/messages.md`
and is synced into `prompt_spec.json` via
`flow-ml/scripts/sync-smart-restart-knowledge.sh`:
- smart_restart_resource_unavailable - Db2 down, terminating, forcibly
  stopped, or at its connection ceiling. Retryable with back-off; restart
  resumes from the last checkpoint automatically.
- smart_restart_configuration - missing or mis-bound load module / plan /
  profile defaults table. Not retryable without remediation.
- smart_restart_application_logic - application code violated a
  Smart/RRSAF invariant (SQL after init failure, recursive retry on a hard
  error, corrupted save area). Not retryable; fix and recompile.
- smart_restart_input_syntax - RAINPUT / SQLBATCH / SYSTSIN parameter
  mistake (missing parenthesis, unrecognized keyword, non-integer where
  integer required). Not retryable; correct the input record.
