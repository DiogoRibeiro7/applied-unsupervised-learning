# Security Policy

## Supported versions

This is a research and applied machine learning repository. Security fixes are applied to the
default branch and to the latest tagged release when practical.

## Reporting a vulnerability

Please report vulnerabilities privately instead of opening a public issue.

Use GitHub's private vulnerability reporting when available, or contact the
maintainer listed in `pyproject.toml`.

Include:

- Affected file, notebook, command, or API endpoint.
- Steps to reproduce the issue.
- Expected impact.
- Any suggested mitigation.

## Scope

Relevant reports include unsafe model artifact loading, path traversal, unsafe
deserialization, dependency vulnerabilities, or API behaviours that could expose
data unexpectedly.

This repository does not include private datasets, credentials, or production
infrastructure. Please do not submit real secrets in reports, issues, notebooks,
or test fixtures.
