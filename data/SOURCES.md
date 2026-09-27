# Sample Documentation Sources

Public technical documentation used as the seed corpus (`data/sample_docs/`), downloaded 2026-09-27.

| File | Format | Topic / Version | Source | License |
|---|---|---|---|---|
| fastapi_first_steps.md | Markdown | FastAPI tutorial | https://github.com/fastapi/fastapi/blob/master/docs/en/docs/tutorial/first-steps.md | MIT |
| fastapi_dependencies.md | Markdown | FastAPI tutorial | https://github.com/fastapi/fastapi/blob/master/docs/en/docs/tutorial/dependencies/index.md | MIT |
| pydantic_v2_migration_guide.md | Markdown | Pydantic v1 → v2 | https://github.com/pydantic/pydantic/blob/main/docs/migration.md | MIT |
| docker_compose_getting_started.md | Markdown | Docker Compose | https://github.com/docker/docs/blob/main/content/manuals/compose/gettingstarted.md | Apache-2.0 |
| langgraph_persistence.md | Markdown | LangGraph persistence | https://docs.langchain.com/oss/python/langgraph/persistence | See source site |
| redis_data_types.html | HTML | Redis data types | https://redis.io/docs/latest/develop/data-types/ | See source site |
| kubernetes_deployments.html | HTML | Kubernetes Deployments | https://kubernetes.io/docs/concepts/workloads/controllers/deployment/ | CC BY 4.0 |
| postgresql_16_index_types.pdf | PDF | PostgreSQL 16 | https://www.postgresql.org/docs/16/indexes-types.html (printed to PDF) | PostgreSQL License |
| python_3_12_whats_new.pdf | PDF | Python 3.12 | https://docs.python.org/3.12/whatsnew/3.12.html (printed to PDF) | PSF License |

Preparation applied to the downloaded files:
- FastAPI docs pull code from the repo via `{* ../../docs_src/... *}` include lines; those were replaced with the referenced source files as fenced `python` blocks so the docs are self-contained.
- The LangGraph page's "Documentation Index" preamble (site boilerplate) was removed.

Version-specific docs (Pydantic v2, PostgreSQL 16, Python 3.12) are included to exercise version metadata extraction and filtered retrieval.
