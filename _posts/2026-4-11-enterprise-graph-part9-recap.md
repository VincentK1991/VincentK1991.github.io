---
layout: post
title: Context Graph Recap and Future Direction
---
<br>


# What I actually built

Looking back at this series, I want to take stock of what the proof of concept actually is, separate from the aspirational vision I described in the intro.

The working system is a metadata-driven knowledge graph with the following components:

A **Java backend** running Apache Jena Fuseki as the triplestore, with a Spring Boot service sitting in front of it. The backend handles SPARQL query routing (reasoned vs. raw vs. text search), loads OWL/RDFS ontologies and Jena rule language rules from the TBox named graphs, and manages IRI minting for extracted entities. The reasoning engine is Jena's GenericRuleReasoner in hybrid mode — forward chaining for stable rules materialized at load time, backward chaining for domain-specific rules evaluated at query time.

A **TypeScript indexing pipeline** built on Temporal for workflow orchestration. It takes PDF and plain-text documents, OCRs them with liteparse, chunks and embeds them with OpenAI's text-embedding-3-small, extracts entities and relationships from each chunk with GPT-4o using ontology-guided structured output, asserts the results to the graph via the Java backend, runs two-tier entity normalization (Jaro-Winkler rule-based + GPT-4o LLM-as-judge), and handles rollback via Temporal's saga pattern if anything fails.

An **Astro + React web frontend** providing Wikipedia-style entity pages, a force-directed entity graph visualization, a normalization explorer, a SPARQL query explorer, a BigQuery SQL explorer, and a reasoning playground for testing inference rules.

Two domain **ontologies** with forward and backward chaining rules: economic census (covering census counties, statistical observations, surveys, geographic hierarchies) and public health (covering health outcomes, interventions, populations). Both have BigQuery semantic binding layers describing tables from ACS, Opportunity Atlas, and BLS QCEW.

A separate **insurance benchmark dataset** using Ontop for SPARQL-to-SQL query rewriting — this was a side exploration into virtual knowledge graphs that ended up as a benchmark rather than a production feature.

# Summary of Core Design Decisions

Here is a summary of the core design decisions made throughout this project and my assessment of them:


| Decision                     | Choice made                                               | Alternatives                                              | Assessment                                                                                                           |
| ---------------------------- | --------------------------------------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Graph Framework**          | RDF over Labeled Property Graph (LPG)                     | Labeled Property Graph (LPG) / Neo4j                      | RDF's upfront cost pays off with native reasoning, non-destructive normalization, and schema evolution capabilities. |
| **Entity Identity**          | Globally unique URIs (instead of local IDs)               | Local database IDs / UUIDs                                | Essential for stable identity over time, allowing entities to be citable by external systems and dereferenced for both human and AI consumption. |
| **Ontology Location**        | Ontology lives in the graph (TBox and ABox together)      | Ontology lives in application code / ORM models           | Enables a self-documenting database. LLMs and MCP tools can inspect the graph's ontology dynamically without needing access to application source code. |
| **Architecture**             | Metadata-driven over code-driven                          | Code-driven / hardcoded application logic                 | Metadata-driven configuration allows adding new datasets and ontology update without code changes.                   |
| **Graph Lifecycle**          | Continuous                        | Batch "nuke and rebuild"                                  | Accumulating data rather than wiping it ensures stable entity identity and full auditability.                        |
| **Data Integration**         | Semantic Binding Layer (R2RML / ONTOP)                    | Full ETL copy into triplestore                            | Prevents data staleness and avoids "triple explosion" by querying source databases directly.                         |
| **Normalization Strategy**   | Non-destructive merge via `owl:sameAs`                    | Destructive hard merges                                   | Avoids the trap of hard merges. Solves provenance, trackability, and stable identity.                                |
| **Named Graph Architecture** | Separate graphs for Asserted, Normalization, and Inferred | Single monolithic graph                                   | Isolates ground truth, makes identity decisions reversible, and allows easy regeneration of inferences.              |
| **Serving Layer**            | URI-backed Web Pages                                      | Dynamic search results / no stable URLs                   | Creates a robust serving layer for humans, enabling unbreakable external citations.                                  |
| **Zero Hallucination**       | Strict Provenance Tracking                                | Relying on LLM internal knowledge / vector proximity only | Turns hallucinations into auditable, correctable errors by tracking exact source provenance.                         |


# Future directions

If I were to continue this as more than a proof of concept, the things I'd prioritize are:

**Full MCP integration.** The semantic layer is built. The missing piece is wiring it to Claude (or another LLM) via MCP tools so you can ask natural language questions and get grounded, source-cited answers. This is the headline feature and the thing that makes everything else worth having.

**SHACL validation at ingestion.** Right now the system operates under open world assumption — anything that gets past the Zod schema validation in the TypeScript pipeline can go into the graph. Adding SHACL validation gates at the Java ingest endpoint would catch structural inconsistencies before they propagate.

**Global deduplication sweep.** The per-document normalization step handles new entities well, but it doesn't catch duplicates that appeared in different documents at different times without overlapping indexing runs. A periodic batch job that runs full pairwise dedup over all entities would clean these up.

**Ontology alignment with external vocabularies.** The current ontologies are bespoke. Aligning them with Schema.org, DBpedia, or FHIR (for health) would enable federation — being able to run SPARQL SERVICE queries against external SPARQL endpoints using shared ontology terms. This is where the RDF vision of "web of data" actually comes true.

**Production hardening.** Prometheus + Grafana for observability, Kubernetes for container orchestration, proper authentication on the SPARQL endpoint, rate limiting on the MCP tools. None of this was in scope for a proof of concept.

# Final thoughts

I started this series motivated by the question of whether the old pre-LLM technology stack — knowledge graphs, formal ontologies, semantic web standards — has a role to play in the era of LLMs. My conclusion after building this proof of concept is: yes, but not in the way people usually describe it.

The usual framing is knowledge graphs as alternatives to LLMs for structured reasoning. I think that framing is mostly wrong. LLMs are much better than knowledge graphs at understanding natural language, generating text, and reasoning over fuzzy context. Knowledge graphs are much better than LLMs at provenance, precise entity identity, schema enforcement, and long-term fact accumulation.

The more interesting question is how they work together. The knowledge graph is the grounding mechanism — it's where you store things that need to be remembered precisely, sourced, and auditable. The LLM is the interface — it understands what you're asking, figures out how to query the graph, synthesizes results into natural language, and decides when to trust the graph and when to be uncertain.

In the agentic era, the knowledge graph is essentially a long-term, structured external memory that the AI can query with precision. The AI doesn't need to know RDF. It doesn't need to write SPARQL. It just needs a set of tools that abstract over the graph in a way that's semantically rich and trustworthy. That's what this system is trying to be.

Whether any of this matters in practice depends entirely on whether you're building for a use case where provenance, auditability, and precise entity identity matter. For a personal wiki, probably not — a markdown folder plus a vector search is enough. For an enterprise system dealing with regulatory compliance, multi-source data integration, or any domain where "where did this fact come from" is a real question — I think it matters a lot.

That's the argument this blog series has been trying to make. I hope it was useful.x
---
**Navigation:**
- Previous: [Part 8: Knowledge Unification with AI]({% post_url 2026-4-11-enterprise-graph-part8-AI-query %})
