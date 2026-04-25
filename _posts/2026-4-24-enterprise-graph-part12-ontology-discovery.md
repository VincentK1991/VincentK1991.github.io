---
layout: post
title: "Enterprise Graph Part 12: Ontology Discovery"
date: 2026-04-24
categories: [Enterprise Graph, Ontology, AI, Agents]
---
install the skills here `gh skill install VincentK1991/present-at-hand ontology-discovery`
<br>

# Moving Beyond Flat Taxonomies

Welcome back to the Enterprise Graph series! Today we're diving into **ontology discovery**. If you've been following along, you know that building a knowledge graph isn't just about dumping data into a database. It's about giving that data meaning. 

The goal of ontology discovery is to find the sweet spot between two extremes that we absolutely want to avoid. On one hand, we don't want a flat, rigid taxonomy that doesn't fit the actual data or the specific questions and insights the user is after. On the other hand, we don't want a totally unstructured, "no-ontology" approach—because that won't generate any insights either; it's just as messy as the raw text itself! 

What we actually want is a rich, non-flat ontology that is laser-focused on the user's needs. We want an ontology equipped with inference rules that work hand-in-hand with the underlying unstructured text. When your ontology is expressive and your rules are dialed in, you don't just get better search—you generate actual, actionable insights. In this context, an "insight" is a new fact or relationship that wasn't explicitly written in the original text, but can be logically deduced by the machine. For example, if the text says "Alice manages the Engineering team" and "Bob is on the Engineering team," an inference rule can generate the insight that "Alice is Bob's boss." This is why we want these rules: they do the heavy logical lifting upfront, so your AI doesn't have to hallucinate connections at query time.

But how do you actually *build* this ontology? You don't want to sit in a room for six months debating definitions, but you also can't just let an LLM hallucinate a schema and hope for the best. 

This is where an **agent-first, human-in-the-loop** workflow comes in.

## Agent-First, Human-in-the-Loop

Think of the AI agent not as a script runner, but as a junior ontology researcher. The agent reads your unstructured content, identifies recurring entities, spots synonym clusters, and drafts a candidate vocabulary. 

But the agent doesn't make the final call. That's where you—the human domain expert—come in. The agent conducts a "domain interview" with you, asking adaptive questions to lock down the scope. What are the core boundaries? What competency questions does this graph *need* to answer? What are the critical relations we must be able to infer?

You guide the process, answer the hard questions, and approve the iterations. The agent does the heavy lifting of drafting the `.ttl` ontology files and the SPARQL `CONSTRUCT` rules. It justifies its modeling decisions with evidence from the corpus, makes trade-offs explicit, and presents alternatives before finalizing anything.

## Why Not Do It The Traditional Way?

Traditional approach to doing ontology is incredibly time-consuming, laborious, and mistake-prone. It's essentially a waterfall model for knowledge management. In the era of agentic engineering, this just isn't scalable. The speed of coding is no longer the rate-limiting step; the bottleneck is the speed of discovery and verification. By putting an AI agent in the driver's seat to do the heavy lifting of drafting, extracting, and measuring, you flip the model. The human expert becomes an editor and an approver rather than a blank-page author. This makes building rich, enterprise-grade ontologies fast, agile, and actually scalable.

## A CLI-Centric Architecture

To make this workflow seamless, we're leaning heavily into a CLI-centric architecture. You can actually try this out yourself by installing the ontology-discovery skill. Just run:

```bash
gh skill install VincentK1991/present-at-hand ontology-discovery
```

Here's how the toolchain works:
1. **You use the CLI to install skills** (like the one above).
2. **The skills use the CLI to install necessary tools** (like Apache Jena for RDF processing and Node.js for extraction/inference scripts).
3. **The skills then use those tools via the CLI** to actually perform the tasks—running extractions, validating triples, and calculating quality metrics.


There is a beautiful, virtuous cycle here. Because we use the CLI to install skills, and skills use the CLI to perform tasks, a skill could potentially use the CLI to install *other* skills or tools as needed. 

Why does this matter? Because the CLI is the universal, standard interface for any coding agent operating in a terminal. By building our tools around CLI commands rather than proprietary APIs or closed-ecosystem plugins, we ensure that our workflow is completely portable across any agentic environment. 

In fact, you could argue that the CLI is the preferred interface for collaboration between humans and AI agents. Think about the alternatives: building a custom backend API or a bespoke web framework. Those are heavy, hard to port over, and notoriously difficult for a coding agent to interact with fluidly. A web UI requires the agent to navigate the DOM; a custom API requires the agent to learn the specific endpoints and authentication quirks. 

But a CLI? A CLI is native to the agent's environment. It's text-in, text-out. Not only is it scriptable, it's inherently **composable** with other terminal tools. You can pipe the output of an ontology extraction directly into a `grep` command, or wrap the whole pipeline in a standard bash script. This composability is already the bedrock of Linux terminals, which happen to be the universal interface for coding agents. By leaning into the CLI, we aren't inventing a new paradigm for agents to learn; we're using the one they already speak fluently.

Just like RDF provides an open, expressive standard for the ontology itself, the CLI provides an open, expressive standard for the agents that build it. It's a portable, standalone setup that keeps everything modular and scriptable.

## How This Fits Into the Enterprise Graph Pipeline

If you've been following the series, you might be wondering exactly where this discovery loop fits in. 

In **[Part 3: Introduction to RDF and Ontology]({% post_url 2026-4-11-enterprise-graph-part3-rdf %})**, we talked about the theory of formal ontologies—Classes, Properties, `rdfs:domain`, and `rdfs:range`. This agentic discovery loop is the *practical application* of that theory. The agent is literally writing the `.ttl` files that define those RDF structures based on the unstructured text it reads.

*Why `.ttl` (Turtle) and not a bespoke YAML or JSON format?* Because RDF is an open, expressive, and mathematically rigorous standard. If you invent your own YAML format for an ontology, you also have to invent your own parser, your own validation logic, and your own inference engine—you are entirely on your own. By using `.ttl` and SPARQL, we inherit decades of battle-tested semantic web tooling (like Apache Jena) that can validate constraints, run complex queries, and perform logical inference out of the box. Furthermore, RDF is highly portable and speaks the same language used throughout the world of formal ontology, from big enterprises to government databases. And crucially, because it's a massive public standard, AI agents already know it fluently.

Furthermore, in **[Part 4: Data Indexing]({% post_url 2026-4-11-enterprise-graph-part4-indexing %})** and **[Part 5: Data Normalization]({% post_url 2026-4-11-enterprise-graph-part5-normalization %})**, we discussed the mechanics of extracting entities from PDFs and text using LLMs. But here's the catch: you can't normalize data into a graph if you don't know what shape the graph should take! Ontology discovery is the necessary prerequisite. The agentic loop we just described creates the "target schema" that your indexing and normalization pipelines use to structure the incoming data.

## The Iterative Discovery Loop

Ontology discovery is absolutely not a "one-and-done" script. It's an iterative, metric-driven loop. The agent runs through a canonical protocol:

1. **Read & Segment**: The agent parses the unstructured corpus to build a candidate concept inventory.
2. **Domain Interview**: You (the human) lock in the scope, confirm boundaries, and set competency questions.
3. **Draft Ontology & Rules**: The agent authors the `.ttl` ontology and the SPARQL `CONSTRUCT` rules (one rule per file). Because this is CLI-native, the agent can instantly validate its own work using standard tools like Apache Jena's `riot`:
   ```bash
   riot --validate ontology.ttl
   ```
4. **Extraction**: The agent runs ontology-guided extraction across multiple sources. This is crucial—single-source extraction hides gaps. We need to see if the vocabulary actually covers the corpus. The agent uses a Node script to do this:
   ```bash
   node ./scripts/extract-to-ttl.mjs \
     --text source.txt \
     --ontology ontology.ttl \
     --output asserted.ttl \
     --mode create
   ```
   This script reads the raw text, applies the drafted ontology, and outputs structured RDF triples (`asserted.ttl`).
5. **Inference**: The agent runs forward-chaining inference using the drafted rules to generate new triples. It takes the raw facts we just extracted and applies our logical rules to deduce new facts:
   ```bash
   node ./scripts/infer-to-ttl.mjs \
     --ontology ontology.ttl \
     --triples asserted.ttl \
     --rules ./rules \
     --output inferred.ttl
   ```
6. **Metrics**: This is where the magic happens. The agent measures quality using repeatable metrics by running a reporting script:
   ```bash
   node ./scripts/metrics.mjs \
     --ontology ontology.ttl \
     --asserted asserted.ttl \
     --inferred inferred.ttl \
     --rules ./rules \
     --format md
   ```
   This generates a markdown report answering critical questions: Are there too many "dead rules" that never fire? Is the hierarchy too flat? Are we getting high inference gain but low precision?

### Deep Dive: What Do These Quality Metrics Look Like?

When the agent runs the metrics script, it's looking for specific red flags that indicate a poor ontology design. Here are a few examples of what it flags and how you (and the agent) fix it:

*   **The "Flat Hierarchy" Flag**: If the metrics show that 95% of your classes are direct subclasses of `owl:Thing` with no deeper nesting, your ontology is too flat. It's essentially just a list of tags. 
    *   *The Fix*: The agent will propose adding meaningful subclass structures (e.g., instead of just `Employee`, breaking it down into `Manager` and `Engineer`) or splitting overloaded classes.
*   **The "Dead Rule" Flag**: If you wrote a brilliant SPARQL rule to infer a `manages` relationship, but the metrics show it fired exactly 0 times across 10 documents, you have a dead rule. 
    *   *The Fix*: The agent will investigate. Is the `WHERE` clause too strict? Is the extraction prompt failing to pull the prerequisite facts? The agent will propose either relaxing the rule or tightening the extraction prompt.
*   **The "Low Typed-Subject" Flag**: If the extraction is pulling lots of entities but failing to assign them an RDF type (e.g., it knows "Alice" exists, but doesn't label her as a `Person`), your downstream inference rules will fail.
    *   *The Fix*: The agent will refine the class definitions in the ontology to make them clearer for the LLM during the extraction phase.

7. **Improve & Repeat**: Based on the metrics and your feedback, the agent revises the ontology and rules, then runs the loop again.

The loop only ends when you give explicit approval. It's a balance of ontology structure quality and inference quality, driven by real data and measurable diagnostics.

If you're building an enterprise graph, you need this kind of rigor. You need to justify your modeling decisions, make your trade-offs explicit, and run controlled iterations to see how your metrics change.

Give the [ontology-discovery skill](https://github.com/VincentK1991/present-at-hand) a spin, and let me know how it changes your workflow. See you in the next post!

