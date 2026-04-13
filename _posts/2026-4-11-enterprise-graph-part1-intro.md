---
layout: post
title: Project based Intro to Context Graph, Semantic layers, Ontology, and where AI fits in
---
<br>

# Welcome to my blog series on the Context Graph and AI

This is the first blog post in a series covering my personal exploration into the intersection of AI and enterprise knowledge management. One thing I have been following closely since the advent of modern AI is the adaptation of older, pre-LLM technologies to accompany or augment the limitations of current LLMs. The area that I'm most interested in is around long-term memory (i.e., any persistence of information beyond the context window), search (RAG, GraphRAG, etc.), and specialized domain expertise (skills, subject matter ontology, etc.). These technologies, like knowledge graphs or vector search, have a deep history preceding the current AI boom.

# Personal Wiki vs. Microsoft Fabric IQ vs. Context Graph

My day job deals a lot with enterprise-scale knowledge management — essentially, how to capture and store the tacit knowledge of business processes, and then use it with AI to automate routine tasks and augment human decision-making. 

Part of the motivation to write this blog came from Andrej Karpathy's discussion of the ["Personal Wiki"](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). His discussion led to an explosion of AI engineers vibe-coding their own solutions, all putting their spin on the idea. With broad brushstrokes, we can describe this idea as an AI workflow pipeline that summarizes, extracts, and curates entities and concepts into Wikipedia-like or blog-like pages with links to other pages for navigation. You could say this blog series is my spin on the topic — specifically, how can we take this out of the scope of a "personal wiki" and scale it into an "enterprise wiki".

Right away, we can see that a personal or consumer-grade wiki has very different concerns compared to an enterprise-grade wiki. 

A **Personal Wiki** operates at a much smaller scale. Pages can just be markdown files, and simple file or keyword search on the file system is sufficient. It probably doesn't need formal provenance tracking. The definitions of terms for the AI don't need to be formal, domain-specific, or standardized. 

These concerns become hard requirements when we move away from personal use to groups where multiple people collaborate on a knowledge base at a larger scale. Search and traversal become required because the wiki pages grow beyond what basic markdown can handle. Formal ontology is required for LLMs to understand exactly what concepts are relevant. Provenance is required to understand where concepts come from and what contributed to them.

On the other end of the spectrum, we have emerging enterprise solutions like **Microsoft Fabric IQ**. Fabric IQ is also a way to do enterprise knowledge management, and it's actually quite similar to some of our ideas. However, it uses a Labeled Property Graph (LPG) instead of RDF. More importantly, it doesn't really handle unstructured data (like PDFs, reports, or text files). It's mostly focused on relational-to-graph mapping with a semantic binding layer over existing structured data. 

Our **Context Graph** (the proof-of-concept I'm building in this series) aims to bridge these gaps. It's an enterprise-scale knowledge graph built on RDF (which gives us rich ontologies, inference rules, and stable URIs). Crucially, it handles *both* unstructured data (via a robust indexing pipeline with OCR and LLM extraction) and structured data (via a semantic binding layer to databases like BigQuery and Postgres). It maintains stable entity identity over time and tracks the exact provenance of every single assertion.

Here is a quick breakdown of how these three approaches compare across key dimensions:

<div style="max-width: 600px; margin: 2rem auto;">
  <canvas id="comparisonChart"></canvas>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
  document.addEventListener("DOMContentLoaded", function() {
    const ctx = document.getElementById('comparisonChart').getContext('2d');
    new Chart(ctx, {
      type: 'radar',
      data: {
        labels: ['Unstructured Data', 'Structured Data', 'Enterprise Scale', 'Formal Ontology', 'Granular Provenance'],
        datasets: [{
          label: 'Personal Wiki',
          data: [5, 1, 1, 1, 1],
          fill: true,
          backgroundColor: 'rgba(255, 152, 0, 0.2)',
          borderColor: 'rgb(255, 152, 0)',
          pointBackgroundColor: 'rgb(255, 152, 0)',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: 'rgb(255, 152, 0)'
        }, {
          label: 'Microsoft Fabric IQ',
          data: [1, 5, 5, 3, 3],
          fill: true,
          backgroundColor: 'rgba(156, 39, 176, 0.2)',
          borderColor: 'rgb(156, 39, 176)',
          pointBackgroundColor: 'rgb(156, 39, 176)',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: 'rgb(156, 39, 176)'
        }, {
          label: 'Context Graph PoC',
          data: [5, 3, 4, 4, 5],
          fill: true,
          backgroundColor: 'rgba(3, 169, 244, 0.2)',
          borderColor: 'rgb(3, 169, 244)',
          pointBackgroundColor: 'rgb(3, 169, 244)',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: 'rgb(3, 169, 244)'
        }]
      },
      options: {
        elements: {
          line: { borderWidth: 3 }
        },
        scales: {
          r: {
            angleLines: { display: true },
            suggestedMin: 0,
            suggestedMax: 5,
            ticks: { stepSize: 1, display: false }
          }
        },
        plugins: {
          legend: { position: 'bottom' }
        }
      }
    });
  });
</script>

| Dimension                | Personal Wiki (Karpathy)   | Microsoft Fabric IQ                          | Context Graph (Our PoC)                              |
| ------------------------ | -------------------------- | -------------------------------------------- | ---------------------------------------------------- |
| **Target Scale**         | Individual / Consumer      | Enterprise                                   | Enterprise (PoC)                                     |
| **Graph Technology**     | Implicit (Markdown links)  | Labeled Property Graph (LPG)                 | RDF (Resource Description Framework)                 |
| **Data Sources**         | Unstructured (Notes, Text) | Primarily Structured (Relational, Lakehouse) | Both (Unstructured via LLM + Structured via Binding) |
| **Ontology & Reasoning** | Informal / None            | Business semantics, LPG traversal            | Formal OWL/RDFS, forward & backward chaining         |
| **Entity Identity**      | File paths / Local IDs     | Internal database IDs                        | Stable, globally unique URIs                         |
| **Provenance**           | Minimal                    | System-level lineage                         | Granular (RDF-star down to the document chunk)       |


# The Knowledge Graph Architecture

The system of the enterprise-scale wiki that I imagine is built like a knowledge graph with three major parts: indexing, querying, and serving. 

1. **Indexing:** Deals with bringing information from various structured or unstructured sources into the wiki.
2. **Querying:** Deals with the different retrieval mechanisms of the information to answer questions.
3. **Serving:** Deals with how to serve the wiki content to both humans and AI.

# Proof of Concept

I should note that even though I think this is what the enterprise wiki of the future will look like, this strictly remains my personal exploration and proof-of-concept. I don't have the scale or the time to perfect it to the production quality required for a massive enterprise. 

However, I think the architectural decisions I have made here are valid, which is what really matters in the era of agentic engineering. Almost every specific component I describe here could probably be thrown away or swapped out, and the conclusions would still be relevant because they focus on the essential complexities of the system. This is my intention with this proof-of-concept: in the era of AI, the speed of discovery and verification becomes the bottleneck, while the speed of coding is no longer the rate-limiting step.

# Organization of the blog series

Here is how the rest of the series is laid out:

- **Part 1:** Intro (You are here)
- **Part 2:** [System architecture & design]({% post_url 2026-4-11-enterprise-graph-part2-design %})
- **Part 3:** [Introduction to ontology, RDF]({% post_url 2026-4-11-enterprise-graph-part3-rdf %})
- **Part 4:** [Data indexing pipeline]({% post_url 2026-4-11-enterprise-graph-part4-indexing %})
- **Part 5:** [Data normalization]({% post_url 2026-4-11-enterprise-graph-part5-normalization %})
- **Part 6:** [Dealing with structured data]({% post_url 2026-4-11-enterprise-graph-part6-structured-data %})
- **Part 7:** [Data visualization]({% post_url 2026-4-11-enterprise-graph-part7-data-visualization %})
- **Part 8:** [Knowledge unification with AI]({% post_url 2026-4-11-enterprise-graph-part8-AI-query %})
- **Part 9:** [Recap and future direction]({% post_url 2026-4-11-enterprise-graph-part9-recap %})

