---
layout: post
title: Context Graph, AI, Semantic layers, Ontology
---

<br>

# Welcome to my blog series on the Context Graph and AI

this is first blog post in the series of my personal exploration into the intersection of AI and enterprise knowledge management. One thing I have been following since the advent of AI is adaptation of old pre-LLM technologies to accompany or augment limitation of the current AL/LLMs. The area that I'm interested in is around long term memory (i.e. any persistence of information beyond the context windows), search (RAG, graphRAG, etc.), specialized domain expertise (skills, subject matter ontology, etc.). These technologies, like knowledge graphs or vector search, has deep history preceeding AI.

# Personal wiki vs Enterprise Knowledge Management

My current work deal a lot with enterprise-scale knowledge management i.e. how to capture and store tacit knowledge of business process and use it with AI to and automate routine tasks and augment human decision making. But part of the motivation to write this blog came from the most famous AI influencer Andrej Kapathy in his discussion of "Personal Wiki" https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f  -- . This discussion leads to explosion of AI engineers vibe coding the solutions. They all put their spin on this idea. But with broad brushstrokes we can describe this idea as a pipeline of AI workflow that summarize, extract, curate entities and concepts into a wikipedia or blog like pages with links to other pages for navigation. I may say that this blog series is my spin on the topic -- how can we take this out of the scope of "personal wiki" into "enterprise wiki".

Right away we can see that personal or consumer grade wiki vs enterprise grade wiki will have very different concerns. Personal wiki will be much smaller scales where pages can be markdowns and simple file or keyword search on the file systems is sufficient. It probably will not need formal provenance tracking. The definitions of terms for AI will not need to be formal or domain specific or standardize. 
These concerns become requirements when we move away from personal to groups where multiple people collaborate knowledge base on larger scales. Search and traversal is required because now the wiki pages could grow beyond basic markdown can handle. Formal ontology is required for LLMs to understand what concepts are relevant. Provenance is required to understand where concepts come from and what contribute to that. 

# Knowledge graph

The system of enterprise scale wiki that I imagine will be like a knowledge graph with 3 major parts: indexing, querying, and serving. The indexing part deal with bringing information from various structured or unstructured sources into the wiki, the querying part deals with different retrieval mechanisms of the information to answer the questions, and lastly the serving part deals with how to serve the wiki content to human or AI. 

# proof of concept 

I should note that even though I think this is what the enterprise wiki will be like; this strictly remain my personal exploration proof-of-concept and I don't have the scale or the time to perfect it to the quality required for large scale enterprise. However I think that the decision that I have made here are valid which is what matters in the era of agentic engineering. Almost every specific components here that I will describe can probably be thrown away or replaced and the conclusion is still relevant focusing on the essential complexities of the system. And this is my intention of this proof-of-concept in the era of AI where the speed of discovery and verification becomes bottlebeck while speed of coding is no longer the rate limit step.

# Organization of the blog series

- intro

- system architecture & design

- introduction to ontology, rdf

- data indexing pipeline

- data normalization

- dealing with structured data

- data visualization

- knowledge unification with AI

- recap and future direction