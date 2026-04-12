---
layout: post
title: Context Graph, AI, Semantic layers, Ontology
---

<br>

# Why knowledge graph?

- linking many entities, names, concepts together allow human or AI to traverse and discover related entities

- 

# Why RDF and not LPG

this is a difficult choice. I am more familiar with the labeled property graph (LPG) technology. LPG an easier system to understand, querying looks easier to understand, and the concept maps nicely to programming objects. But ultimately I go with RDF because of the richness of the ontology support, the possibility to uncode business rules in ontology, and the idea of linked data. One potential implication of RDF being ontology driven development is that the ontology lives outside of the application code which gives a nice separation of process between the application code vs business logic.

Ultimately I think some specific design decisions here is applicable in LPG styled graph but it will require more application level code and ultimately expose more mainatenance surface. 


## table of comparison between RDF and LPG



# Database

apache Jena fuseki for RDF triple store, ontop for SQL translation, postgres for document + chunk storage. 




# Indexing pipeline

indexing pipeline using temporal for pipeline execution. the indexing pipeline is in typescript. 

# Querying

this will be done by MCP or CLI that allows LLMs to query concepts or traverse entities from the graph. Because I have made MCPs elsewhere and because largely the querying will be dependent on the choices of indexing, I leave the implementation of MCPs as future scope.

# Serving 

we envision entity or documents to be served as a web page similar to a wikipedia page. The relationships between entities are links from pages to pages. this web page should be citable by external systems, such as someone should be able to reference this piece of knowledge in a report (pdf), presentation or send it as a weblink in an email. This means the web page must have stable urls. This requirement right away impose a severe limitation -- that our knowledge graph must either grow continuously or keep tract of entity identity over time as opposed to the "nuke and rebuild" style where update happen as a batch process and ids don't survive between built. This requirement also plays a crucial role in selecting RDF as opposed to LPG. This is because uri is an inherent and foundational concept of RDF while any stable id in LGP is a matter of application level code. This again does not mean url stable identity cannot be done in LPG, it simply means that it is not thought of as a built-in foundation of LPG.

good for spot check, sanity check. protection against hallucination and provide trust and verifiable grounding for LLMs.

This is done using Astro server side rendering with React Island for interactive parts such as the graph visualization. 

# system of knowledge record

Ultimately what I have in mind is a system of knowledge record where any entity or concept is tracked who contribute,  when it is recorded, from which document it come from, and what other concepts are related to it, and how they are related. When we feed this information to LLMs, we want it to be able to navigate this web of knowledge in place of us, and able to make sense of the knowledge and use the knowledge as references when it makes decision. The reference here is crucial for we want to gaurantee 0% hallucination from LLMs, meaning every assertion must be backed up by sources, and we can quantify the agreement of asertion and cited sources. 

# Table

