---
layout: post
title: "Enterprise Graph Part 10: Appendix - Glossary"
date: 2026-04-14
categories: [Enterprise Graph, RDF, LPG]
---

# The "Different Word, Same Concept" Dictionary

When mapping concepts between the two architectures, it is helpful to know the direct (or near-direct) translations.

*   **Resource (RDF) / Node (LPG)**: The entities in your graph. In RDF, a resource is anything identified by a URI (a person, a place, a concept). In LPG, a node is a distinct object in the database.
*   **Predicate (RDF) / Relationship (LPG)**: The directed connection between two entities. In RDF, this is always a URI. In LPG, it has a specific "type" (e.g., ACTED_IN) and a direction.
*   **Literal (RDF) / Property Value (LPG)**: The actual data values (strings, integers, dates). In RDF, literals can only be the "Object" at the end of a triple. In LPG, these are the values inside the key-value properties.
*   **Class (RDF) / Label (LPG)**: How you group entities. In RDF, you say a resource is an rdf:type of a specific Class (e.g., Person). In Neo4j, you apply a Label (e.g., :Person) to a node.
*   **SPARQL (RDF) / Cypher (LPG)**: The query languages. SPARQL is a W3C standard based on matching graph patterns of triples. Cypher is an open-source declarative language based on ASCII-art pattern matching `(node)-[relationship]->(node)`.

# Important RDF & Semantic Web Terminology

These terms are foundational to the Semantic Web stack and do not have direct 1:1 equivalents in standard Neo4j databases.

*   **Triple**: The atomic data entity in RDF. It consists of three parts: Subject, Predicate, and Object (e.g., Alice -> knows -> Bob). Every piece of data in an RDF graph is expressed as a triple.
*   **IRI / URI (Internationalized Resource Identifier)**: The global unique identifier used in RDF. Every subject, predicate, and class must be an IRI, ensuring that when two datasets merge, identical IRIs refer to the exact same concept.
*   **Ontology**: A formal naming and definition of the types, properties, and interrelationships of the entities that really or fundamentally exist for a particular domain.
*   **OWL (Web Ontology Language)**: A rich vocabulary used to define complex ontologies. It allows you to express logic, such as "A mother is a subclass of person and must be female," enabling machines to infer new knowledge.
*   **RDFS (RDF Schema)**: A lighter-weight vocabulary for modeling, providing basic tools like `rdfs:Class`, `rdfs:subClassOf`, and `rdfs:subPropertyOf`.
*   **Blank Node**: A node in an RDF graph that does not have an IRI. It acts as an existential variable (e.g., "someone holds this job, but I don't know who").
*   **Inferencing / Reasoning**: The ability of a semantic graph engine to automatically generate new triples based on the rules defined in your ontology (e.g., if A is a brotherOf B, the engine infers A is Male).

# Important LPG / Neo4j Terminology

These terms define how data is modeled and traversed in a Labeled Property Graph.

*   **Relationship Type**: Every relationship in an LPG must have exactly one type (e.g., PURCHASED). It defines the nature of the connection.
*   **Edge Properties**: A defining feature of LPGs. You can attach data directly to the connection itself. For example, on a PURCHASED relationship, you can add properties for date: "2023-10-25" and amount: 50.00. (Note: Doing this in standard RDF requires a complex workaround called "Reification").
*   **Traversal**: The process of navigating through a graph by moving from nodes to relationships to other nodes. LPGs like Neo4j are highly optimized for deep, complex traversals (Index-Free Adjacency).
*   **Index-Free Adjacency**: The underlying architectural design in native graph databases (like Neo4j) where every node maintains direct memory pointers to its adjacent relationships. This allows queries to perform at lightning speed regardless of the total size of the database.

# Schema, Ontology, and Semantic Binding

Here is how you can define and differentiate them across the RDF and LPG landscapes.

### 1. Schema: The "Shape" of the Data
A Schema dictates the structural rules and constraints of your data. It answers the question: "What is this data allowed to look like?"
*   **In LPG (Neo4j)**: A schema is primarily about data integrity and structural constraints. It enforces rules at the database level, such as ensuring a User node must have an email property (existence constraint), or that the email must be unique (uniqueness constraint), or restricting a property to a specific data type (string, integer). It is very similar to a traditional relational database schema, though Neo4j allows you to be schema-less if you prefer.
*   **In RDF (Semantic Web)**: A schema (specifically RDFS - RDF Schema) acts as a basic vocabulary. It defines the basic hierarchy of your data. It declares that a Dog is a subclass of Animal, and that the property hasOwner should only connect an Animal (its domain) to a Person (its range). It is less about strict database constraints and more about basic classification.

### 2. Ontology: The "Meaning" and "Logic" of the Data
An Ontology goes far beyond the structural shape of data; it defines the rich, logical relationships and real-world meaning of the domain. It answers the question: "What does this data actually mean, and what else can we logically infer from it?"
*   **In RDF (Semantic Web)**: Ontologies are the superpower of this stack, usually written in OWL (Web Ontology Language). They allow you to define complex logical axioms. For example, an ontology can state that hasParent is the inverse of hasChild, that a person can have exactly two biological parents (cardinality), or that Plant and Animal are disjoint classes (nothing can be both). Because of this logic, an RDF engine can use a Reasoner to infer new data that was never explicitly written into the database.
*   **In LPG (Neo4j)**: Native LPGs do not have built-in ontologies or automated reasoning engines. The "meaning" of the data is usually locked inside the application code or the heads of the developers. To get ontology-like features in Neo4j, developers must either write complex Cypher queries to manually traverse and infer relationships, or use plugins (like Neosemantics/n10s) to import RDF ontologies into the property graph environment.

### 3. Semantic Binding: The "Universal Agreement" of the Data
Semantic Binding is the process of mapping a local piece of data to a universally recognized definition so that different systems can understand each other without prior coordination. It answers the question: "How do I prove my data means the same thing as your data?"
*   **In RDF (Semantic Web)**: Semantic binding is baked into the very foundation of the technology via IRIs (Internationalized Resource Identifiers). When you create a node for a person, you don't just give it a label called `:Person`. You bind it to a public vocabulary, such as `http://schema.org/Person` or `http://xmlns.com/foaf/0.1/Person`. By doing this, any other machine or database in the world instantly knows exactly what concept you are referring to, allowing separate datasets to merge seamlessly. This applies across different data modalities:
    *   **Unstructured Data**: NLP pipelines and entity extraction tools can semantically bind text by tagging words (e.g., the word "Apple" in a document) directly with a global IRI (e.g., `http://dbpedia.org/resource/Apple_Inc.`).
    *   **Structured Data (SQL / Ontop Virtualization)**: In an OBDA (Ontology-Based Data Access) setup using tools like **Ontop**, semantic binding is achieved via mapping languages (like R2RML). Crucially, **semantic binding is NOT a strict 1-to-1 mapping to the physical database structure**. While a "Direct Mapping" approach might blindly map 1 table to 1 class and 1 column to 1 property, true semantic binding is highly **domain-specific**. A binding connects a logical concept in the ontology to an arbitrary SQL query — which could span an entire table, a single column, a JOIN between three tables, or a filtered subset of rows. You are binding the *meaning* of the data to the ontology, regardless of how the physical database administrator chose to store it. For a concrete worked example showing three different binding strategies on the same two SQL tables, see [Part 11: Appendix — Graph-SQL Mapping]({% post_url 2026-4-14-enterprise-graph-part11-appendix-graph-sql-mapping %}).
*   **In LPG (Neo4j)**: Semantic binding is generally absent by default. If you label a node `:Apple`, the database doesn't know if you mean the fruit, the technology company, or a record label. The label is just a string of text relevant only to that specific database. If you want to merge two Neo4j databases, a human usually has to sit down and write a translation script to map the definitions together.
*   **In Modern Data Engineering (e.g., dbt Labs, Microsoft Fabric IQ)**: Outside of pure graph databases, "Semantic Binding" often refers to mapping business logic to physical data tables. In the **dbt Semantic Layer**, it is the process of binding abstract business metrics (like "Revenue" or "Active Users") to the underlying SQL columns and tables, allowing users to query metrics consistently across BI tools without knowing the physical schema. In **Microsoft Fabric IQ**, semantic binding connects the schema of business entity types (an ontology) to concrete data sources (like Data Lake tables) without moving the data, enriching the semantic layer with actual operational data for AI agents and analytics.

### Where Schema and Ontology Diverge: Concrete Examples

The definitions above can sound similar. The clearest way to tell them apart is to look at cases where one is present and the other is absent, or where the same sentence means something completely different depending on which lens you apply.

---

**Example 1: Schema Without Ontology — "An Employee must have an email"**

This is a pure schema claim. It is about data validity and enforcement.

```turtle
# SHACL (RDF's schema/validation language) — this is schema, NOT ontology
ex:EmployeeShape
    a sh:NodeShape ;
    sh:targetClass ex:Employee ;
    sh:property [
        sh:path schema:email ;
        sh:minCount 1 ;          # must have at least one email
        sh:datatype xsd:string ; # must be a string
    ] .
```

If a database record for an employee has no email, this constraint **rejects it**. The engine refuses to store the data. It says nothing whatsoever about what an Employee *means*, how it relates to a Person, or what can be inferred about it. It is purely structural enforcement — the database equivalent of a `NOT NULL` column constraint.

*The ontology equivalent of this would not reject anything. It would instead let you state that Employee is a subclass of Person, which allows the reasoner to infer new facts. That is a completely different purpose.*

---

**Example 2: Ontology Without Schema — "A Manager is a type of Employee"**

This is a pure ontology claim. It is about logical inference, not data validation.

```turtle
# OWL (ontology) — this is ontology, NOT schema
ex:Manager rdfs:subClassOf ex:Employee .
ex:Employee rdfs:subClassOf schema:Person .
```

No data is rejected by this. No constraint is enforced. Instead, the **reasoner** reads these axioms and automatically infers: any node declared as `ex:Manager` is also implicitly an `ex:Employee` and also implicitly a `schema:Person` — even if those types were never written into the database. The inference propagates silently.

A schema system would never produce this inference. It would at most check that a Manager node has the required properties — it would not promote the node to additional types it was never given.

---

**Example 3: The Same Words, Two Radically Different Meanings — `rdfs:domain`**

This is the subtlest and most important divergence. The keyword `rdfs:domain` looks like a schema constraint — it appears to restrict which classes are allowed to use a property. But in OWL, it is actually an **inference trigger**, not a validator.

```turtle
# This looks like a schema rule: "only Animals can have an owner"
ex:hasOwner rdfs:domain ex:Animal .
```

**If you read this as a schema claim**, you expect: any triple where a non-Animal is the subject of `ex:hasOwner` will be rejected or flagged as invalid.

**What OWL actually does**: if a node of type `ex:Plant` is found with an `ex:hasOwner` triple, the OWL reasoner does **not** reject it. Instead, it **infers** that the Plant must also be an `ex:Animal`. It adds that type silently. If you also declared `ex:Plant owl:disjointWith ex:Animal`, the reasoner would then detect a logical contradiction — but still would not "reject" data the way a schema does. It would flag an *inconsistency* that a human must resolve.

| Situation | Schema behavior | Ontology (OWL) behavior |
| :--- | :--- | :--- |
| Plant has `ex:hasOwner` | **Reject** the data — invalid | **Infer** Plant is also an Animal |
| Employee has two departments | **Reject** if cardinality = 1 | **Infer** the two departments are the same entity (`owl:sameAs`) |
| Manager node stored, no `ex:Employee` type | No action | **Infer** Manager is also an Employee (subclass inheritance) |

---

**Example 4: Schema Present, Ontology Absent — a Neo4j database**

A well-designed Neo4j database might have:

```cypher
CREATE CONSTRAINT ON (e:Employee) ASSERT e.email IS UNIQUE;
CREATE CONSTRAINT ON (e:Employee) ASSERT EXISTS(e.name);
```

This schema rigorously enforces data quality — no duplicates, no missing names. But if you ask the database "Is a Manager also an Employee?", "Can you infer that Alice knows Bob because they are both in the same department?", or "Is the `worksFor` here the same concept as `worksFor` in our partner's database?" — the database has no answer. Those questions require an ontology, and none exists here.

---

**The One-Sentence Summary**

> A **schema** is a gatekeeper: it rejects data that does not conform to the rules.
> An **ontology** is a reasoner: it accepts all data and then derives new facts from it.
> They solve different problems and can — and often should — exist side by side in the same system.

### Summary for your Glossary
To put it into a quick reference format for your readers:

| Concept | What it defines | LPG / Neo4j Approach | RDF / Semantic Web Approach |
| :--- | :--- | :--- | :--- |
| **Schema** | The Rules & Shape | Strict constraints (types, uniqueness) to ensure data integrity. | Basic vocabulary (RDFS) to define hierarchies, domains, and ranges. |
| **Ontology** | The Logic & Meaning | Not native. Handled in application code or via specialized plugins. | Core feature (OWL). Highly logical; used by machines to infer new facts automatically. |
| **Semantic Binding** | The Universal Context | Localized. A label like `:Person` only has meaning within that specific database. | Built-in via IRIs. A node is mapped to a global vocabulary (e.g., Schema.org) so any system understands it. |

> For a deep dive into how SQL and graph architectures are bridged — including OBDA/VKG, R2RML, DDL, Materialization vs. Virtualization, and Direct Mapping — see [Part 11: Appendix — Graph-SQL Mapping]({% post_url 2026-4-14-enterprise-graph-part11-appendix-graph-sql-mapping %}).

---

# Advanced Concept Pairs

These pairs go deeper than simple vocabulary translations. Each one reveals a genuine philosophical or architectural divergence between the two graph worlds.

---

## 1. Named Graphs (RDF) / No Native Equivalent (LPG)

### What a Named Graph Is

In standard RDF, data is a set of triples: `(Subject, Predicate, Object)`. A **Named Graph** adds a fourth element — a URI that labels the *entire set of triples* as a group: `(Subject, Predicate, Object, GraphName)`. This four-element structure is called a **quad**.

```turtle
# Default graph — no name, just triples
ex:Alice ex:knows ex:Bob .

# Named graph — the same triple, but attributed to a specific source
GRAPH <http://company.com/HR/2024-Q1> {
    ex:Alice ex:worksIn ex:Engineering .
    ex:Alice ex:salary  "80000"^^xsd:decimal .
}

GRAPH <http://company.com/HR/2024-Q2> {
    ex:Alice ex:salary  "85000"^^xsd:decimal .
}
```

The graph name is itself a URI, so it can carry metadata: who produced these triples, when, how trustworthy the source is, and which version of the data it represents.

### What Named Graphs Are Used For

*   **Provenance**: "These facts came from the HR system's Q1 report." You can query *which graph* a fact lives in, not just the fact itself.
*   **Versioning**: Two named graphs can hold the same predicate with different values at different points in time. You query a specific graph to get the value at that version.
*   **Access control**: Permissions can be applied at the named graph level — grant read access to `<http://company.com/public>` but restrict `<http://company.com/payroll>`.
*   **Retracting facts**: To delete a whole block of facts (e.g., "remove everything from the Q1 report"), you drop the named graph with one command. Without named graphs, you would need to know and list every triple to delete.
*   **Trust and confidence**: In federated Linked Data, you can weight facts differently depending on which named graph (which source) they come from.

```sparql
-- SPARQL: query across named graphs to see salary history
SELECT ?quarter ?salary
WHERE {
  ?quarter a ex:QuarterlyReport .
  GRAPH ?quarter {
    ex:Alice ex:salary ?salary .
  }
}
ORDER BY ?quarter
```

### The LPG Gap

Neo4j has no native named graph concept. The common workarounds each have costs:

| Workaround | How | Cost |
| :--- | :--- | :--- |
| **Property on every node/relationship** | Add `source: "HR-Q1"` to each element | Must remember to tag every single element; querying by source requires a full property scan |
| **Separate Neo4j databases** | One database per context | Cross-context queries require Fabric/Composite Databases; no unified query |
| **Context node** | Create a `(:Report)` node and connect every fact-node to it | Doubles the graph size; traversal becomes awkward |

None of these is as clean or query-efficient as a first-class named graph. Provenance in LPG is an afterthought; in RDF it is a built-in architectural primitive.

---

## 2. `owl:sameAs` (RDF) / `MERGE` (LPG) — and the Identity Spectrum

### The Core Problem: Entity Identity

Both systems must answer: *"When two records refer to the same real-world thing, how do we say so?"* The approaches are architecturally opposite.

### `MERGE` in LPG — an ETL operation

`MERGE` in Cypher is a **write-time deduplication command**. It finds a node matching a pattern and, if it exists, uses it; if not, creates it. The result is a single node in the database.

```cypher
-- If a Person with this email already exists, use it; otherwise create it
MERGE (p:Person {email: "alice@company.com"})
ON CREATE SET p.name = "Alice", p.created = timestamp()
ON MATCH  SET p.lastSeen = timestamp()
```

`MERGE` collapses duplicates *physically* at ingest time. Once merged, the two original records no longer exist as separate nodes. This is irreversible without a log. Identity is resolved once, permanently, by a human writing an ETL pipeline.

### `owl:sameAs` in RDF — a logical assertion

`owl:sameAs` is a **runtime logical assertion**. It does not collapse two nodes into one. Instead it declares that two IRIs are logically interchangeable — everything true of one is true of the other.

```turtle
# Alice in the HR system and Alice in the CRM are the same person
<http://hr.company.com/employee/1>  owl:sameAs  <http://crm.company.com/contact/42> .
```

The two IRIs remain separate. A reasoner, when asked about either one, will treat them as identical, propagating all properties across the `owl:sameAs` link automatically. The original source data is never modified.

### The `owl:sameAs` Problem at Scale

`owl:sameAs` is a very strong claim — full logical equivalence — and it carries three dangerous properties:

*   **Symmetric**: if A `sameAs` B, then B `sameAs` A. Your assertion propagates in both directions whether you intended it to or not.
*   **Transitive**: if A `sameAs` B and B `sameAs` C, then A `sameAs` C. A single incorrect link can chain together thousands of unrelated entities.
*   **Total**: every property of A becomes a property of B. A salary value stored for one IRI is now also the salary of everything linked by the chain.

In large Linked Data deployments, careless `owl:sameAs` usage causes what practitioners call the **"sameAs bomb"** — a transitive closure that merges thousands of unrelated entities into a single logical node.

### The Identity Spectrum: Alternatives to `owl:sameAs`

Because `owl:sameAs` is so strong, a range of weaker alternatives exists. These form a spectrum from logical certainty to probabilistic similarity:

| Expression | Strength | Meaning | Transitive in OWL? |
| :--- | :---: | :--- | :---: |
| `owl:sameAs` | Absolute | These two IRIs are the same entity in all possible worlds | Yes |
| `owl:equivalentClass` | High | These two *classes* have identical members | Yes |
| `skos:exactMatch` | High | These two *concepts* are interchangeable in most information retrieval contexts | No (by design) |
| `skos:closeMatch` | Medium | These two concepts are similar but not fully interchangeable | No |
| `skos:relatedMatch` | Low | These two concepts are associatively related | No |
| `schema:sameAs` | Varies | Schema.org's looser equivalent, used in web markup | No |
| `prov:wasDerivedFrom` | None | This entity was derived from that one (provenance, not identity) | No |
| Record linkage score | None (external) | A probabilistic match computed by an algorithm (e.g., 92% likely same entity) | N/A |

**The practical recommendation**: use `skos:exactMatch` when aligning two controlled vocabularies or datasets that have approximately the same concept. Reserve `owl:sameAs` only for cases where you can guarantee full logical equivalence — typically when you control both IRIs and know they were always meant to be the same thing.

In LPG, the equivalent to this spectrum is a **custom relationship type on the edge** between two duplicate candidate nodes:

```cypher
-- Instead of physically merging, keep both nodes and express confidence
(alice_hr)-[:PROBABLY_SAME_AS {confidence: 0.92, method: "email-match"}]->(alice_crm)
(alice_hr)-[:SAME_AS]->(alice_crm)
```

This preserves both source records and makes the identity claim inspectable and reversible — at the cost of requiring application logic to handle the ambiguity.

---

## 3. Instance / Class Fluidity (RDF) vs. Strict Instance-Label Separation (LPG)

### The LPG Model: Two Strictly Separate Layers

In a Labeled Property Graph, the data world has exactly two layers that never mix:

*   **Nodes**: instances of things that exist in the domain (Alice, the Engineering department, Invoice #42).
*   **Labels**: category tags applied to nodes (`:Person`, `:Department`, `:Invoice`).

A label is just a string. It cannot have properties, it cannot be connected to other labels via relationships, and it cannot itself be a node. You cannot traverse from a node to its label as if the label were a first-class citizen.

```cypher
-- You CANNOT do this in Cypher -- labels are not nodes
MATCH (label:Label {name: "Person"})-[:SUBCLASS_OF]->(parent:Label {name: "LegalEntity"})

-- Instead, you model hierarchies as a SEPARATE graph of nodes
MATCH (label:Category {name: "Person"})-[:SUBCLASS_OF]->(parent:Category {name: "LegalEntity"})
```

If you want a taxonomy or class hierarchy in Neo4j, you must build it as a second graph of `:Category` (or `:Class`) nodes — entirely separate from the instance nodes that carry `:Person` labels. The label and the category node are unconnected by default. Keeping them in sync requires custom application logic.

### The RDF Model: Everything Is a Resource

In RDF, there is only one kind of thing: a **resource** (identified by an IRI). A class is a resource. An instance is a resource. A property is a resource. The same IRI can simultaneously be used as a class (having instances declared via `rdf:type`) and as an instance (being declared as an instance of some other class via `rdf:type`). This is called **metaclass** structure.

```turtle
# Dog is a class — it has instances (Fido, Rex)
ex:Dog  a  rdfs:Class .
ex:Fido a  ex:Dog .

# Dog is ALSO an instance — it is a member of the class ex:Species
ex:Dog  a  ex:Species .

# Species is ALSO an instance — it is a member of ex:TaxonomicRank
ex:Species  a  ex:TaxonomicRank .

# And Dog can have properties, just like any resource
ex:Dog  ex:averageLifespan  "12 years" .
ex:Dog  ex:kingdom          ex:Animalia .
```

Here, `ex:Dog` is simultaneously a class (it classifies Fido) and an instance (it is classified by `ex:Species`). The class hierarchy and the instance data live in the same graph, use the same query language, and can be traversed together in a single SPARQL query.

### A Concrete Scenario: A Legal Ontology

Consider modeling courts. A legal system has:

*   Specific courts: "Supreme Court of the United States" (an instance)
*   Court types: "Constitutional Court" (a class that groups specific courts)
*   Court categories: "Judicial Body" (a class that groups court types)

**In LPG (Neo4j)** you need two separate subgraphs:

```
Instance graph:                    Taxonomy graph:
(:Court {name:"SCOTUS"})           (:Category {name:"ConstitutionalCourt"})
   |                                      |
   | :HAS_TYPE (application code)         | :SUBCLASS_OF
   ▼                                      ▼
(:Category {name:                  (:Category {name:"JudicialBody"})
  "ConstitutionalCourt"})
```

The `:Court` label and the `(:Category {name:"ConstitutionalCourt"})` node are separate objects. You connect them with a relationship, but nothing enforces that every `:Court` node is connected to the taxonomy, and querying across the two layers requires traversing that bridge relationship.

**In RDF**, there is only one graph:

```sparql
-- A single SPARQL query can traverse both instance and class layers simultaneously
SELECT ?courtName ?categoryName
WHERE {
  ?court  a              ex:ConstitutionalCourt ;
          ex:name        ?courtName .
  ex:ConstitutionalCourt
          rdfs:subClassOf  ?category .
  ?category  ex:name     ?categoryName .
}
```

`ex:ConstitutionalCourt` serves as both the class of `?court` and a resource that itself has properties and relationships in the same query.

### The Trade-Off

| Dimension | LPG (strict separation) | RDF (fluid) |
| :--- | :--- | :--- |
| **Simplicity** | Clear mental model — nodes are data, labels are tags | Steeper learning curve — everything is a resource |
| **Query performance** | Label-based filtering is very fast (native index) | Class-based filtering requires `rdf:type` traversal, which varies by store |
| **Taxonomy modeling** | Requires a separate taxonomy node graph | Taxonomy and instances live in one unified graph |
| **Metaclass support** | Not native — requires workarounds | First-class — a class can be an instance of another class |
| **Tooling** | Familiar to developers from OOP | Familiar to logicians and knowledge engineers |

---

## 4. T-box vs. A-box — The Vocabulary and the Facts

### Where These Terms Come From

T-box and A-box are not database engineering terms — they come from **Description Logic (DL)**, the branch of formal logic that is the mathematical foundation of OWL. Understanding them clarifies a distinction that runs underneath everything else in this glossary: the difference between *defining the vocabulary of a domain* and *asserting facts using that vocabulary*.

*   **T-box (Terminological Box)**: The layer that defines *what kinds of things exist* and *what rules govern them*. Class definitions, property definitions, subclass hierarchies, domain and range restrictions, cardinality constraints, disjointness axioms — all of these are T-box statements. The T-box is the ontology.

*   **A-box (Assertional Box)**: The layer that asserts *which specific things exist* and *what is true about them*. Statements like "Fido is a Dog," "Alice works in Engineering," "Invoice #42 has amount 500" — these are all A-box statements. The A-box is the instance data.

```turtle
# T-box statements — defining the vocabulary
ex:Dog        rdfs:subClassOf  ex:Animal .
ex:hasOwner   rdfs:domain      ex:Animal ;
              rdfs:range       ex:Person .
ex:Manager    rdfs:subClassOf  ex:Employee .

# A-box statements — asserting facts about individuals
ex:Fido       a                ex:Dog .
ex:Fido       ex:hasOwner      ex:Alice .
ex:Alice      a                ex:Manager .
```

### The Key Point: In RDF, Both Live in the Same Graph

This is what surprises most people. In RDF and OWL, T-box and A-box statements are **syntactically identical** — they are both just triples. There is no separate file, no separate table, no structural boundary. The distinction is purely conceptual. A reasoner reads all the triples together and uses the T-box triples to derive new A-box triples:

```
T-box says: ex:Manager rdfs:subClassOf ex:Employee
A-box says:  ex:Alice  a  ex:Manager
Reasoner infers (new A-box): ex:Alice  a  ex:Employee
```

The inference crosses the T-box/A-box boundary — that is precisely what a reasoner does. The T-box is the rule engine; the A-box is the data the rules operate on.

### In LPG (Neo4j): Only A-box Exists Natively

A Neo4j database is pure A-box. The nodes and relationships are instance assertions. There is no formal T-box:

*   The label `:Manager` is not a defined class — it is a string tag. There is no built-in statement that `:Manager` is a subtype of `:Employee`.
*   There are no domain/range restrictions — a `WORKS_IN` relationship can connect any two nodes of any label.
*   Cardinality, disjointness, and inverse properties do not exist as queryable or inferable statements.

The T-box, when it exists at all in a Neo4j system, lives in three places:
1. **Developer documentation** — informal, not machine-readable
2. **Application code** — enforced through Cypher query logic, not the database itself
3. **Imported OWL ontologies** (via the Neosemantics/n10s plugin) — bringing RDF T-box into Neo4j as a special subgraph of nodes

### How T-box/A-box Relates to Schema, Ontology, and Semantic Binding

This is where all the concepts in this glossary connect:

```
Description Logic term    Glossary equivalent         Typical technology
──────────────────────    ───────────────────         ───────────────────
T-box                  ≈  Ontology                    OWL / RDFS
A-box                  ≈  Instance data               RDF triples / graph nodes
(no DL term)           ≈  Schema / validation         SHACL / Neo4j constraints
```

The schema (SHACL, Neo4j constraints) is a third layer that sits alongside both — it does not define meaning (T-box) and is not instance data (A-box). It is a set of data quality rules imposed on the A-box, checked against expectations derived from the T-box. A complete, rigorous RDF system therefore has all three layers simultaneously:

| Layer | What it does | Example |
| :--- | :--- | :--- |
| **T-box (Ontology)** | Defines classes, properties, logical rules | `ex:Manager rdfs:subClassOf ex:Employee` |
| **A-box (Instance data)** | Asserts facts about individuals | `ex:Alice a ex:Manager` |
| **Schema (SHACL)** | Validates A-box data against T-box expectations | "Every `ex:Manager` must have exactly one `ex:manages` relationship" |

In LPG, only the middle row exists natively.

---

## 5. Graph Analytics Algorithms (PageRank, Centrality) — Neither RDF nor LPG, and Why That Matters

### What These Algorithms Are

PageRank, betweenness centrality, closeness centrality, and community detection algorithms are **graph topology algorithms** — they treat the graph as a pure mathematical structure of nodes and edges and compute numerical measures of importance, influence, or position. They are not specific to RDF or LPG. They belong to the field of **network science** and were developed entirely independently of the graph database world.

*   **PageRank**: Measures the importance of a node by the number and quality of edges pointing to it. Originally Google's algorithm for ranking web pages; a node is important if many important nodes point to it. It is an iterative algorithm that propagates scores across the entire graph.
*   **Degree Centrality**: Simply counts the number of direct connections a node has. High degree = many direct neighbors.
*   **Betweenness Centrality**: Counts how often a node appears on the shortest path between every other pair of nodes. High betweenness = a bridge or broker in the graph.
*   **Closeness Centrality**: Measures how close a node is (in average shortest path length) to all other nodes. High closeness = well-positioned to spread information quickly.
*   **Eigenvector Centrality**: A node is important if its neighbors are important — recursive importance. PageRank is a damped variant of this.
*   **Community Detection** (e.g., Louvain, Label Propagation): Finds clusters of nodes that are more densely connected to each other than to the rest of the graph.

### Where They Live: LPG, Not RDF

While these algorithms are mathematically graph-agnostic, in practice they are almost exclusively associated with **LPG systems**:

*   **Neo4j Graph Data Science (GDS) library**: Native implementations of PageRank, all major centrality algorithms, community detection, pathfinding, similarity, and machine learning pipelines — all operating directly on the in-memory graph.
*   **RDF triple stores**: Generally have no built-in graph analytics. To run PageRank on an RDF graph, you would typically export the graph to a separate analytics tool (Apache Spark GraphX, NetworkX, graph analytics platforms) and run the algorithm there, then optionally write results back.

The reason is architectural: LPGs are optimized for **graph traversal** (index-free adjacency — every node holds direct pointers to its neighbors). This makes the repeated neighbor lookups that centrality algorithms require extremely fast. RDF triple stores are optimized for **semantic pattern matching** (SPARQL), which is a different access pattern.

### The Semantic Tension

There is a deeper conceptual mismatch between graph analytics and RDF that is worth naming explicitly:

**Graph analytics treat all edges as equivalent structure.** PageRank does not care whether an edge means `ex:knows`, `ex:worksIn`, or `ex:purchased` — it counts the edge. The semantic meaning is invisible to the algorithm.

**RDF is fundamentally about the meaning of edges.** In RDF, the predicate is a first-class resource with a URI and a formal definition. Two edges with different predicates are not interchangeable — they mean different things, and a reasoner exploits that difference.

This creates a specific problem when running topology algorithms on an RDF graph: **what counts as a node?**

In an LPG, the answer is simple — graph nodes are nodes, relationships are edges, and the algorithms operate on that structure.

In an RDF graph, predicates are also resources (IRIs) — they can appear as subjects of other triples. If you naively project the triple store into a graph for PageRank, do predicate-IRIs become nodes? If so, highly reused predicates like `rdf:type` or `schema:name` would appear as astronomically central nodes, distorting every centrality measure. You must first decide which triples to include and how to project the RDF graph onto a simple node-edge structure before any algorithm can run meaningfully.

### Summary

| Dimension | LPG (Neo4j) | RDF Triple Store |
| :--- | :--- | :--- |
| **Built-in analytics** | Yes — Neo4j GDS library | No — requires external tools |
| **Architectural fit** | High — index-free adjacency optimized for traversal | Low — optimized for semantic pattern matching |
| **Edge semantics in algorithms** | Edges are typed but algorithms can ignore type | Predicates are first-class resources; projection is non-trivial |
| **Typical use** | Fraud detection, recommendations, influence analysis | Not typical — analytics run on exported data |
| **What "centrality" means** | Well-defined over nodes and typed edges | Ambiguous until you decide how to project predicates |