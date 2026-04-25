---
layout: post
title: "Enterprise Graph Part 11: Appendix - Graph-SQL Mapping"
date: 2026-04-14
categories: [Enterprise Graph, RDF, LPG, SQL]
---
<br>

# Bridging SQL and Graph Architectures

The terms below define the key concepts, tools, and philosophical trade-offs that arise when connecting traditional relational (SQL) databases to graph architectures — whether RDF triple stores or Labeled Property Graphs like Neo4j.

### 1. OBDA (Ontology-Based Data Access) / VKG (Virtual Knowledge Graph)
*   **What it is**: A Semantic Web architecture that creates a "virtual" graph layer over traditional relational databases. Instead of physically moving data out of SQL tables and into a graph database, OBDA uses an ontology as a translation lens.
*   **How it works**: When a user queries the virtual graph using SPARQL, the OBDA engine intercepts the query, translates the SPARQL into SQL on-the-fly, runs the SQL against the underlying relational database, and returns the results as graph concepts.
*   **Where it fits**: This is specific to the RDF / Semantic Web stack (using tools like Ontop or Stardog). In recent years, the industry has rebranded OBDA as VKG (Virtual Knowledge Graph) to sound more accessible to business users.
*   **LPG Equivalent**: Native Neo4j does not do this natively. To query SQL data, Neo4j typically requires physically importing the data first (ETL) or using custom middleware (like GraphQL or custom API gateways) to fetch data from external SQL databases on demand.

### 2. R2RML (RDB to RDF Mapping Language)
*   **What it is**: The W3C standard mapping language that makes OBDA possible. It is the instruction manual that tells the machine exactly how to translate a relational database schema into RDF triples.
*   **How it works**: It maps a SQL table row to an RDF Subject (the node), the table columns to Predicates (the edges), and the cell values to Objects (the literal values).
*   **Where it fits**: Strictly RDF.
*   **LPG Equivalent**: Neo4j does not have a standardized mapping language for SQL. Instead, developers write custom Cypher scripts, often using the `LOAD CSV` command, or rely on ETL (Extract, Transform, Load) integration tools to physically map tabular data into Labeled Property Graphs.

### 3. DDL (Data Definition Language)
*   **What it is**: A subset of standard SQL commands (CREATE, ALTER, DROP) used to define the strict, tabular structure of a relational database.
*   **How it relates to Graphs**:
    *   **In RDF (OBDA)**: The SQL DDL is the rigid structure that R2RML mappings are trying to abstract away. The goal of an ontology is to hide the DDL so users can query data logically (e.g., "Find all Employees") rather than structurally (e.g., "Join table_emp with table_dept").
    *   **In LPG (Neo4j)**: Because Neo4j is schema-optional, it has no direct DDL. However, when migrating from SQL to Neo4j, developers use Neo4j constraint commands (e.g., `CREATE CONSTRAINT`) to mimic DDL, ensuring that imported tabular data maintains its integrity (like unique IDs) in the graph environment.

### 4. Materialization vs. Virtualization
These are the two competing philosophies for bridging SQL and Graph architectures.

*   **Virtualization (The OBDA Approach)**: The graph does not physically store the data. It acts as a semantic proxy, translating graph queries to SQL at runtime.
    *   **Pros**: Data is always perfectly real-time; no duplicate storage.
    *   **Cons**: Graph traversal speeds are limited by the speed of the underlying SQL JOIN operations.
    *   **In Modern Data Platforms**: **Microsoft Fabric IQ** (and similar "Zero-ETL" lakehouse approaches) leans heavily into virtualization. It binds its semantic ontology directly to underlying OneLake data sources without copying or moving the data, querying it in place.
*   **Materialization (The ETL Approach)**: The process of extracting SQL data, physically transforming it into nodes and edges, and storing it permanently inside a graph database like Neo4j or an RDF Triple Store.
    *   **Pros**: Enables lightning-fast, deep graph traversals (Index-Free Adjacency).
    *   **Cons**: Requires duplicating the data and building pipelines to keep the graph and the SQL database in sync.
    *   **In Modern Data Platforms**: **dbt (data build tool)** is primarily a materialization engine. It focuses on the "T" in ELT, running SQL transformations to physically build and store tables, views, or incremental models in a data warehouse. (Note: dbt's newer *Semantic Layer* does offer virtualized metric querying on top of those materialized tables, blending the two approaches).

### 5. Direct Mapping
*   **What it is**: A W3C standard for automatically generating a basic RDF graph from a relational database without writing a complex R2RML mapping file. It takes the SQL DDL blindly and converts tables into Classes, columns into Properties, and foreign keys into Relationships.
*   **Where it fits**: It is a quick-start method for RDF developers. However, it usually falls short for complex domain modeling because it perfectly mirrors the relational database's shape rather than applying a true, meaningful ontology.

---

# OBDA, DDL, and R2RML: How They Relate

A common point of confusion is whether OBDA, DDL, and R2RML are the same file, three different files, or some mix of both. The short answer is: **they are three entirely different artifacts, serving three different roles, and none of them is the same thing.** Here is a precise breakdown.

### What Each One Is

**DDL is not a file you write for OBDA. It already exists.**

DDL (Data Definition Language) is the structural definition of the source relational database — the `CREATE TABLE` commands that built the database in the first place. It defines the physical shape: tables, columns, data types, primary keys, and foreign keys. When you approach a relational database as an OBDA source, the DDL is already there. You read it to understand the shape of the data, but you do not write or modify it as part of the OBDA setup.

```sql
-- This is DDL. It already exists in the source database.
-- You did not write it. The database team did.
CREATE TABLE employees (
    id      INT PRIMARY KEY,
    name    VARCHAR,
    role    VARCHAR,
    salary  DECIMAL,
    dept_id INT REFERENCES departments(id)
);
```

**The Ontology is a file you write (or adopt). It defines meaning.**

An OWL or RDFS ontology file (`.owl`, `.ttl`) defines the semantic vocabulary of your domain: what classes exist (`schema:Person`, `ex:Manager`), what properties they have, and what logical relationships hold between them. It lives entirely in the semantic/graph world and has no knowledge of SQL tables. This is the "target" layer of the OBDA system.

```turtle
# This is the ontology file (e.g., company-ontology.ttl).
# It defines what concepts exist in your domain.
ex:Manager rdfs:subClassOf schema:Person .
ex:manages rdfs:domain ex:Manager ;
           rdfs:range  ex:Department .
```

**R2RML is a file you write. It is the bridge.**

R2RML (`.ttl` in Turtle syntax, or Ontop's native `.obda` format) is the mapping file. It is the only artifact that has knowledge of *both* worlds simultaneously. Each mapping rule says: "take the results of this SQL query, and express them as these RDF triples pointing at these ontology classes." R2RML reads the DDL-defined tables on one side and produces instances of the ontology classes on the other.

```turtle
# This is the R2RML mapping file (e.g., company-mapping.ttl).
# It bridges the DDL world and the ontology world.
<#ManagerMap>
    rr:logicalTable [ rr:sqlQuery
        "SELECT id, name, salary, dept_id
         FROM employees WHERE role = 'Manager'" ] ;
    rr:subjectMap [
        rr:template "http://example.org/employee/{id}" ;
        rr:class ex:Manager ] ;
    rr:predicateObjectMap [
        rr:predicate schema:name ;
        rr:objectMap [ rr:column "name" ] ] .
```

**OBDA is not a file at all. It is the running system.**

OBDA (Ontology-Based Data Access) is the name for the overall architecture — the engine (e.g., Ontop) that holds all three artifacts together and answers SPARQL queries by translating them into SQL at runtime. You deploy OBDA; you do not write it.

---

### The Complete File Inventory for an Ontop OBDA Setup

To stand up a working OBDA system with Ontop, you need exactly these files:

```
your-obda-project/
│
├── database/                        ← pre-existing, you do NOT create this
│   └── (relational database with DDL already applied)
│
├── ontology.ttl                     ← you write this (or adopt schema.org, etc.)
│   (defines classes, properties, logical axioms in OWL/RDFS)
│
├── mapping.ttl  (or mapping.obda)   ← you write this
│   (R2RML rules bridging SQL queries → ontology classes)
│
└── ontop.properties                 ← you write this
    (JDBC connection string, driver, credentials for the database)
```

**Do you need all of them?** Yes, all four are required for Ontop to run. The only shortcut is **Direct Mapping**: Ontop can auto-generate both the ontology and the R2RML mapping directly from the DDL, skipping the two files you write. But as discussed earlier, Direct Mapping produces a graph that mirrors the relational schema rather than expressing real domain meaning.

---

### How the Three Artifacts Relate at Runtime

```
  ┌──────────────────┐    R2RML reads     ┌──────────────────┐
  │   DDL            │◀───both sides──────│   R2RML          │
  │   (SQL schema)   │                    │   (mapping file) │
  │   EXISTS already │                    │   YOU WRITE THIS │
  └──────────────────┘                    └────────┬─────────┘
           │                                       │
           │ Ontop reads DDL                       │ Ontop reads R2RML
           │ to reach the data                     │ to know what triples to produce
           ▼                                       ▼
  ┌──────────────────────────────────────────────────────────┐
  │                  ONTOP ENGINE (OBDA system)              │
  │  Intercepts SPARQL → rewrites to SQL → returns triples   │
  └──────────────────────────────────────────────────────────┘
           │                                       │
           │ consults                              │ produces triples conforming to
           ▼                                       ▼
  ┌──────────────────┐                    ┌──────────────────┐
  │   Relational DB  │                    │   Ontology       │
  │   (actual data)  │                    │   (.owl / .ttl)  │
  │   EXISTS already │                    │   YOU WRITE THIS │
  └──────────────────┘                    └──────────────────┘
```

### Summary

| Artifact | Is it a file? | Who writes it? | What world does it live in? | Required? |
| :--- | :---: | :--- | :--- | :---: |
| **DDL** | Yes (SQL) | Database team (pre-existing) | Relational (SQL) only | Pre-exists |
| **Ontology** | Yes (.owl / .ttl) | Domain / ontology expert | Semantic (graph) only | Yes |
| **R2RML** | Yes (.ttl / .obda) | Mapping engineer | Both — it bridges the two | Yes |
| **OBDA / Ontop** | No (it's a system) | You deploy it | Neither — it runs between the two | Yes |

The key insight is that **R2RML is the only artifact that speaks both languages**. The DDL speaks only SQL; the ontology speaks only RDF. R2RML is the translator that makes it possible for the OBDA engine to answer a SPARQL question with a SQL query.

---

# Semantic Binding in Practice: Three Mapping Strategies

The glossary definition of Semantic Binding explains that a binding is not a 1-to-1 mirror of physical database structure — it is a domain-driven mapping from a SQL query to an ontological concept. The following worked example makes this concrete.

> **Are these bindings the R2RML or the ontology?**
>
> The binding rules in Strategies A, B, and C below are all **R2RML** — they live in the mapping file, not the ontology file. Each binding rule is a SQL query on one side and a reference to an ontology class or property on the other.
>
> The **ontology** is a prerequisite but a separate file. It defines what `ex:Manager`, `ex:manages`, `ex:Department`, etc. *mean* in the domain — their logical relationships, subclass hierarchies, and property constraints. The R2RML then populates those ontology-defined concepts with actual rows from the database.
>
> Crucially, **the strategy you choose for R2RML determines what your ontology must define**. If you choose Strategy B (role-filtered), your ontology must declare `ex:Manager` as a subclass of `schema:Person` and define `ex:manages` as a property with the right domain and range. If you choose Strategy A (direct), none of that is needed. The two files are co-designed: the ontology defines the vocabulary, and R2RML decides how database rows populate it.
>
> ```
> Ontology (.ttl)          R2RML mapping (.ttl / .obda)       Database (DDL)
> ─────────────────        ──────────────────────────────     ────────────────
> Defines ex:Manager  ◀─── target: ex:employee/{id}      ◀── source: SELECT *
> subClassOf               a ex:Manager                       FROM employees
> schema:Person                                               WHERE role='Manager'
>
> Defines ex:manages  ◀─── target: ex:employee/{id}
> domain: ex:Manager       ex:manages ex:dept/{dept_id}
> range: ex:Department
> ```
>
> Think of it this way: the **ontology** is the dictionary (what words mean), and **R2RML** is the translation guide (how to express database rows using those words).

**The Data: Two SQL Tables with a Foreign Key**

```sql
-- Table 1
CREATE TABLE employees (
    id       INT PRIMARY KEY,
    name     VARCHAR,
    role     VARCHAR,   -- e.g. 'Engineer', 'Manager'
    salary   DECIMAL,
    dept_id  INT        -- FOREIGN KEY → departments.id
);

-- Table 2
CREATE TABLE departments (
    id       INT PRIMARY KEY,
    name     VARCHAR,
    location VARCHAR
);
```

You now have a choice of how to bind these two tables to your ontology. The three strategies below produce radically different virtual knowledge graphs from the *same* physical data. To make the difference tangible, all three strategy graphs below are drawn using these concrete rows:

```
employees                                    departments
─────────────────────────────────────────    ─────────────────────────────────
id │ name    │ role      │ salary  │ dept_id  id │ name          │ location
───┼─────────┼───────────┼─────────┼────────  ───┼───────────────┼──────────
1  │ Alice   │ Engineer  │  80000  │ 10       10 │ Engineering   │ New York
2  │ Bob     │ Manager   │ 120000  │ 10       20 │ Marketing     │ Boston
3  │ Carol   │ Engineer  │  75000  │ 20
```

---

### Strategy A — Direct (Naïve) Mapping: 1 table = 1 class, 1 column = 1 property

This is the simplest possible binding. Every table becomes one RDF class, every column becomes one property, and the foreign key becomes a relationship.

```turtle
# Binding 1 — employees table → schema:Person nodes
[MappingId] employees-to-person
  source  SELECT id, name, role, salary, dept_id FROM employees
  target  ex:employee/{id} a schema:Person ;
              schema:name        {name} ;
              ex:role            {role} ;
              ex:salary          {salary} ;
              ex:worksIn         ex:department/{dept_id} .

# Binding 2 — departments table → ex:Department nodes
[MappingId] departments-to-dept
  source  SELECT id, name, location FROM departments
  target  ex:department/{id} a ex:Department ;
              schema:name     {name} ;
              ex:location     {location} .
```

Result: **2 bindings** for 2 tables. Every employee row becomes a `schema:Person` node, every department row becomes an `ex:Department` node, and the FK becomes an `ex:worksIn` edge. Simple — but it perfectly mirrors the physical database shape and carries no additional domain meaning.

**What the Virtual Graph Looks Like — Strategy A**

```
  ┌──────────────────────────────────┐      ┌───────────────────────────────────┐
  │ ex:employee/1                    │      │ ex:department/10                  │
  │ a: schema:Person                 │      │ a: ex:Department                  │
  │ schema:name  = "Alice"           ├──────▶ schema:name  = "Engineering"      │
  │ ex:role      = "Engineer"        │      │ ex:location  = "New York"         │
  │ ex:salary    = 80000             │  ┌───▶                                   │
  └──────────────────────────────────┘  │   └───────────────────────────────────┘
                  ex:worksIn            │
  ┌──────────────────────────────────┐  │
  │ ex:employee/2                    │  │   ┌───────────────────────────────────┐
  │ a: schema:Person                 │  │   │ ex:department/20                  │
  │ schema:name  = "Bob"             ├──┘   │ a: ex:Department                  │
  │ ex:role      = "Manager"         │      │ schema:name  = "Marketing"        │
  │ ex:salary    = 120000            │      │ ex:location  = "Boston"           │
  └──────────────────────────────────┘  ┌───▶                                   │
                  ex:worksIn            │   └───────────────────────────────────┘
  ┌──────────────────────────────────┐  │
  │ ex:employee/3                    │  │
  │ a: schema:Person                 │  │
  │ schema:name  = "Carol"           ├──┘
  │ ex:role      = "Engineer"        │
  │ ex:salary    = 75000             │
  └──────────────────────────────────┘
```

All three employees are the same class. Bob the Manager and Alice the Engineer are indistinguishable at the graph level — they are both `schema:Person`. The department is a separate node, and the FK lives on the edge. The graph topology perfectly mirrors the two-table relational schema.

**SPARQL & SQL Equivalents under Strategy A**

*Query 1 — Find all employees and their department location.*

Under Strategy A, traversing `ex:worksIn` in SPARQL maps exactly to the FK JOIN in SQL because the binding explicitly connects `dept_id` to `ex:department/{dept_id}`.

```sparql
-- SPARQL
SELECT ?name ?location
WHERE {
  ?emp a schema:Person ;
       schema:name ?name ;
       ex:worksIn  ?dept .
  ?dept ex:location ?location .
}
```

```sql
-- SQL (what Ontop executes against the database)
SELECT e.name, d.location
FROM   employees e
JOIN   departments d ON e.dept_id = d.id
```

The two-hop graph traversal `(Person)-[worksIn]->(Department)` translates to exactly one JOIN, because the binding put the FK on the edge.

*Query 2 — Find all employees earning more than 70,000.*

Filtering on a SPARQL property value maps to a SQL `WHERE` clause on the bound column.

```sparql
-- SPARQL
SELECT ?name ?salary
WHERE {
  ?emp a schema:Person ;
       schema:name ?name ;
       ex:salary   ?salary .
  FILTER(?salary > 70000)
}
```

```sql
-- SQL
SELECT name, salary
FROM   employees
WHERE  salary > 70000
```

The `FILTER` in SPARQL becomes a `WHERE` predicate in SQL directly on the `employees` table — no JOIN needed, because salary is a property of the `schema:Person` binding, not of the `ex:Department` binding.

---

### Strategy B — Role-Filtered Mapping: 1 WHERE clause = 1 distinct class

The domain expert decides that `Manager` is a meaningfully different concept from a generic `Employee`. The mapping uses a `WHERE` clause to split one physical table into two distinct ontological classes.

```turtle
# Binding 1 — only non-managers → schema:Person
[MappingId] engineers-to-person
  source  SELECT id, name, salary, dept_id FROM employees
          WHERE role <> 'Manager'
  target  ex:employee/{id} a schema:Person ;
              schema:name {name} ; ex:salary {salary} ;
              ex:worksIn  ex:department/{dept_id} .

# Binding 2 — only managers → ex:Manager (a subclass of schema:Person)
[MappingId] managers-to-manager
  source  SELECT id, name, salary, dept_id FROM employees
          WHERE role = 'Manager'
  target  ex:employee/{id} a ex:Manager ;
              schema:name {name} ; ex:salary {salary} ;
              ex:manages  ex:department/{dept_id} .

# Binding 3 — departments table (unchanged)
[MappingId] departments-to-dept
  source  SELECT id, name, location FROM departments
  target  ex:department/{id} a ex:Department ;
              schema:name {name} ; ex:location {location} .
```

Result: **3 bindings** from the same 2 tables. The single `employees` table is now split into two semantically distinct classes (`schema:Person` and `ex:Manager`), and the FK relationship takes on a different predicate (`ex:worksIn` vs. `ex:manages`) depending on the role. The number of bindings has no relationship to the number of physical tables.

**What the Virtual Graph Looks Like — Strategy B**

```
  ┌──────────────────────────────────┐      ┌───────────────────────────────────┐
  │ ex:employee/1                    │      │ ex:department/10                  │
  │ a: schema:Person                 │      │ a: ex:Department                  │
  │ schema:name  = "Alice"           ├──────▶ schema:name  = "Engineering"      │
  │ ex:salary    = 80000             │      │ ex:location  = "New York"         │
  └──────────────────────────────────┘      │                                   │
                  ex:worksIn            ┌───▶                                   │
  ┌──────────────────────────────────┐  │   └───────────────────────────────────┘
  │ ex:employee/2                    │  │
  │ a: ex:Manager          ◀─ NOTE   │  │
  │ schema:name  = "Bob"             ├──┘
  │ ex:salary    = 120000            │        ex:manages (not ex:worksIn!)
  └──────────────────────────────────┘

  ┌──────────────────────────────────┐      ┌───────────────────────────────────┐
  │ ex:employee/3                    │      │ ex:department/20                  │
  │ a: schema:Person                 │      │ a: ex:Department                  │
  │ schema:name  = "Carol"           ├──────▶ schema:name  = "Marketing"        │
  │ ex:salary    = 75000             │      │ ex:location  = "Boston"           │
  └──────────────────────────────────┘      └───────────────────────────────────┘
                  ex:worksIn
```

Bob is now a structurally different node type (`ex:Manager`) with a different outgoing edge label (`ex:manages`). The `ex:role` property has been promoted from a plain data value into ontological class membership — meaning it is now encoded in the graph topology itself rather than stored as a property. Because the ontology declares `ex:Manager rdfs:subClassOf schema:Person`, Ontop's OWL 2 QL reasoning automatically expands any query for `schema:Person` to include all subclasses — so Bob is returned alongside Alice and Carol when a SPARQL user queries for `schema:Person`. To retrieve only non-managers, an explicit `FILTER NOT EXISTS { ?emp a ex:Manager }` is required.

**SPARQL & SQL Equivalents under Strategy B**

*Query 1 — Find all managers and the departments they manage.*

Querying for the class `ex:Manager` in SPARQL is equivalent to filtering `WHERE role = 'Manager'` in SQL, because the binding that produces `ex:Manager` nodes was defined on exactly that filtered source query.

```sparql
-- SPARQL
SELECT ?name ?deptName
WHERE {
  ?emp a ex:Manager ;
       schema:name ?name ;
       ex:manages  ?dept .
  ?dept schema:name ?deptName .
}
```

```sql
-- SQL
SELECT e.name, d.name AS dept_name
FROM   employees e
JOIN   departments d ON e.dept_id = d.id
WHERE  e.role = 'Manager'
```

The `a ex:Manager` type constraint in SPARQL carries the hidden `WHERE role = 'Manager'` filter — the user never writes it explicitly, but the binding encodes it invisibly.

*Query 2 — Find all employees (managers included, via subclass inference).*

Because the ontology declares `ex:Manager rdfs:subClassOf schema:Person`, Ontop's OWL 2 QL reasoner expands `?emp a schema:Person` to also cover all instances of every declared subclass. The engine internally rewrites the query into a UNION over all bindings whose target class is `schema:Person` or a subclass of it — which includes both the non-manager binding and the manager binding.

```sparql
-- SPARQL
SELECT ?name
WHERE {
  ?emp a schema:Person ;
       schema:name ?name .
}
```

```sql
-- SQL (Ontop unfolds across both bindings, then the optimizer collapses the UNION)
SELECT name FROM employees WHERE role <> 'Manager'
UNION
SELECT name FROM employees WHERE role = 'Manager'
-- collapses to:
SELECT name FROM employees
```

This is the key benefit of declaring the subclass hierarchy in the ontology: a plain `?emp a schema:Person` query naturally spans all employees, including managers, without requiring the SPARQL user to know about the class split.

*Query 2b — Find only non-manager employees.*

To explicitly exclude managers, a `FILTER NOT EXISTS` pattern is required against the `ex:Manager` class. The subclass boundary no longer acts as an implicit filter; the filter must be stated.

```sparql
-- SPARQL
SELECT ?name
WHERE {
  ?emp a schema:Person ;
       schema:name ?name .
  FILTER NOT EXISTS { ?emp a ex:Manager }
}
```

```sql
-- SQL
SELECT name
FROM   employees
WHERE  role <> 'Manager'
```

*Query 3 — Find everyone (managers and non-managers alike).*

Because the ontology declares `ex:Manager rdfs:subClassOf schema:Person`, no `UNION` is needed. A plain `?emp a schema:Person` already covers the full employee population. This is the direct payoff of the subclass hierarchy: the ontological relationship does the work that would otherwise require an explicit `UNION` pattern.

```sparql
-- SPARQL
SELECT ?name
WHERE {
  ?emp a schema:Person ;
       schema:name ?name .
}
```

```sql
-- SQL
SELECT name
FROM   employees
```

This reveals the advantage of Strategy B over a design where `ex:Manager` and `schema:Person` are treated as disjoint classes: retrieving all employees is no more complex than retrieving all persons. The binding choice — combined with the subclass declaration — shapes which queries are naturally concise and which require additional SPARQL patterns like `FILTER NOT EXISTS`.

---

### Strategy C — JOIN-Based Mapping: 2 tables collapse into 1 enriched concept

The domain expert decides that the physical separation of `employees` and `departments` is a database implementation detail that users should never have to see. A single binding JOINs both tables to produce fully self-contained `schema:Person` nodes with department information embedded directly.

```turtle
# Binding 1 — JOIN produces enriched schema:Person nodes
[MappingId] employee-with-dept
  source  SELECT e.id, e.name, e.role, e.salary,
                 d.name AS dept_name, d.location
          FROM employees e
          JOIN departments d ON e.dept_id = d.id
  target  ex:employee/{id} a schema:Person ;
              schema:name           {name} ;
              ex:role               {role} ;
              ex:salary             {salary} ;
              ex:departmentName     {dept_name} ;
              ex:departmentLocation {location} .
```

Result: **1 binding** from 2 tables. The FK relationship disappears entirely from the graph. A SPARQL user can now query `ex:departmentLocation` directly on a `schema:Person` without knowing that departments were ever stored in a separate table.

**What the Virtual Graph Looks Like — Strategy C**

```
  ┌───────────────────────────────────────┐
  │ ex:employee/1                         │
  │ a: schema:Person                      │
  │ schema:name          = "Alice"        │
  │ ex:role              = "Engineer"     │
  │ ex:salary            = 80000          │
  │ ex:departmentName    = "Engineering"  │  ◀─── no edge, just a plain property
  │ ex:departmentLocation= "New York"     │
  └───────────────────────────────────────┘

  ┌───────────────────────────────────────┐
  │ ex:employee/2                         │
  │ a: schema:Person                      │
  │ schema:name          = "Bob"          │
  │ ex:role              = "Manager"      │
  │ ex:salary            = 120000         │
  │ ex:departmentName    = "Engineering"  │
  │ ex:departmentLocation= "New York"     │
  └───────────────────────────────────────┘

  ┌───────────────────────────────────────┐
  │ ex:employee/3                         │
  │ a: schema:Person                      │
  │ schema:name          = "Carol"        │
  │ ex:role              = "Engineer"     │
  │ ex:salary            = 75000          │
  │ ex:departmentName    = "Marketing"    │
  │ ex:departmentLocation= "Boston"       │
  └───────────────────────────────────────┘
```

There are no edges in this graph at all. The three nodes are islands. The information that Alice and Bob belong to the same department is present (both have `ex:departmentName = "Engineering"`), but it is encoded as duplicate property values rather than as a shared node with a relationship. The `ex:Department` class does not exist anywhere in this graph. Strategy C has traded graph connectivity for query simplicity: one-hop lookups are trivial, but anything that requires reasoning about the department as a first-class entity is impossible without rewriting the binding.

**SPARQL & SQL Equivalents under Strategy C**

*Query 1 — Find all employees in a specific city.*

Under Strategy C, `ex:departmentLocation` is a direct property on `schema:Person`. Filtering on it in SPARQL requires no graph traversal — but Ontop silently executes the JOIN underneath, because the binding was defined on a JOIN source.

```sparql
-- SPARQL
SELECT ?name ?deptName
WHERE {
  ?emp a schema:Person ;
       schema:name           ?name ;
       ex:departmentName     ?deptName ;
       ex:departmentLocation "New York" .
}
```

```sql
-- SQL (what Ontop actually runs)
SELECT e.name, d.name AS dept_name
FROM   employees e
JOIN   departments d ON e.dept_id = d.id
WHERE  d.location = 'New York'
```

From the SPARQL user's perspective, this is a single-node property lookup with no traversal. The JOIN is entirely hidden inside the binding. The user does not need to know that `location` lives in a separate `departments` table.

*Query 2 — Find the average salary per department name.*

Because department name is now a direct property on `schema:Person`, aggregation across employees in the same department requires no graph hop.

```sparql
-- SPARQL
SELECT ?deptName (AVG(?salary) AS ?avgSalary)
WHERE {
  ?emp a schema:Person ;
       ex:departmentName ?deptName ;
       ex:salary         ?salary .
}
GROUP BY ?deptName
```

```sql
-- SQL
SELECT d.name AS dept_name, AVG(e.salary) AS avg_salary
FROM   employees e
JOIN   departments d ON e.dept_id = d.id
GROUP BY d.name
```

*Query 3 — What Strategy C cannot do: traversing to a Department node.*

The trade-off of collapsing the JOIN into the binding is that `ex:Department` no longer exists as a queryable node. A query that expects separate department entities will return nothing.

```sparql
-- SPARQL — returns NO results under Strategy C
SELECT ?dept ?location
WHERE {
  ?dept a ex:Department ;
        ex:location ?location .
}
```

```sql
-- The binding for ex:Department simply does not exist.
-- Ontop has no mapping to translate this pattern into SQL.
```

This is the fundamental trade-off of Strategy C: queries that treat the department as a first-class entity (e.g., "How many employees does each department have?" asked from the department's perspective) are impossible without rewriting or adding a second binding for departments.

---

### The Takeaway

| Strategy | Physical Tables | Bindings | Ontological Classes | What drives the count? |
| :--- | :---: | :---: | :---: | :--- |
| A — Direct | 2 | 2 | 2 | Table count |
| B — Role-filtered | 2 | 3 | 3 | Domain concept count |
| C — JOIN-collapsed | 2 | 1 | 1 | Query grain |

The number of bindings is determined not by the number of tables or columns in the database, but by **how many meaningful, distinct concepts exist in the domain ontology**. A domain expert, not a database administrator, owns this decision. This is precisely why semantic binding is described as domain-specific rather than structurally mechanical.

**Structural Comparison of the Three Graphs**

```
Strategy A                   Strategy B                   Strategy C
────────────────────────     ────────────────────────     ────────────────────────
Node types:    2             Node types:    3             Node types:    1
  schema:Person                schema:Person                schema:Person (only)
  ex:Department                ex:Manager
                               ex:Department

Edge types:    1             Edge types:    2             Edge types:    0
  ex:worksIn                   ex:worksIn                   (none — islands only)
                               ex:manages

Role encoded as: property    Role encoded as: node type   Role encoded as: property
Dept encoded as: node        Dept encoded as: node        Dept encoded as: property

Can query Department?  yes   Can query Department?  yes   Can query Department?  NO
Can query all staff?   yes   Can query all staff?   yes   Can query all staff?   yes
Exclude one subclass?  N/A   Exclude one subclass?  FILTER Exclude one subclass?  N/A
Can add dept props?    yes   Can add dept props?    yes   Can add dept props?    N/A
```

The binding choice is an irreversible architectural decision for the virtual graph. It determines which questions are one-line queries, which questions require additional SPARQL patterns (such as `FILTER NOT EXISTS` to exclude a subclass, or `UNION` when class hierarchies are absent), and which questions cannot be answered at all without changing the bindings.
