# Datasets

The dataset picker reads `public/data/manifest.json` (built by `scripts/manifest.mjs`). Each entry has `id`, `group`, `name`, `description`, `source`, `nodes`, `edges`, `levels`, `order`, an optional `tier`, and **either** `file` (a wgv-graph JSON under `public/data/`, with its size in `bytes`) **or** `generate` (`{ model, params, seed }`). Generate entries are built on demand in the data worker. They are deterministic and reproduce the advertised counts exactly. `DataClient.load( entry )` (`src/data/client.js`) handles both kinds.

Two groups:

- `imsv`: the eight datasets from the ImsvGraphVis paper, with their original clustering hierarchies.
- `generated`: thirteen synthetic graphs, from readable to the 20,000-node target scale.

`public/data/samples/` is not in the picker. It holds one small graph in every file format the app reads (see [Sample files](#sample-files)).

## ImsvGraphVis datasets (`public/data/imsv/`)

These were converted by `scripts/convert-imsv.mjs` (`npm run data:convert`) from the preprocessed `Saved/Data/Graph/*.igv.json` files of [ImsvGraphVis](https://github.com/kwonoh/ImsvGraphVis) (O.-H. Kwon, C. Muelder, K. Lee, K.-L. Ma, *A Study of Layout, Rendering, and Interaction Methods for Immersive Graph Visualization*, IEEE TVCG 22(7), 2016). Each file keeps the python-louvain hierarchy used in the paper (`hierarchy.method`), and the app uses it unchanged. The original data source is in each file's `source` field.

| id | name | nodes | edges | clusters per level (finest first) | node labels | original source |
|---|---|---:|---:|---|---|---|
| `imsv/karate` | Zachary's karate club | 34 | 78 | 10 / 4 | indices | W. W. Zachary, J. Anthropol. Res. 33, 452 (1977) |
| `imsv/dolphins` | Dolphins | 62 | 159 | 10 / 5 | names | D. Lusseau et al., Behav. Ecol. Sociobiol. 54, 396 (2003) |
| `imsv/lesmis` | Les Misérables | 77 | 254 | 12 / 6 | names | D. E. Knuth, The Stanford GraphBase (1993) |
| `imsv/polbooks` | Political books | 105 | 441 | 8 / 4 | book titles | V. Krebs, unpublished; via M. Newman |
| `imsv/adjnoun` | Adjectives & nouns | 112 | 425 | 10 / 4 | words | M. E. J. Newman, Phys. Rev. E 74, 036104 (2006) |
| `imsv/football` | College football | 115 | 613 | 12 / 8 | college names | M. Girvan and M. E. J. Newman, PNAS 99, 7821 (2002) |
| `imsv/serengeti-foodweb` | Serengeti food web | 161 | 591 | 11 / 4 | indices | E. B. Baskerville et al., PLoS Comput. Biol. 7, e1002321 (2011) |
| `imsv/celegansneural` | C. elegans neurons | 297 | 2,148 | 12 / 5 | indices | D. J. Watts and S. H. Strogatz, Nature 393, 440 (1998) |

All of them are small enough that every label and every bundle can be read. They are the reference for the paper's look. The app counts the coarser level as "the communities" of these graphs (6 for Les Misérables) and shows the finer one as a second number. Community names, search and the tour read best on the five graphs whose nodes have names; on the other three the labels are node indices.

## Generated graphs (`public/data/generated/`)

Build them with `npm run data:generate` (`scripts/generate.mjs`; `--only id,id` builds a subset, `--dry` only prints the table). Every preset is a generator spec (`src/data/generators.js`, seed 1) clustered with the default settings of the clustering front end (`src/data/hierarchy.js`): Leiden, seed 1, resolution 1, leaf clusters of at most 200 nodes, and the "ladder" that adds grouping or splitting levels until the hierarchy has two to four levels with a moderate branching. The data worker uses exactly the same settings (`test/data-pipeline.test.js` compares the two). Presets whose JSON is at most 1.5 MB are committed as files. The two 20,000-node graphs are stored only as specs in `generated/presets.json` and are generated in the worker when chosen.

| id | tier | model | nodes | edges | avg deg | clusters per level (finest first) | communities | modularity | leaf size min / median / max | stored |
|---|---|---|---:|---:|---:|---|---:|---:|---|---|
| `caveman-24` | readable | ring of cliques | 192 | 672 | 7.0 | 24 / 4 | 24 | 0.905 | 8 / 8 / 8 | file, 8 KB |
| `hsbm-500` | readable | hierarchical SBM | 500 | 2,492 | 10.0 | 25 / 7 | 7 | 0.741 | 9 / 17 / 34 | file, 26 KB |
| `lfr-2k` | overview | LFR | 2,000 | 11,367 | 11.4 | 48 / 7 | 48 | 0.758 | 16 / 28 / 119 | file, 128 KB |
| `hsbm-5k` | overview | hierarchical SBM | 5,000 | 39,817 | 15.9 | 126 / 26 / 5 | 26 | 0.841 | 7 / 37 / 133 | file, 409 KB |
| `galaxy-3k` | overview | hubs and satellites | 3,000 | 3,472 | 2.3 | 38 / 6 | 38 | 0.949 | 37 / 71 / 193 | file, 46 KB |
| `ba-5k` | overview | Barabási–Albert | 5,000 | 19,990 | 8.0 | 37 / 16 | 16 | 0.349 | 61 / 131 / 190 | file, 196 KB |
| `lfr-10k` | overview | LFR | 10,000 | 74,560 | 14.9 | 166 / 25 / 6 | 166 | 0.701 | 5 / 47 / 187 | file, 786 KB |
| `hsbm-20k` | stress | hierarchical SBM | 20,000 | 239,772 | 24.0 | 229 / 36 / 6 | 36 | 0.865 | 11 / 83 / 199 | spec (2.6 MB as a file) |
| `lfr-20k-dense` | stress | LFR | 20,000 | 384,444 | 38.4 | 189 / 134 / 27 / 4 | 134 | 0.676 | 30 / 109 / 199 | spec (4.2 MB as a file) |
| `lattice-12` | sanity | 3D lattice 12³ | 1,728 | 4,752 | 5.5 | 83 / 13 | 13 | 0.773 | 8 / 18 / 100 | file, 64 KB |
| `rgg-3k` | sanity | random geometric 3D | 3,000 | 14,987 | 10.0 | 274 / 181 / 24 / 6 | 24 | 0.859 | 2 / 10 / 29 | file, 170 KB |
| `mesh-torus` | sanity | torus mesh 60 × 30 | 1,800 | 5,400 | 6.0 | 110 / 17 / 3 | 17 | 0.815 | 8 / 16 / 27 | file, 73 KB |
| `ws-2k` | sanity | Watts–Strogatz | 2,000 | 8,000 | 8.0 | 220 / 175 / 28 / 8 | 28 | 0.915 | 3 / 8 / 14 | file, 112 KB |

"Communities" is the number the app shows (panel, legend, insights drawer, tour): the cluster count of the level that carries the detected communities. The other levels are finer splits or coarser groupings around it. "Modularity" is that level's modularity at resolution 1.

### Recommended uses

- **Readable** (`imsv/*`, `caveman-24`, `hsbm-500`): use these for labels, the selection, the node card and the paper's look. Individual edges stay traceable, and bundles are thin enough to follow by eye. Small generated graphs carry pseudo-word labels, so search and label layout work with realistic text lengths.
- **Overview** (`lfr-2k`, `hsbm-5k`, `galaxy-3k`, `ba-5k`, `lfr-10k`): use these for community structure. Edges merge into bundles whose thickness and colour gradient show which communities are connected. At overview distance the picture carries community names; node labels appear as the camera comes close and on selection. `hsbm-5k` has the cleanest nested structure. `galaxy-3k` is nearly a tree with strong hubs and a clear two-level hierarchy. `ba-5k` is the hard case: a preferential-attachment graph has hubs but no real communities (modularity 0.35), so bundling cannot simplify much. Use it to judge how the view degrades.
- **Stress** (`hsbm-20k`, `lfr-20k-dense`): use these for the target scale, GPU cost and phone limits. At this size the edge renderer draws every edge as a strip of 8 to 10 cached curve segments (`hsbm-20k`: 10 segments, 5.3 million strand vertices per frame, a 42 MB curve cache), and edges dominate the frame cost. Above 150,000 edges the automatic edge blend is additive. Both presets load without errors; their frame rates on real GPUs were not measured, because the development browser renders on the CPU.
- **Sanity** (`lattice-12`, `rgg-3k`, `mesh-torus`, `ws-2k`): these graphs have a known shape, which makes layout errors visible. A lattice should unfold into a box, a random geometric graph into a solid blob without folds, the torus mesh into a torus, and a small-world ring into a closed loop with a few chords. The stress layout is the mode that recovers these shapes best.

### Generators

`DataClient.listModels()` returns the models with their parameters (`key`, `label`, `type`, `min`, `max`, `step`, `value`), in the shape the "Generate…" form of the panel expects. `seed` is a parameter of every model. The ranges of the form go up to 30,000 nodes for the random models.

| model | parameters | notes |
|---|---|---|
| `hsbm` | `n`, `avgDegree`, `depth`, `branching`, `mixing` (number, or one value per level, finest first), `sizeSpread`, `degreeSpread` | Nested blocks. `mixing` is the probability that an edge leaves its block at each level. This is the best input for bundled-edge views. |
| `lfr` | `n`, `avgDegree`, `maxDegree`, `mu`, `degreeExponent`, `communityExponent`, `minCommunity`, `maxCommunity` | Power-law degrees and community sizes, configuration model inside and between communities (LFR-like). |
| `ba` | `n`, `m` | Preferential attachment: hubs, weak communities. |
| `ws` | `n`, `k`, `beta` | Ring lattice with rewiring. |
| `rgg3d` | `n`, `avgDegree`, `periodic` | Points in the unit cube or 3-torus, linked when close. |
| `lattice3d` | `nx`, `ny`, `nz`, `periodic` | Regular grid; coordinates as labels. |
| `caveman` | `cliques`, `cliqueSize`, `rewire` | Connected caveman; `rewire` relaxes it. |
| `galaxy` | `n`, `groups`, `systems`, `sizeSpread`, `moons`, `mesh`, `bridges` | Groups of systems: a hub with satellites, some satellites with moons; the hubs of a group are linked, groups meet at their largest hubs. |
| `geosf` | `n`, `avgDegree`, `exponent`, `dimension`, `periodic` | Geometric inhomogeneous random graph: points with power-law weights, linked when close relative to their weights. Hubs and locality at once. |
| `mesh` | `nu`, `nv`, `wrapU`, `wrapV`, `triangles` | Triangle or quad mesh of a sheet, a tube or a torus. |

Generator output is identical for the same spec within the same JavaScript engine. In Node on the development machine, generating `hsbm-20k` takes about 0.1 s and clustering it about 0.3 s; `lfr-20k-dense` about 0.2 s and 0.3 s.

To add a preset, append it to `PRESETS` in `scripts/generate.mjs` and run `npm run data:generate`. The script writes a file or a spec depending on size, and rebuilds the manifest.

## Opening your own data

Drop a file on the window or use "Open file…". It goes through `DataClient.loadFile`, and reading, parsing, clustering and preparation run in the worker; nothing is uploaded.

### Formats

`src/data/formats.js` is the registry (`data.formats()`); the parsers are `parse-xml.js`, `parse-text.js`, `parse-json.js`, `edgelist.js` and `format.js`.

| Format | Extensions | What is read |
|---|---|---|
| GraphML | `.graphml`, `.xml` | Including yEd files: node attributes, edge weights, nested graphs as groups. |
| GEXF (Gephi) | `.gexf` | GEXF 1.1 to 1.3: attributes, weights, `modularity_class` and parent nodes as groups. |
| GML | `.gml` | Node attributes, `value` / `weight` on edges. |
| Graphviz DOT | `.dot`, `.gv` | The common subset of DOT; subgraphs become groups. |
| Pajek | `.net`, `.paj` | Vertices, arcs, edges, lists, matrix, partitions. |
| Matrix Market | `.mtx` | Sparse or dense adjacency matrix; a rectangular matrix is read as a bipartite graph. |
| Edge list | `.csv`, `.tsv`, `.txt`, `.edges`, `.edgelist`, `.el`, `.ssv`, `.tab`, `.ncol`, `.dat` | One edge per line, see below. |
| JSON | `.json` | wgv-graph (below), ImsvGraphVis `*.igv.json`, d3 / networkx node-link, Cytoscape.js, graphology / sigma, adjacency lists, arrays of edges. |

Any of them may be gzipped (`*.gz`); it is decompressed in the browser.

**Detection** looks at the content first and at the file name second, because dropped files are often misnamed: a leading `<` is GraphML or GEXF by its root element, a leading `{` or `[` is JSON, `%%MatrixMarket`, `*Vertices`, `graph [` and `(di)graph … {` name their formats, and anything else is the format of the extension, or an edge list. Binary files, PDFs, UTF-16 text and text that is mostly not UTF-8 are refused with a message that says why, and the graph on screen stays. A drop is filtered by extension; "Open file…" takes any file and lets the content decide.

**Edge lists**: one edge per line, as `a b`, `a,b`, `a;b` or `a<TAB>b`; a line with a single id is an isolated node. Ids can be any string, and quoted CSV fields are supported. Lines starting with `#`, `%` or `//` are comments. A header row like `source,target` is detected. A weight column is taken from a header named `weight`, `value`, `count`, `strength`, `w` or `freq`, or, without a header, from a numeric third column. All-integer ids keep their numeric order; any other ids keep first-appearance order. The ids become node labels.

**wgv-graph JSON** is the canonical format (see the header of `src/data/format.js`):

```json
{
  "format": "wgv-graph", "version": 1,
  "name": "My graph",
  "nodeCount": 4,
  "labels": ["a", "b", "c", "d"],
  "edges": [0, 1, 1, 2, 2, 3],
  "weights": [1, 2, 1],
  "attributes": {
    "age":  { "type": "number", "values": [31, null, 7, 12] },
    "kind": { "type": "category", "values": [0, 1, -1, 0], "categories": ["x", "y"] }
  },
  "hierarchy": { "method": "leiden", "levels": [[0, 0, 1, 1]] }
}
```

`labels`, `weights`, `attributes`, `hierarchy`, `meta` and `positions` are optional. `hierarchy.levels[0]` is the finest partition; `levels[l][i]` is the cluster of node `i` at level `l`, and the levels are nested.

### What happens to a file

- Graphs are treated as undirected: duplicate edges and reversed duplicates are merged (weights summed) and self loops are dropped.
- **Node attributes** (numeric or categorical, at most 32) appear as extra metrics in the lens and on the node card. **Edge weights** are used by the community detection; the layout and the drawing do not use them.
- **Hierarchy.** A complete hierarchy in the file (wgv-graph, `.igv.json`) is used unchanged. Groups in the file (a group attribute, subgraphs, parent nodes) become the top of the hierarchy, with computed levels below. A file without either gets its hierarchy from Leiden as described above. `options.cluster` of `DataClient` (`{ mode: 'auto' | 'always' | 'never' | 'planted', method: 'leiden' | 'louvain', resolution, … }`) overrides this; in the app, the Communities section of the panel re-clusters the graph on screen.
- **Limits**: 250,000 nodes, 1,000,000 edges, 256 MB per file (packed and unpacked). A file beyond them is refused with a message.

### Sample files

`public/data/samples/` holds Les Misérables (77 nodes, 254 edges) as `lesmis.graphml`, `.gexf`, `.gml`, `.dot`, `.net`, `.mtx`, `.csv`, `.tsv`, `.node-link.json`, `.cytoscape.json`, `.graphology.json` and `.adjacency.json`. As far as each format can express them, the files carry the names, a category attribute `community`, a numeric attribute `degree` and edge weights (1 + the number of common neighbours, a quantity derived from the edges, not the original co-appearance counts). They are written by `node scripts/generate.mjs --samples` and are used by the format tests, by `dev/data.html` and for trying drag and drop.

### Shipping a dataset with the app

Put a wgv-graph file in `public/data/<group>/<id>.json` and run `node scripts/manifest.mjs`. Precomputing the hierarchy keeps loading fast. A conversion from any supported text format, run from the repository root:

```js
// node convert.mjs my-graph.graphml > public/data/generated/my-graph.json
import { readFileSync } from 'node:fs';
import { parseGraph } from './src/data/formats.js';
import { applyClustering } from './src/data/hierarchy.js';
import { serializeGraph } from './src/data/format.js';

const file = process.argv[ 2 ];
const { graph } = parseGraph( readFileSync( file, 'utf8' ), file );
const g = applyClustering( { ...graph, name: 'My graph' } );
delete g.clustering; // derived again from the stored hierarchy when the file is loaded
process.stdout.write( serializeGraph( g ) + '\n' );
```

Keep files under about 1.5 MB (roughly 100,000 edges). For larger graphs, prefer a `generate` spec or loading at run time. A folder other than `imsv` and `generated` becomes a group of its own; add it to `GROUP_ORDER` and `groups` in `scripts/manifest.mjs` to give it a place and a title in the picker.
