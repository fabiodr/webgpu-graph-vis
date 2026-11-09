# Datasets

The dataset picker reads `public/data/manifest.json` (built by `scripts/manifest.mjs`). Each entry has `id`, `group`, `name`, `description`, `source`, `nodes`, `edges`, `levels`, plus **either** `file` (a wgv-graph JSON under `public/data/`) **or** `generate` (`{ model, params, seed }`). Generate entries are built on demand in the data worker. They are deterministic and reproduce the advertised counts exactly. `DataClient.load( entry )` (`src/data/client.js`) handles both kinds.

Two groups:

- `imsv`: the eight datasets from the ImsvGraphVis paper, with their original clustering hierarchies.
- `generated`: a ladder of synthetic graphs from readable to the 20k-node target scale.

## ImsvGraphVis datasets (`public/data/imsv/`)

These were converted by `scripts/convert-imsv.mjs` from the preprocessed `Saved/Data/Graph/*.igv.json` files of [ImsvGraphVis](https://github.com/kwonoh/ImsvGraphVis) (O.-H. Kwon, C. Muelder, K. Lee, K.-L. Ma, *A Study of Layout, Rendering, and Interaction Methods for Immersive Graph Visualization*, IEEE TVCG 22(7), 2016). Each file keeps the python-louvain hierarchy used in the paper (`hierarchy.method`). The original data source is in each file's `source` field.

| id | name | nodes | edges | clusters per level | original source |
|---|---|---:|---:|---|---|
| `imsv/karate` | Zachary's karate club | 34 | 78 | 10 / 4 | W. W. Zachary, J. Anthropol. Res. 33, 452 (1977) |
| `imsv/dolphins` | Dolphins | 62 | 159 | 10 / 5 | D. Lusseau et al., Behav. Ecol. Sociobiol. 54, 396 (2003) |
| `imsv/lesmis` | Les Misérables | 77 | 254 | 12 / 6 | D. E. Knuth, The Stanford GraphBase (1993) |
| `imsv/polbooks` | Political books | 105 | 441 | 8 / 4 | V. Krebs, unpublished; via M. Newman |
| `imsv/adjnoun` | Adjectives & nouns | 112 | 425 | 10 / 4 | M. E. J. Newman, Phys. Rev. E 74, 036104 (2006) |
| `imsv/football` | College football | 115 | 613 | 12 / 8 | M. Girvan and M. E. J. Newman, PNAS 99, 7821 (2002) |
| `imsv/serengeti-foodweb` | Serengeti food web | 161 | 591 | 11 / 4 | E. B. Baskerville et al., PLoS Comput. Biol. 7, e1002321 (2011) |
| `imsv/celegansneural` | C. elegans neurons | 297 | 2,148 | 12 / 5 | D. J. Watts and S. H. Strogatz, Nature 393, 440 (1998) |

All of them sit in the readable tier: every label and every bundle can be read. They are the reference for the paper's look. Only `lesmis` has meaningful node names; the others use indices.

## Generated ladder (`public/data/generated/`)

Build it with `npm run data:generate` (`scripts/generate.mjs`). Every preset is a generator spec (`src/data/generators.js`, seed 1) clustered with the default Louvain settings (`src/data/louvain.js`: seed 1, resolution 1, leaf clusters refined to ≤ 200 nodes, near-duplicate levels collapsed). The worker uses exactly the same settings. Presets whose JSON is at most 1.5 MB are committed as files. The two target-scale graphs are stored only as specs in `generated/presets.json`; the worker generates them in about 0.5 s (generate + Louvain + prepare measured in headless Chromium).

| id | tier | model | nodes | edges | avg deg | clusters per level (finest first) | leaf size min / median / max | top Q | stored |
|---|---|---|---:|---:|---:|---|---|---:|---|
| `caveman-24` | readable | ring of cliques | 192 | 672 | 7.0 | 24 | 8 / 8 / 8 | 0.905 | file, 8 KB |
| `hsbm-500` | readable | hierarchical SBM | 500 | 2,492 | 10.0 | 25 / 9 / 7 | 7 / 17 / 34 | 0.741 | file, 27 KB |
| `lfr-2k` | overview | LFR | 2,000 | 11,367 | 11.4 | 52 / 42 | 3 / 26 / 116 | 0.758 | file, 130 KB |
| `hsbm-5k` | overview | hierarchical SBM | 5,000 | 39,817 | 15.9 | 125 / 23 | 7 / 35 / 135 | 0.845 | file, 400 KB |
| `ba-5k` | overview | Barabási–Albert | 5,000 | 19,990 | 8.0 | 1012 / 43 / 18 | 2 / 5 / 61 | 0.318 | file, 213 KB |
| `lfr-10k` | overview | LFR | 10,000 | 74,560 | 14.9 | 178 / 110 | 10 / 43 / 187 | 0.727 | file, 772 KB |
| `hsbm-20k` | stress | hierarchical SBM | 20,000 | 239,772 | 24.0 | 293 / 212 / 39 / 30 | 7 / 53 / 199 | 0.866 | spec (2.7 MB as a file) |
| `lfr-20k-dense` | stress | LFR | 20,000 | 384,444 | 38.4 | 392 / 117 | 6 / 42 / 194 | 0.676 | spec (4.1 MB as a file) |
| `lattice-12` | sanity | 3D lattice 12³ | 1,728 | 4,752 | 5.5 | 765 / 183 / 42 / 12 | 2 / 2 / 4 | 0.745 | file, 76 KB |
| `rgg-3k` | sanity | random geometric 3D | 3,000 | 14,987 | 10.0 | 264 / 58 / 23 | 2 / 9 / 47 | 0.854 | file, 162 KB |
| `ws-2k` | sanity | Watts–Strogatz | 2,000 | 8,000 | 8.0 | 234 / 90 / 38 / 29 | 6 / 8 / 15 | 0.913 | file, 113 KB |

### Recommended uses

- **Readable** (`imsv/*`, `caveman-24`, `hsbm-500`): use these for labels, the selection halo, and the paper's look. Graphs with up to ~150 nodes show every label (`labels: 'auto'`). Around 500 nodes, labels appear for the hovered or selected node, its neighbours, and high-degree nodes. Individual edges stay traceable, and bundles are thin enough to follow by eye. Small generated graphs carry pseudo-word labels, so label layout gets exercised with realistic text lengths.
- **Overview** (`lfr-2k`, `hsbm-5k`, `ba-5k`, `lfr-10k`): use these for cluster structure. Edges merge into bundles whose thickness and colour gradient show which clusters talk to each other. Labels are only useful on selection. `hsbm-5k` has the cleanest nested structure. `ba-5k` is the hard case: a preferential-attachment graph has hubs but no real communities (Q ≈ 0.32), so Louvain returns many tiny leaf clusters and bundling cannot simplify much. Use it to judge how the view degrades. `lfr-10k` checks medium-scale bundling, opacity and layout convergence.
- **Stress** (`hsbm-20k`, `lfr-20k-dense`): use these for the target scale, GPU cost and mobile limits. The edge renderer draws one instanced ribbon per edge with 8–12 segments at this size, which is roughly 4–7 million vertices per frame, each evaluating a B-spline from storage buffers. Edges dominate the frame cost. The hierarchy-based Barnes–Hut layout is cheap by comparison, at O(N log N) with leaf clusters ≤ 200 nodes. `hsbm-20k` (240k edges, 4 levels) looks good when bundled. `lfr-20k-dense` (384k edges) is the worst case for overdraw and vertex load; prefer additive blending with lower opacity there. On phones, expect to use the low quality setting at this tier. The overview tier is the comfortable range. Frame rates were not measured on real GPUs here: the headless test browser renders on the CPU.
- **Sanity** (`lattice-12`, `rgg-3k`, `ws-2k`): these graphs have known geometry, which makes layout bugs visible. A lattice should unfold into a box, a random geometric graph into a solid blob without folds, and a small-world ring into a closed loop with a few chords. Louvain hierarchies are deep here (pairs at the bottom of the lattice), which also exercises long bundling paths.

### Generators

`DataClient.listModels()` returns the models with their parameters (`key`, `label`, `type`, `min`, `max`, `step`, `value`/`default`), in the shape the UI's generator form expects. `seed` is a parameter of every model.

| model | parameters | notes |
|---|---|---|
| `hsbm` | `n`, `avgDegree`, `depth`, `branching`, `mixing` (number, or one value per level, finest first), `sizeSpread`, `degreeSpread` | Nested blocks. `mixing` is the probability that an edge leaves its block at each level. This is the best input for bundled-edge views. |
| `lfr` | `n`, `avgDegree`, `maxDegree`, `mu`, `degreeExponent`, `communityExponent`, `minCommunity`, `maxCommunity` | Power-law degrees and community sizes, configuration model inside and between communities (LFR-like). |
| `ba` | `n`, `m` | Preferential attachment. |
| `ws` | `n`, `k`, `beta` | Ring lattice with rewiring. |
| `rgg3d` | `n`, `avgDegree`, `periodic` | Points in the unit cube or 3-torus, grid hashing. |
| `lattice3d` | `nx`, `ny`, `nz`, `periodic` | Coordinates as labels. |
| `caveman` | `cliques`, `cliqueSize`, `rewire` | Connected caveman; `rewire` relaxes it. |

All generators handle 20k nodes and 300k–400k edges in about 0.05–0.25 s in Node. Louvain on 20k nodes and 400k edges takes 0.1–0.4 s. Generator output is identical for the same spec within the same JavaScript engine. The manifest counts were computed in Node and re-verified in Chromium.

To add a preset, append it to `PRESETS` in `scripts/generate.mjs` and run `npm run data:generate`. The script writes a file or a spec depending on size, and rebuilds the manifest.

## Adding your own data

**Open a file in the app** (drag and drop, or "Open file…"). It goes through `DataClient.loadFile`, and everything runs in the worker. Accepted formats:

- **wgv-graph JSON**, the canonical format (see `src/data/format.js`): `{ "format": "wgv-graph", "version": 1, "name", "nodeCount", "labels"?, "edges": [s0, t0, s1, t1, …], "hierarchy"?: { "levels": [[…finest…], …] }, "meta"? }`.
- **ImsvGraphVis** `*.igv.json`, the preprocessed files of the original project, hierarchy included.
- **Node-link JSON** (d3, or networkx `json_graph.node_link_data`): `{ "nodes": [{ "id", "label"? }], "links" | "edges": [{ "source", "target" }] }`.
- **Text edge lists** (`.txt`, `.csv`, `.tsv`, `.edges`, `.edgelist`, `.el`, `.mtx`, …): one edge per line, as `a b`, `a,b`, `a;b` or `a<TAB>b`. Ids can be any string, and quoted CSV fields are supported. Extra columns such as weights are ignored. Lines starting with `#`, `%` or `//` are comments. A header row like `source,target` is detected, and Matrix Market coordinate files work. All-integer ids keep their numeric order; any other ids keep first-appearance order. The ids become node labels.
- Any of the above gzipped (`*.gz`), decompressed in the browser.

Graphs are treated as undirected: duplicate edges, reversed duplicates and self loops are dropped. If a file has no hierarchy, the worker computes one with Louvain (seed 1, resolution 1; leaf clusters larger than 200 nodes are refined). If the file has a hierarchy, it is used unchanged. `options.cluster.mode` (`'auto'` | `'always'` | `'never'` | `'planted'`) overrides this.

**Ship a dataset with the app**: put a wgv-graph file in `public/data/<group>/<id>.json` and run `node scripts/manifest.mjs`. Precomputing the hierarchy keeps loading instant. A minimal conversion from an edge list:

```js
// node convert.mjs my-edges.csv > public/data/generated/my-graph.json
import { readFileSync } from 'node:fs';
import { parseEdgeList } from './src/data/edgelist.js';
import { ensureHierarchy } from './src/data/louvain.js';
import { serializeGraph } from './src/data/format.js';

const g = ensureHierarchy( parseEdgeList( readFileSync( process.argv[ 2 ], 'utf8' ), { name: 'My graph' } ) );
process.stdout.write( serializeGraph( g ) + '\n' );
```

Keep files under ~1.5 MB (about 100k edges). For larger graphs, prefer a `generate` spec or loading at runtime. For readable bundles, aim for a hierarchy with a branching factor of about 3–10 between levels and leaf clusters of about 10–200 nodes.
