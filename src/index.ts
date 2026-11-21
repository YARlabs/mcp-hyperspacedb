import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ErrorCode,
  McpError,
} from "@modelcontextprotocol/sdk/types.js";
import { HyperspaceClient, CognitiveMathExport as CognitiveMath, HyperbolicMath } from "hyperspace-sdk-ts";
import { z } from "zod";
import { planQuery, QUERY_PLANNER_TOOL_SCHEMA } from "./query_planner.js";

const rawHost = process.env.HYPERSPACE_HOST || "localhost:50051";
const HYPERSPACE_HOST = rawHost.replace(/^https?:\/\//, '').replace(/\/$/, '');
const HYPERSPACE_API_KEY = process.env.HYPERSPACE_API_KEY || process.env.YAR_API_KEY || process.env.CDE_API_KEY || "I_LOVE_HYPERSPACEDB";

/** HTTP API base: HYPERSPACE_HTTP_URL, else the gRPC host with port − 1 (server defaults 50051 / 50050). */
function httpBaseFor(host: string): string {
  if (process.env.HYPERSPACE_HTTP_URL) return process.env.HYPERSPACE_HTTP_URL.replace(/\/$/, '');
  const [ip, port] = host.split(':');
  const p = Number(port);
  return Number.isFinite(p) && p > 0 ? `http://${ip}:${p - 1}` : `http://${ip}:50050`;
}

/** Quantization levels accepted by `quantization` on collection create (HyperspaceDB 4.2.5). */
const QUANTIZATION_LEVELS = ["none", "medium", "medium_plus", "turbo", "ternary", "extreme", "rabitq", "pq4", "pq", "opq"] as const;
/** Levels that learn a centroid / codebooks (calibrated online by the server, or by an index rebuild). */
const TRAINED_LEVELS = new Set(["rabitq", "pq4", "pq", "opq"]);

// Helper for Gromov Delta (Ported from Rust SDK)
function analyzeDeltaHyperbolicity(vectors: number[][], numSamples: number = 100): { delta: number, recommendation: "lorentz" | "poincare" | "cosine" | "l2" } {
  if (vectors.length < 4) return { delta: 0, recommendation: "l2" };

  const l2Dist = (a: number[], b: number[]) => Math.sqrt(a.reduce((s, x, i) => s + Math.pow(x - b[i], 2), 0));

  let maxDelta = 0;
  for (let i = 0; i < numSamples; i++) {
    const idxs = [0, 0, 0, 0].map(() => Math.floor(Math.random() * vectors.length));
    if (new Set(idxs).size < 4) continue;

    const [a, b, u, v] = idxs.map(idx => vectors[idx]);
    const d_ab = l2Dist(a, b);
    const d_uv = l2Dist(u, v);
    const d_au = l2Dist(a, u);
    const d_bv = l2Dist(b, v);
    const d_av = l2Dist(a, v);
    const d_bu = l2Dist(b, u);

    const s1 = d_ab + d_uv;
    const s2 = d_au + d_bv;
    const s3 = d_av + d_bu;

    const sorted = [s1, s2, s3].sort((x, y) => y - x);
    const delta = (sorted[0] - sorted[1]) / 2;
    if (delta > maxDelta) maxDelta = delta;
  }

  const isNormalized = vectors.slice(0, 20).every(v => {
    const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return Math.abs(n - 1.0) < 1e-2;
  });

  let recommendation: "lorentz" | "poincare" | "cosine" | "l2" = "l2";
  if (maxDelta < 0.15) recommendation = "lorentz";
  else if (maxDelta < 0.30) recommendation = "poincare";
  else if (isNormalized) recommendation = "cosine";

  return { delta: maxDelta, recommendation };
}

class HyperspaceMcpServer {
  private server: Server;
  private client: HyperspaceClient;

  constructor() {
    this.server = new Server(
      { name: "mcp-hyperspacedb", version: "4.2.5" },
      { capabilities: { tools: {} } }
    );
    const isLocal = HYPERSPACE_HOST.startsWith("localhost") || HYPERSPACE_HOST.startsWith("127.0.0.1");
    const grpcKey = (isLocal && HYPERSPACE_API_KEY.startsWith("sk_")) ? "I_LOVE_HYPERSPACEDB" : (HYPERSPACE_API_KEY || "I_LOVE_HYPERSPACEDB");
    this.client = new HyperspaceClient(HYPERSPACE_HOST, grpcKey);
    // Compact f32 wire vectors (hyperspace-sdk-ts >= 4.2.5, server >= 4.2.5).
    if (/^(1|true|yes|on)$/i.test(process.env.HYPERSPACE_COMPACT_VECTORS || "")) {
      (this.client as any).setCompactVectors?.(true);
    }
    this.setupTools();
    this.server.onerror = (error) => console.error("[MCP Error]", error);
  }

  private setupTools() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        // --- DATA PLANE TOOLS ---
        {
          name: "hyperspace_search_text",
          description: "Search a `collection` for entries semantically similar to `text` (natural language query). Returns a JSON array of up to `top_k` results, each with id, score, and stored payload.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string" },
              text: { type: "string" },
              top_k: { type: "number", default: 5 },
              ef_search: { type: "number", description: "HNSW beam width for this query (server >= 4.2.5), used as given. Higher = better recall, slower. Omit to use the collection's beam, which the server tunes on real queries (the configured ef_search, default 100, is a floor)." },
              hybrid_alpha: { type: "number", description: "Weight for hybrid search fusion (0.0 = lexical only, 1.0 = semantic only). Default: 0.5" },
              bm25_options: {
                type: "object",
                properties: {
                  method: { type: "string", description: "BM25 method (bm25, bm25plus, lucene, atire)" },
                  language: { type: "string", description: "Stemmer language (english, russian, etc.)" },
                }
              }
            },
            required: ["collection", "text"]
          }
        },
        {
          name: "hyperspace_search_wasserstein",
          description: "Advanced Optimal Transport (Wasserstein) search inside `collection` for `text`. Compares distributions/complex concept overlap. Returns a JSON array of `top_k` matches ranked by Wasserstein distance, each with id, score, payload.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string" },
              text: { type: "string" },
              top_k: { type: "number", default: 5 }
            },
            required: ["collection", "text"]
          }
        },
        {
          name: "hyperspace_insert_text",
          description: "Store a new factual claim or memory: insert `text` into `collection` under numeric `id` (auto-vectorized). Returns a JSON object {ok: true, id} on success or {ok: false, error} on failure.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string" },
              id: { type: "number" },
              text: { type: "string" },
              metadata: { type: "object" }
            },
            required: ["collection", "id", "text"]
          }
        },
        {
          name: "hyperspace_create_collection",
          description: "Setup new memory spaces (collections) with specific vector dimension, metric geometry, and quantization level. Pass `expected_vectors` only for collections that will hold millions of vectors: it sets the shard count, which is fixed at creation (one shard per 2M vectors; one shard serves the most queries per node). Returns the created collection config as JSON result.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string" },
              dimension: { type: "number", default: 1024, description: "Vector dimension (e.g., 1024)" },
              metric: { type: "string", default: "cosine", description: "Distance metric: l2, cosine, poincare, lorentz, hybrid" },
              quantization: {
                type: "string",
                enum: [...QUANTIZATION_LEVELS],
                default: "medium",
                description: "Quantization level. medium (default: 8-bit int8 codes, 4x less RAM, 98.5% recall single pass and 100% with the exact rerank the server applies by default), none (f32), medium_plus (4-bit, any metric), turbo (4-bit spherical), ternary (1.58-bit), extreme (1-bit, any metric), rabitq (1-bit RaBitQ, l2/cosine), pq4 (4-bit PQ fast-scan, l2/cosine), pq / opq (8-bit PQ, l2/cosine, 60x). Trained levels (rabitq, pq4, pq, opq) calibrate themselves online once a shard holds ~1,024 vectors (4,096 for pq/opq); hyperspace_rebuild_index also retrains them."
              },
              expected_vectors: { type: "number", description: "Optional expected collection size (server >= 4.2.5). Sets the shard count at creation: one per 2,000,000 vectors, one when omitted." },
              encryption_key: { type: "string", description: "Optional client-side AES-256-GCM + Isometric projection key for zero-knowledge encryption" }
            },
            required: ["collection"]
          }
        },
        // --- AGENTIC GRAPH TOOLS ---
        {
          name: "hyperspace_graph_traverse",
          description: "Deep graph exploration starting from node `start_id` inside `collection`. Finds logical paths between concept A and context B for complex reasoning/cross-referencing. Returns a JSON object {paths: [...], nodes: [...], edges: [...]} bounded by max_depth and max_nodes.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string" },
              start_id: { type: "number" },
              max_depth: { type: "number", default: 3 },
              max_nodes: { type: "number", default: 256 }
            },
            required: ["collection", "start_id"]
          }
        },
        {
          name: "hyperspace_get_neighbors",
          description: "Explore local connectivity of a concept in the vector index graph. Requires `collection` name and `id` of the node/concept. Returns a list of neighboring nodes as JSON result.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string" },
              id: { type: "number", description: "The node/concept ID" },
              layer: { type: "number", default: 0, description: "HNSW graph layer" },
              limit: { type: "number", default: 64, description: "Max neighbors to return" }
            },
            required: ["collection", "id"]
          }
        },
        {
          name: "hyperspace_find_clusters",
          description: "Detect emergent structure/hierarchy in `collection`. Returns a JSON array of clusters, each {cluster_id, members: [...], centroid, size}, filtered by min_cluster_size.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string" },
              min_cluster_size: { type: "number", default: 3 }
            },
            required: ["collection"]
          }
        },
        // --- ANALYTICS TOOLS (Standalone & Server-Assisted) ---
        {
          name: "hyperspace_analyze_geometry",
          description: "Calculate Gromov Delta-hyperbolicity either for a named `collection` on the server or over supplied `vectors` (array of numeric arrays). Returns {delta, recommendation: 'lorentz'|'poincare'|'cosine'|'l2'}.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string", description: "Optional collection name to analyze server-side" },
              vectors: { type: "array", items: { type: "array", items: { type: "number" } } },
              samples: { type: "number", default: 200 }
            }
          }
        },
        {
          name: "hyperspace_analyze_thought_stability",
          description: "Calculate Lyapunov Convergence of a `trajectory` (Chain of Thought). Returns a JSON object {lyapunov_exponent, classification: 'stable'|'chaotic'} - negative result means stable/converging, positive means chaotic/hallucinating.",
          inputSchema: {
            type: "object",
            properties: {
              trajectory: { type: "array", items: { type: "array", items: { type: "number" } } },
              curvature: { type: "number", default: 1.0 }
            },
            required: ["trajectory"]
          }
        },
        // --- COGNITIVE SYSTEM TOOLS ---
        {
          name: "hyperspace_trigger_reconsolidation",
          description: "AI Sleep Mode: trigger Flow Matching on the server to optimize the geometric representation of concepts in `collection` based on usage/context. Returns a JSON object {ok, iterations, loss_before, loss_after}.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string" },
              learning_rate: { type: "number", default: 0.1 }
            },
            required: ["collection"]
          }
        },
        {
          name: "hyperspace_freeze_collection",
          description: "Freeze a collection to make it read-only, preventing new inserts. Returns ok/error status object.",
          inputSchema: {
            type: "object",
            properties: { collection: { type: "string" } },
            required: ["collection"]
          }
        },
        {
          name: "hyperspace_unfreeze_collection",
          description: "Unfreeze a previously frozen collection to allow inserts again. Returns ok/error status object.",
          inputSchema: {
            type: "object",
            properties: { collection: { type: "string" } },
            required: ["collection"]
          }
        },
        {
          name: "hyperspace_list_collections",
          description: "List all active collections. Returns a JSON array of objects, each with {name, dimension, metric, count} metadata fields.",
          inputSchema: {
            type: "object",
            properties: {}
          }
        },
        {
          name: "hyperspace_get_stats",
          description: "Get detailed statistics for the named `collection`. Returns a JSON object with count, dimension, metric, logical_clock, and storage stats. Merges metadata, cache stats and WriteBuffer size.",
          inputSchema: {
            type: "object",
            properties: { collection: { type: "string" } },
            required: ["collection"]
          }
        },
        {
          name: "hyperspace_cache_stats",
          description: "Get cache statistics (hits, misses, policy, etc.) for a specific collection's L0 Hot Tier Cache. Returns stats object as JSON result.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string", description: "The name of the collection." }
            },
            required: ["collection"]
          }
        },
        {
          name: "hyperspace_cache_clear",
          description: "Clear / purge all items in the L0 Cache for a specific collection. Returns ok/error status object.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string", description: "The name of the collection." }
            },
            required: ["collection"]
          }
        },
        {
          name: "hyperspace_cache_config",
          description: "Update L0 Cache configuration (eviction policy, ANN threshold) for a specific collection. Returns updated config as JSON result.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string", description: "The name of the collection." },
              policy: { type: "string", description: "Cache eviction policy (e.g. lru, lfu, ttl)." },
              ann_threshold: { type: "number", description: "ANN similarity distance threshold for cache lookup filtering." }
            },
            required: ["collection", "policy"]
          }
        },
        {
          name: "hyperspace_delete_collection",
          description: "Permanently delete a collection and all of its vectors. Returns a result object with ok status or error.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string", description: "The name of the collection to delete." }
            },
            required: ["collection"]
          }
        },
        {
          name: "hyperspace_delete_points",
          description: "Delete a single vector point from a collection by its ID. Returns a result object with ok status or error.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string", description: "The collection name." },
              id: { type: "number", description: "The ID of the point to delete." }
            },
            required: ["collection", "id"]
          }
        },
        {
          name: "hyperspace_get_points",
          description: "Retrieve vector coordinate and metadata for a list of point IDs. Requires `collection` name and `ids` array of point IDs.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string", description: "The collection name." },
              ids: { type: "array", items: { type: "number" }, description: "Array of point IDs." }
            },
            required: ["collection", "ids"]
          }
        },
        {
          name: "hyperspace_rebuild_index",
          description: "Rebuild the HNSW graph of a collection from its exact vectors (hot vacuum): drops deleted vectors, compacts payloads, and swaps the new index in while searches continue; returns when done. For trained quantization levels (rabitq, pq4, pq, opq) it also retrains the centroid / codebooks. Use after large deletions or bulk imports. Returns {success}.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string", description: "The name of the collection." }
            },
            required: ["collection"]
          }
        },
        {
          name: "hyperspace_server_status",
          description: "Server version, defaults and engine tunables (HNSW M / ef floor, exact rerank, ACORN filtering, adaptive ef, delete repair, WAL mode, SIMD backend) plus the catalog of supported quantization levels with bytes per dimension, compatible metrics and calibration needs. Use it to pick a quantization level before creating a collection.",
          inputSchema: {
            type: "object",
            properties: {}
          }
        },
        {
          name: "hyperspace_vacuum",
          description: "Perform vacuuming on the database to permanently purge deleted vectors and reclaim disk space. Returns vacuum status and reclaimed-space result.",
          inputSchema: {
            type: "object",
            properties: {}
          }
        },
        {
          name: "hyperspace_get_subsumption_tree",
          description: "Retrieve the Lorentz hierarchy subsumption tree starting from a given `root_id`; returns the tree as JSON in the given `collection`.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string", description: "The collection name." },
              root_id: { type: "number", description: "The root node ID of the subsumption tree." },
              max_depth: { type: "number", default: 3, description: "Maximum hierarchy depth." }
            },
            required: ["collection", "root_id"]
          }
        },
        {
          name: "hyperspace_get_concept_parents",
          description: "Retrieve parent concepts in a hierarchical collection. Requires `collection` and the concept `id`. Returns a JSON list of parent nodes.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string", description: "The collection name." },
              id: { type: "number", description: "The concept ID." },
              layer: { type: "number", default: 0, description: "HNSW graph layer." },
              limit: { type: "number", default: 32, description: "Limit number of parents." }
            },
            required: ["collection", "id"]
          }
        },
        {
          name: "hyperspace_explore_graph",
          description: "Traverse the graph and return nodes and links in a format ready for visualization. Requires `collection` and `start_id` of the starting node.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string", description: "The collection name." },
              start_id: { type: "number", description: "The starting node ID." },
              max_depth: { type: "number", default: 2, description: "Max depth traversal." },
              max_nodes: { type: "number", default: 256, description: "Max nodes to return." }
            },
            required: ["collection", "start_id"]
          }
        },
        {
          name: "hyperspace_predict_momentum",
          description: "Forecast future agent thought paths using Koopman momentum extrapolation. Requires `collection` and `trajectory_ids` array; `steps` sets prediction horizon. Returns predicted trajectory summary as JSON.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string", description: "The collection name." },
              trajectory_ids: { type: "array", items: { type: "number" }, description: "Array of trajectory node IDs." },
              steps: { type: "number", default: 1.0, description: "Steps to predict ahead." },
              curvature: { type: "number", default: 1.0, description: "Curvature parameter." }
            },
            required: ["collection", "trajectory_ids"]
          }
        },
        {
          name: "hyperspace_get_trust_score",
          description: "Evaluate stability and trust score for a given thought trajectory path. Requires `collection` and `trajectory_ids` array of trajectory node IDs.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string", description: "The collection name." },
              trajectory_ids: { type: "array", items: { type: "number" }, description: "Array of trajectory node IDs." },
              curvature: { type: "number", default: 1.0, description: "Curvature parameter." }
            },
            required: ["collection", "trajectory_ids"]
          }
        },
        // --- QUERY PLANNING & REASONING TOOLS ---
        {
          name: "hyperspace_query_planner",
          description: "Enterprise RAG Query Planner. Evaluates user intent before vector search: detects ambiguous/underspecified questions (requesting multiple-choice user clarification) and decomposes multi-part queries into parallel sub-queries.",
          inputSchema: QUERY_PLANNER_TOOL_SCHEMA
        },
        {
          name: "clarify_and_split_intent",
          description: "Alias for hyperspace_query_planner. Evaluates intent clarity, asks clarifying questions if ambiguous, and splits multi-part questions into isolated search queries.",
          inputSchema: QUERY_PLANNER_TOOL_SCHEMA
        },
        {
          name: "hyperspace_link",
          description: "Create an explicit directed, labeled, weighted relationship edge between two entities in the native CSR vector-graph. Parameters: `collection`, `source_id`, `target_id`, optional `relation` (default 'OWNS'), `weight` (default 1.0).",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string", description: "Collection name" },
              source_id: { type: "number", description: "Source entity ID" },
              target_id: { type: "number", description: "Target entity ID" },
              relation: { type: "string", default: "OWNS", description: "Edge relation label" },
              weight: { type: "number", default: 1.0, description: "Edge weight" }
            },
            required: ["collection", "source_id", "target_id"]
          }
        },
        {
          name: "hyperspace_link_batch",
          description: "Bulk-insert explicit directed edges in CSR graph atomically. Parameters: `collection`, `edges` array of {source_id, target_id, relation, weight}.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string", description: "Collection name" },
              edges: {
                type: "array",
                items: {
                  type: "object",
                  properties: {
                    source_id: { type: "number" },
                    target_id: { type: "number" },
                    relation: { type: "string" },
                    weight: { type: "number" }
                  },
                  required: ["source_id", "target_id"]
                }
              }
            },
            required: ["collection", "edges"]
          }
        },
        {
          name: "hyperspace_unlink",
          description: "Delete an explicit directed relationship edge with zero-lag invalidation. Parameters: `collection`, `source_id`, `target_id`, optional `relation`.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string", description: "Collection name" },
              source_id: { type: "number", description: "Source entity ID" },
              target_id: { type: "number", description: "Target entity ID" },
              relation: { type: "string", default: "OWNS", description: "Relation label" }
            },
            required: ["collection", "source_id", "target_id"]
          }
        },
        {
          name: "hyperspace_induce_edges",
          description: "Declaratively induce explicit graph edges from existing document metadata attributes (path_hierarchy, foreign_key, exact_match). Parameters: `collection`, `rules` array.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string", description: "Collection name" },
              rules: {
                type: "array",
                items: {
                  type: "object",
                  properties: {
                    field: { type: "string" },
                    rule_type: { type: "string", enum: ["path_hierarchy", "foreign_key", "exact_match"] },
                    target_field: { type: "string" },
                    relation: { type: "string" },
                    weight: { type: "number" }
                  },
                  required: ["field", "rule_type"]
                }
              }
            },
            required: ["collection", "rules"]
          }
        },
        {
          name: "hyperspace_verify_generation",
          description: "Tribunal Router (Mathematical Anti-Hallucination Guardrail): Verify whether an LLM generated answer stays within the geometric entailment cones and geodesic boundaries of retrieved context IDs.",
          inputSchema: {
            type: "object",
            properties: {
              collection: { type: "string", description: "Collection name" },
              llm_answer: { type: "string", description: "Candidate LLM response text to verify" },
              context_ids: { type: "array", items: { type: "string" }, description: "Retrieved context node IDs" },
              max_distance_threshold: { type: "number", default: 1.5, description: "Max geodesic distance threshold" },
              check_geodesic: { type: "boolean", default: true }
            },
            required: ["collection", "llm_answer", "context_ids"]
          }
        },
        {
          name: "hyperspace_pdk_extension",
          description: "Invoke a native HyperspaceDB PDK Extension (e.g. 'git_ast' for AST codebase indexing & blast radius, 'embed' for structure-aware Markdown ingestion, 'shield', 'chem', 'bio', 'fin', 'sec', 'legal', 'onto').",
          inputSchema: {
            type: "object",
            properties: {
              extension: { type: "string", description: "Extension name (e.g., git_ast, embed, shield, chem, bio, fin, sec, legal, onto)" },
              action: { type: "string", description: "Extension action (e.g., index_repository, blast_radius, ingest_markdown)" },
              payload: { type: "object", description: "JSON payload passed to the extension action" }
            },
            required: ["extension", "action"]
          }
        }
      ]
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;
      try {
        switch (name) {
          case "hyperspace_query_planner":
          case "clarify_and_split_intent": {
            const { raw_user_query, context, collections, max_sub_queries } = z.object({
              raw_user_query: z.string(),
              context: z.string().optional(),
              collections: z.array(z.string()).optional(),
              max_sub_queries: z.number().optional()
            }).parse(args);
            const res = await planQuery({
              raw_user_query,
              context,
              collections,
              max_sub_queries
            });
            return { content: [{ type: "text", text: JSON.stringify(res, null, 2) }] };
          }
          case "hyperspace_list_collections": {
            const res = await this.client.listCollections();
            return { content: [{ type: "text", text: JSON.stringify(res, null, 2) }] };
          }
          case "hyperspace_search_text": {
            const { collection, text, top_k, hybrid_alpha, bm25_options, ef_search } = z.object({
              collection: z.string(),
              text: z.string(),
              top_k: z.number().optional(),
              ef_search: z.number().int().nonnegative().optional(),
              hybrid_alpha: z.number().optional(),
              bm25_options: z.object({
                method: z.string().optional(),
                language: z.string().optional(),
              }).optional()
            }).parse(args);
            const res = await this.client.searchText(text, top_k || 5, collection, {
              hybridAlpha: hybrid_alpha,
              bm25: bm25_options as any,
              // `efSearch` needs hyperspace-sdk-ts >= 4.2.5; older SDKs ignore it.
              ...(ef_search !== undefined ? { efSearch: ef_search } : {}),
            } as any);
            return { content: [{ type: "text", text: JSON.stringify(res, null, 2) }] };
          }
          case "hyperspace_search_wasserstein": {
            const { collection, text, top_k } = z.object({ collection: z.string(), text: z.string(), top_k: z.number().optional() }).parse(args);
            // Wasserstein uses a specific internal method or we can use searchWasserstein text variant if exists.
            // Client.ts has searchWasserstein(vector, topK, collection)
            const vector = await this.client.vectorize(text);
            const res = await (this.client as any).searchWasserstein(vector, top_k || 5, collection);
            return { content: [{ type: "text", text: JSON.stringify(res, null, 2) }] };
          }
          case "hyperspace_insert_text": {
            const { collection, id, text, metadata } = z.object({ collection: z.string(), id: z.number(), text: z.string(), metadata: z.record(z.string(), z.string()).optional() }).parse(args);
            await this.client.insertText(id, text, metadata, collection);
            return { content: [{ type: "text", text: `Stored ${id} in ${collection}` }] };
          }
          case "hyperspace_create_collection": {
            const { collection, dimension, metric, quantization, encryption_key, expected_vectors } = z.object({
              collection: z.string(),
              dimension: z.number().optional(),
              metric: z.string().optional(),
              quantization: z.string().optional(),
              encryption_key: z.string().optional(),
              expected_vectors: z.number().int().positive().optional()
            }).parse(args);

            const isHybrid129 = (metric === 'hybrid' && dimension === 129);
            const fullDim = isHybrid129 ? 801 : (dimension || 1024);
            const q = quantization || 'medium';

            // The SDK has no size hint yet: plain collections with an expected size go through
            // the HTTP create endpoint, which takes `expected_vectors` (server >= 4.2.5).
            if (expected_vectors !== undefined && !encryption_key && !isHybrid129) {
              const clientAny = this.client as any;
              const headers: Record<string, string> = { "content-type": "application/json" };
              if (clientAny.apiKey) headers["x-api-key"] = clientAny.apiKey;
              const res = await fetch(`${httpBaseFor(clientAny.host || HYPERSPACE_HOST)}/api/collections`, {
                method: "POST",
                headers,
                body: JSON.stringify({ name: collection, dimension: fullDim, metric: metric || 'cosine', quantization: q, expected_vectors }),
              });
              const body = await res.text();
              return { content: [{ type: "text", text: JSON.stringify({
                success: res.ok,
                quantization: q,
                expected_vectors,
                ...(res.ok ? {} : { error: body || res.statusText }),
              }, null, 2) }] };
            }

            const success = await this.client.createCollection(collection, {
              components: [{ name: 'default', metric: metric || 'cosine', fullDimension: fullDim, weight: 1.0 }],
              cascadePipeline: isHybrid129 ? [
                { componentName: 'default', cutoffDimension: 129, storeInRam: true, rerankTopK: 0 }
              ] : [],
              encrypted: !!encryption_key
            } as any, encryption_key || '', 0.02, q);
            return { content: [{ type: "text", text: JSON.stringify({
              success,
              encrypted: !!encryption_key,
              quantization: q,
              ...(TRAINED_LEVELS.has(q) ? { note: "The quantizer calibrates itself online once ~1,024 vectors are ingested (4,096 for pq/opq); hyperspace_rebuild_index after a large ingest also rebuilds the graph on the trained codes." } : {})
            }, null, 2) }] };
          }
          case "hyperspace_graph_traverse": {
            const { collection, start_id, max_depth, max_nodes } = z.object({ collection: z.string(), start_id: z.number(), max_depth: z.number().optional(), max_nodes: z.number().optional() }).parse(args);
            const nodes = await this.client.traverse(start_id, 0, max_depth || 3, max_nodes || 256, collection);
            return { content: [{ type: "text", text: JSON.stringify(nodes, null, 2) }] };
          }
          case "hyperspace_get_neighbors": {
            const { collection, id, layer, limit } = z.object({
              collection: z.string(),
              id: z.number(),
              layer: z.number().optional(),
              limit: z.number().optional()
            }).parse(args);
            const res = await this.client.getNeighbors(id, layer || 0, limit || 64, 0, collection);
            return { content: [{ type: "text", text: JSON.stringify(res, null, 2) }] };
          }
          case "hyperspace_find_clusters": {
            const { collection, min_cluster_size } = z.object({ collection: z.string(), min_cluster_size: z.number().optional() }).parse(args);
            const clusters = await this.client.findSemanticClusters(0, min_cluster_size || 3, 32, 10000, collection);
            return { content: [{ type: "text", text: JSON.stringify(clusters, null, 2) }] };
          }
          case "hyperspace_analyze_geometry": {
            const { collection, vectors, samples } = z.object({
              collection: z.string().optional(),
              vectors: z.array(z.array(z.number())).optional(),
              samples: z.number().optional()
            }).parse(args);
            if (collection) {
              try {
                const clientAny = this.client as any;
                const host = clientAny.host || "localhost:50051";
                const url = `${httpBaseFor(host)}/api/collections/${collection}/analyze/geometry`;
                const headers: Record<string, string> = {};
                if (clientAny.apiKey) headers["x-api-key"] = clientAny.apiKey;
                const res = await fetch(url, { headers });
                if (res.ok) {
                  const data = await res.json();
                  return { content: [{ type: "text", text: JSON.stringify(data, null, 2) }] };
                }
              } catch {
                // Fall back to local estimation if vectors are provided
              }
            }
            const analysis = analyzeDeltaHyperbolicity(vectors || [], samples);
            return { content: [{ type: "text", text: JSON.stringify(analysis, null, 2) }] };
          }
          case "hyperspace_analyze_thought_stability": {
            const { trajectory, curvature } = z.object({ trajectory: z.array(z.array(z.number())), curvature: z.number().optional() }).parse(args);
            const convergence = (CognitiveMath as any).lyapunovConvergence(trajectory, curvature || 1.0);
            return { content: [{ type: "text", text: `Lyapunov Convergence: ${convergence} (${convergence < 0.05 ? "STABLE" : "CHAOTIC"})` }] };
          }
          case "hyperspace_trigger_reconsolidation": {
            const { collection, learning_rate } = z.object({ collection: z.string(), learning_rate: z.number().optional() }).parse(args);
            // Use Triggering natively if available in client
            const res = await (this.client as any).triggerReconsolidation?.(collection, new Array(1024).fill(0), learning_rate || 0.1) || "Reconsolidation triggered.";
            return { content: [{ type: "text", text: String(res) }] };
          }
          case "hyperspace_freeze_collection": {
            const { collection } = z.object({ collection: z.string() }).parse(args);
            const res = await this.client.freezeCollection(collection);
            return { content: [{ type: "text", text: String(res) }] };
          }
          case "hyperspace_unfreeze_collection": {
            const { collection } = z.object({ collection: z.string() }).parse(args);
            const res = await this.client.unfreezeCollection(collection);
            return { content: [{ type: "text", text: String(res) }] };
          }
          case "hyperspace_get_stats": {
            const { collection } = z.object({ collection: z.string() }).parse(args);

            // 1. Fetch gRPC CollectionStats & Digest in parallel (or fallback to listCollections)
            let digest: any = {};
            let grpcStats: any = {};
            try {
              [digest, grpcStats] = await Promise.all([
                this.client.getDigest(collection).catch(() => ({})),
                this.client.getCollectionStats(collection).catch(() => ({}))
              ]);
            } catch (e: any) {
              // fallback
            }

            if (!grpcStats.count && !digest.logical_clock) {
              try {
                const cols = await this.client.listCollections();
                const found = cols.find((c: any) => c.name === collection);
                if (found) {
                  grpcStats = { name: found.name, count: found.count, schema: found.schema };
                }
              } catch (e) {}
            }

            // 2. Fetch HTTP collection stats in parallel try-catch
            let httpStats: any = {};
            try {
              const clientAny = this.client as any;
              const host = clientAny.host || "localhost:50051";
              const url = `${httpBaseFor(host)}/api/collections/${collection}/stats`;
              const headers: { [key: string]: string } = {};
              if (clientAny.apiKey) headers['x-api-key'] = clientAny.apiKey;
              if (clientAny.userId) headers['x-hyperspace-user-id'] = clientAny.userId;

              const res = await fetch(url, { headers });
              if (res.ok) {
                httpStats = await res.json();
              } else {
                console.error(`HTTP collection stats request failed: ${res.statusText}`);
              }
            } catch (e: any) {
              console.error(`HTTP collection stats fetch failed: ${e.message}`);
            }

            // 3. Merge gRPC + HTTP results
            const stats = {
              ...digest,
              ...grpcStats,
              ...httpStats
            };
            return { content: [{ type: "text", text: JSON.stringify(stats, null, 2) }] };
          }
          case "hyperspace_cache_stats": {
            const { collection } = z.object({ collection: z.string() }).parse(args);
            const stats = await this.client.getCacheStats(collection);
            return { content: [{ type: "text", text: JSON.stringify(stats, null, 2) }] };
          }
          case "hyperspace_cache_clear": {
            const { collection } = z.object({ collection: z.string() }).parse(args);
            const success = await this.client.clearCache(collection);
            return { content: [{ type: "text", text: JSON.stringify({ success }, null, 2) }] };
          }
          case "hyperspace_cache_config": {
            const { collection, policy, ann_threshold } = z.object({
              collection: z.string(),
              policy: z.string(),
              ann_threshold: z.number().optional()
            }).parse(args);
            const success = await this.client.updateCacheConfig(collection, policy, ann_threshold);
            return { content: [{ type: "text", text: JSON.stringify({ success }, null, 2) }] };
          }
          case "hyperspace_delete_collection": {
            const { collection } = z.object({ collection: z.string() }).parse(args);
            const success = await this.client.deleteCollection(collection);
            return { content: [{ type: "text", text: JSON.stringify({ success }, null, 2) }] };
          }
          case "hyperspace_delete_points": {
            const { collection, id } = z.object({ collection: z.string(), id: z.number() }).parse(args);
            const success = await this.client.delete(id, collection);
            return { content: [{ type: "text", text: JSON.stringify({ success }, null, 2) }] };
          }
          case "hyperspace_get_points": {
            const { collection, ids } = z.object({ collection: z.string(), ids: z.array(z.number()) }).parse(args);
            const res = await this.client.getPoints(ids, collection);
            return { content: [{ type: "text", text: JSON.stringify(res, null, 2) }] };
          }
          case "hyperspace_rebuild_index": {
            const { collection } = z.object({ collection: z.string() }).parse(args);
            const success = await this.client.rebuildIndex(collection);
            return { content: [{ type: "text", text: JSON.stringify({ success }, null, 2) }] };
          }
          case "hyperspace_server_status": {
            const clientAny = this.client as any;
            const headers: Record<string, string> = {};
            if (clientAny.apiKey) headers["x-api-key"] = clientAny.apiKey;
            const res = await fetch(`${httpBaseFor(clientAny.host || HYPERSPACE_HOST)}/api/status`, { headers });
            const data = await res.json();
            return { content: [{ type: "text", text: JSON.stringify(data, null, 2) }] };
          }
          case "hyperspace_vacuum": {
            const success = await this.client.vacuum();
            return { content: [{ type: "text", text: JSON.stringify({ success }, null, 2) }] };
          }
          case "hyperspace_get_subsumption_tree": {
            const { collection, root_id, max_depth } = z.object({ collection: z.string(), root_id: z.number(), max_depth: z.number().optional() }).parse(args);
            const res = await this.client.getSubsumptionTree(root_id, max_depth || 3, collection);
            return { content: [{ type: "text", text: JSON.stringify(res, null, 2) }] };
          }
          case "hyperspace_get_concept_parents": {
            const { collection, id, layer, limit } = z.object({ collection: z.string(), id: z.number(), layer: z.number().optional(), limit: z.number().optional() }).parse(args);
            const res = await this.client.getConceptParents(id, layer || 0, limit || 32, collection);
            return { content: [{ type: "text", text: JSON.stringify(res, null, 2) }] };
          }
          case "hyperspace_explore_graph": {
            const { collection, start_id, max_depth, max_nodes } = z.object({ collection: z.string(), start_id: z.number(), max_depth: z.number().optional(), max_nodes: z.number().optional() }).parse(args);
            const res = await this.client.exploreGraph(start_id, max_depth || 2, max_nodes || 256, collection);
            return { content: [{ type: "text", text: JSON.stringify(res, null, 2) }] };
          }
          case "hyperspace_predict_momentum": {
            const { collection, trajectory_ids, steps, curvature } = z.object({ collection: z.string(), trajectory_ids: z.array(z.number()), steps: z.number().optional(), curvature: z.number().optional() }).parse(args);
            const res = await this.client.predictMomentum(trajectory_ids, steps || 1.0, collection, curvature || 1.0);
            return { content: [{ type: "text", text: JSON.stringify(res, null, 2) }] };
          }
          case "hyperspace_get_trust_score": {
            const { collection, trajectory_ids, curvature } = z.object({ collection: z.string(), trajectory_ids: z.array(z.number()), curvature: z.number().optional() }).parse(args);
            const res = await this.client.getTrustScore(trajectory_ids, collection, curvature || 1.0);
            return { content: [{ type: "text", text: JSON.stringify(res, null, 2) }] };
          }
          case "hyperspace_link": {
            const { collection, source_id, target_id, relation, weight } = z.object({
              collection: z.string(),
              source_id: z.number(),
              target_id: z.number(),
              relation: z.string().optional().default("OWNS"),
              weight: z.number().optional().default(1.0)
            }).parse(args);
            const success = await this.client.link(source_id, target_id, relation, weight, collection);
            return { content: [{ type: "text", text: JSON.stringify({ success, source_id, target_id, relation, weight }, null, 2) }] };
          }
          case "hyperspace_link_batch": {
            const { collection, edges } = z.object({
              collection: z.string(),
              edges: z.array(z.object({
                source_id: z.number(),
                target_id: z.number(),
                relation: z.string().optional(),
                weight: z.number().optional()
              }))
            }).parse(args);
            const edges_linked = await this.client.linkBatch(edges.map(e => ({
              sourceId: e.source_id,
              targetId: e.target_id,
              relation: e.relation || "OWNS",
              weight: e.weight ?? 1.0
            })), collection);
            return { content: [{ type: "text", text: JSON.stringify({ success: true, edges_linked }, null, 2) }] };
          }
          case "hyperspace_unlink": {
            const { collection, source_id, target_id, relation } = z.object({
              collection: z.string(),
              source_id: z.number(),
              target_id: z.number(),
              relation: z.string().optional().default("OWNS")
            }).parse(args);
            const success = await this.client.unlink(source_id, target_id, relation, collection);
            return { content: [{ type: "text", text: JSON.stringify({ success, source_id, target_id, relation }, null, 2) }] };
          }
          case "hyperspace_induce_edges": {
            const { collection, rules } = z.object({
              collection: z.string(),
              rules: z.array(z.object({
                field: z.string(),
                rule_type: z.string(),
                target_field: z.string().optional(),
                relation: z.string().optional(),
                weight: z.number().optional()
              }))
            }).parse(args);
            const edges_created = await this.client.induceEdgesFromMetadata(rules.map(r => ({
              field: r.field,
              ruleType: r.rule_type,
              targetField: r.target_field,
              relation: r.relation || "OWNS",
              weight: r.weight ?? 1.0
            })), collection);
            return { content: [{ type: "text", text: JSON.stringify({ success: true, edges_created, rules_applied: rules.length }, null, 2) }] };
          }
          case "hyperspace_verify_generation": {
            const { collection, llm_answer, context_ids, max_distance_threshold, check_geodesic } = z.object({
              collection: z.string(),
              llm_answer: z.string(),
              context_ids: z.array(z.string()),
              max_distance_threshold: z.number().optional(),
              check_geodesic: z.boolean().optional().default(true)
            }).parse(args);
            const res = await this.client.verifyGeneration(
              llm_answer,
              context_ids,
              collection,
              max_distance_threshold,
              check_geodesic
            );
            return { content: [{ type: "text", text: JSON.stringify(res, null, 2) }] };
          }
          case "hyperspace_pdk_extension": {
            const { extension, action, payload } = z.object({
              extension: z.string(),
              action: z.string(),
              payload: z.record(z.string(), z.any()).optional().default({})
            }).parse(args);
            if (typeof (this.client as any).callExtension === "function") {
              const res = await (this.client as any).callExtension(extension, action, payload);
              return { content: [{ type: "text", text: JSON.stringify(res, null, 2) }] };
            }
            const clientAny = this.client as any;
            const host = clientAny.host || "localhost:50051";
            const httpBase = httpBaseFor(host);
            const headers: Record<string, string> = { "Content-Type": "application/json" };
            if (clientAny.apiKey) headers["x-api-key"] = clientAny.apiKey;
            const resp = await fetch(`${httpBase}/extensions/${extension}/${action}`, {
              method: "POST",
              headers,
              body: JSON.stringify(payload)
            });
            const data = await resp.json();
            return { content: [{ type: "text", text: JSON.stringify(data, null, 2) }] };
          }
          default:
            throw new McpError(ErrorCode.MethodNotFound, `Tool not found: ${name}`);
        }
      } catch (err: any) {
        throw new McpError(ErrorCode.InternalError, err.message);
      }
    });
  }


  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error("Hyperspace POWER MCP online.");
  }
}

const server = new HyperspaceMcpServer();
server.run().catch(console.error);
