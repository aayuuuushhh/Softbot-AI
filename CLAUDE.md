Project instructions for Claude Code. Read this fully before writing any code.

1. What this is
Uddhar turns multi-modal disaster imagery (pre/post-event satellite pairs + crowd-sourced ground uploads) into an intelligent, web-informed resource allocation engine.

What was damaged — deep learning-based structural damage classification using PyTorch, fusing overhead satellite change detection with granular ground-level image analysis.

What each zone needs — dynamic management of human resources, food, water, and medical supplies computed via spatial graph analysis.

How to distribute it — an autonomous agent that searches the web for historical disaster response data (e.g., Nepal 2015, Jajarkot 2023, Rasuwa 2026) to determine the most optimal, evidence-based resource distribution strategies.

The core insight
Traditional relief logistics rely on manual data entry and static distribution formulas that fail when infrastructure collapses. By utilizing deep learning for immediate visual triage and internet-grounded autonomous agents for resource routing, Uddhar bypasses the bottleneck of manual requests. Need is computed visually, and distribution is optimized by learning from the successes and failures of previous global disasters.

2. Non-negotiable constraints
These are the architectural commitments for the ML-driven pipeline. Do not violate them for convenience.

Constraint	Rule
GPU-Accelerated Core	The perception layer relies on PyTorch and CUDA. The system must efficiently batch process satellite tiles and incoming ground images for rapid inference.
Multi-Modal Perception	Damage classification cannot rely solely on satellite imagery, which is vulnerable to monsoon cloud cover. The architecture must accept and fuse user-uploaded ground/drone imagery.
Web-Grounded Allocation	Resource distribution algorithms must not rely exclusively on hardcoded heuristics. An agent must query the web for historical precedents and adapt the allocation weights based on real-world disaster data.
Traceable AI	Black-box routing is dangerous in disaster response. When the agent re-allocates medical teams or food, it must attach a citation (e.g., "Strategy adapted from 2015 Gorkha earthquake logistics report").
Graph-Based Reasoning	Road networks and distribution nodes must be modeled as graphs (using PyTorch Geometric) to compute reachability and optimize supply chains dynamically.
3. Architecture
Plaintext
INPUT            pre.tif, post.tif  [+ user_uploaded_ground_images.jpg]
                                    │
─────────────────────────────────── ▼ ──────────────────────────────────────
PERCEPTION       S1  overhead inference   PyTorch ViT/CNN change detection on satellite pairs
                 S2  ground inference     OpenCV + PyTorch classification on uploaded photos
                 S3  fusion               Geospatial mapping of damage confidence
                                    │
─────────────────────────────────── ▼ ──────────────────────────────────────
REASONING        S4  extract              Bounding boxes → damage severity (minor/major/destroyed)
                 S5  graph routing        PyTorch Geometric network of blocked/open roads
                 S6  inventory state      Current human resources, food, and medical supplies
                                    │
─────────────────────────────────── ▼ ──────────────────────────────────────
AGENTIC ROUTING  S7  historical search    Agent queries web for similar past disaster logistics
                 S8  allocation           Graph-based optimization of available vs. needed resources
                                    │
─────────────────────────────────── ▼ ──────────────────────────────────────
OUTPUT           damage overlay · dynamic allocation dashboard · routing API
                 JSON endpoints for mobile field teams
4. Core Features
F1 — Multi-Modal Damage Classification
Input: pre.tif, post.tif, crowdsourced images.
Output: Geotagged damage severity mappings.

Satellite Change Detection: Implement a Siamese Convolutional Neural Network (or Vision Transformer) in PyTorch to compare pre- and post-disaster satellite tiles. The model outputs a semantic segmentation mask highlighting destroyed structures and blocked transit corridors.

Ground-Level Verification: Expose a FastAPI endpoint for field workers to upload images of damaged buildings. Run a lightweight classification model (e.g., ResNet or MobileNet via PyTorch) to score structural integrity.

Data Fusion: When ground images and satellite detections overlap, weight the ground image higher for severity classification, bypassing cloud-cover limitations.

F2 — Resource and Inventory Management
Input: Current stock of supplies and personnel.
Output: Real-time database state of available relief.

Maintain a live ledger of human resources (doctors, rescue personnel, engineers), food rations, water purification units, and medical kits.

Track the location of holding centers, staging nodes, and transit vehicles.

The system continually subtracts assigned resources from the global pool and tracks them in transit via a Next.js/React frontend dashboard.

F3 — Agentic Resource Allocation & Web Research
Input: F1 (Damage map), F2 (Inventory state).
Output: Optimized dispatch manifests and routing graphs.

This is the system's intelligent core. Instead of static math, an autonomous agent handles distribution:

Context Gathering: The agent analyzes the scale of the current disaster (e.g., "Flood in Rasuwa, 200 buildings destroyed, 5 roads blocked").

Web Research: It searches the internet for case studies and logistical data from similar events (e.g., optimal doctor-to-patient ratios in isolated flood zones, or food distribution hierarchies used during the 2014 Sindhupalchok landslides).

Graph Optimization: Using PyTorch Geometric, the system maps the affected wards as nodes and the roads as edges. The agent adjusts edge weights based on its research and runs an optimization algorithm to route human and physical resources to maximize impact.

Actionable Dispatch: The agent outputs a prioritized list of dispatch orders: "Send 2 medical teams and 400 food rations to Zone A via Helicopter (Road blocked)."

5. Stack Configuration
Layer	Choice	Reason
Deep Learning	PyTorch, PyTorch Geometric	State-of-the-art for computer vision (satellite/ground) and Graph Neural Networks (supply chain routing).
Computer Vision	OpenCV	Pre-processing, normalization, and bounding box manipulation for uploaded imagery.
Backend API	FastAPI (Python)	High-performance asynchronous API, perfect for serving PyTorch models and handling concurrent image uploads.
Database	MongoDB	Flexible schema for handling heterogeneous data (inventory, user uploads, complex JSON agent logs).
Agentic Framework	LangChain / LlamaIndex	Orchestrating the web-search agent, parsing historical disaster data, and structuring LLM outputs.
Web Frontend	Next.js, React, Tailwind	Fast, responsive dashboard for resource tracking and map visualization.
Infrastructure	Docker	Containerized deployment ensuring environment consistency from local development to production servers.
6. Repository Layout
Plaintext
uddhar/
  models/
    satellite_change.pt       PyTorch weights for overhead detection
    ground_damage.pt          PyTorch weights for image uploads
  core/
    vision/
      inference.py            S1 & S2 PyTorch model wrappers
      preprocess.py           OpenCV image normalization
    spatial/
      graph_network.py        PyTorch Geometric routing logic
  agents/
    allocator.py              LLM orchestration for resource distribution
    researcher.py             Web search tool for historical disaster data
  api/
    main.py                   FastAPI entry point
    routes/
      upload.py               Handling ground image ingestion
      inventory.py            Resource management endpoints
  web/                        Next.js + React dashboard
  data/
    training/                 Datasets for fine-tuning damage models
7. Testing and Validation
Model Accuracy: Benchmark the PyTorch models against datasets like xBD (for satellite change detection) and custom local datasets (for Nepali architecture damage). Require an F1 score of > 0.80 on the destroyed class.

Agent Sanity Checks: The resource allocation agent must undergo property testing. It must never allocate more resources than exist in the database, and it must cite valid, historically sound reasoning for its distribution logic.

Graph Connectivity: Assert that resources are never routed through nodes marked as blocked by the computer vision pipeline unless an air-transport flag is actively verified.

8. Prior Art & Reference Implementations
When building the multi-modal perception pipelines and reporting endpoints, refer to the following repository for structural and architectural inspiration:

DisasterIQ ([https://github.com/DarkNem4377/DisasterIQ.git](https://github.com/DarkNem4377/DisasterIQ.git))

Why we reference it: It implements highly similar core mechanics—specifically turning before/after satellite imagery into ranked priority zones, generating visual damage overlays, creating AI situation briefs, and exporting field-ready PDF reports.

How to use it: Look to this repository for baseline approaches on handling GeoTIFF image pipelines, generating the priority algorithms (F2/F3), and formatting the offline PDF outputs (F4). While Uddhar extends this concept with ground-image fusion and an agentic web-research layer, DisasterIQ serves as a proven baseline for the deterministic perception layer.
