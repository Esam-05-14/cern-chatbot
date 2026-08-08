"""Multi-topic LHCb / CERN GitLab Q&A pairs for chatbot quality evals.

Each item may include optional scoring hints used by ``cerngitlab_chat.eval.score``.
"""

from __future__ import annotations

from typing import Any

EVAL_QUESTIONS: list[dict[str, Any]] = [
    # --- Rec / reconstruction ---
    {
        "id": "rec_particle_adder_location",
        "topic": "reconstruction",
        "question": (
            "Search lhcb/Rec for ParticleAdder. Where is it defined, what interface "
            "does it implement, and what does combine() do at a high level?"
        ),
        "reference": "Phys/VertexFit/src/ParticleAdder.cpp; IParticleCombiner.",
        "must_include_any": ["ParticleAdder", "IParticleCombiner", "VertexFit"],
        "must_cite_project_any": ["lhcb/Rec"],
        "require_get_file_content": True,
        "forbid_bad_projects": True,
    },
    {
        "id": "rec_particle_adder_chi2dof",
        "topic": "reconstruction",
        "question": (
            "Search lhcb/Rec and lhcb/Moore for ParticleAdder and RecoConf ParticleCombiner. "
            "Explain ParticleAdder vs a vertex-fit combiner and why F.CHI2DOF cuts with "
            "ParticleAdder are wrong. Cite paths."
        ),
        "reference": (
            "ParticleAdder adds 4-momenta without a geometric vertex fit; CHI2DOF assumes "
            "vertex-fit quality."
        ),
        "must_include_any": ["ParticleAdder", "CHI2DOF", "vertex"],
        "must_cite_project_any": ["lhcb/Rec"],
        "forbid_phrases_any": [
            "always runs CHI2DOF",
            "overfitting to spurious",
        ],
        "require_get_file_content": True,
        "forbid_bad_projects": True,
        "expect_search_projects_any": ["lhcb/Rec", "lhcb/Moore"],
    },
    # --- Moore / HLT2 ---
    {
        "id": "moore_hlt2conf",
        "topic": "trigger",
        "question": (
            "In lhcb/Moore, where does Hlt2Conf live and what is an example HLT2 line "
            "module path?"
        ),
        "reference": "lhcb/Moore; Hlt/Hlt2Conf/...",
        "must_include_any": ["Hlt2Conf", "Moore"],
        "must_cite_project_any": ["lhcb/Moore"],
        "forbid_bad_projects": True,
    },
    {
        "id": "moore_functors_cut",
        "topic": "trigger",
        "question": (
            "Find an example in lhcb/Moore that imports Functors as F and applies a "
            "CHI2DOF or IP-related cut. Cite the file."
        ),
        "reference": "Moore Hlt2Conf or RecoConf usage with Functors as F.",
        "must_include_any": ["Functors", "F."],
        "must_cite_project_any": ["lhcb/Moore"],
        "forbid_bad_projects": True,
    },
    # --- DaVinci / analysis ---
    {
        "id": "davinci_dtf_tutorial",
        "topic": "analysis",
        "question": (
            "Find DecayTreeFitter usage in lhcb/DaVinci (prefer tutorials/examples). "
            "List constructor arguments and cite the path."
        ),
        "reference": "tutorial6_DecayTreeFit.py; name, input_particles, mass_constraints, input_pvs.",
        "must_include_any": ["DecayTreeFitter", "input_particles"],
        "must_cite_project_any": ["lhcb/DaVinci"],
        "require_get_file_content": True,
        "forbid_bad_projects": True,
    },
    {
        "id": "davinci_get_particles",
        "topic": "analysis",
        "question": (
            "In modern DaVinci, how are particles loaded with get_particles and how can "
            "that feed DecayTreeFitter? Cite an example file."
        ),
        "reference": "PyConf.reading.get_particles → DecayTreeFitter input_particles.",
        "must_include_any": ["get_particles", "DecayTreeFitter"],
        "must_cite_project_any": ["lhcb/DaVinci"],
        "forbid_bad_projects": True,
    },
    # --- Allen / HLT1 ---
    {
        "id": "allen_configuration",
        "topic": "trigger",
        "question": (
            "In lhcb/Allen, point to a configuration or sequence entry area for HLT1 "
            "and cite a concrete path."
        ),
        "reference": "lhcb/Allen configuration / sequence paths.",
        "must_include_any": ["Allen"],
        "must_cite_project_any": ["lhcb/Allen"],
        "forbid_bad_projects": True,
    },
    # --- Gaudi / framework ---
    {
        "id": "gaudi_algorithm",
        "topic": "framework",
        "question": (
            "In lhcb/Gaudi (or closely related), show how an Algorithm is declared or "
            "configured. Cite a real header or Python example."
        ),
        "reference": "Gaudi Algorithm base / Configurable pattern with path citation.",
        "must_include_any": ["Algorithm", "Gaudi"],
        "must_cite_project_any": ["lhcb/Gaudi"],
        "forbid_bad_projects": True,
    },
    # --- Gauss / simulation ---
    {
        "id": "gauss_decfiles",
        "topic": "simulation",
        "question": (
            "Where do Gauss or related LHCb simulation packages reference decay files "
            "or event types? Cite project and path."
        ),
        "reference": "lhcb/Gauss or DecFiles-related references with citations.",
        "must_include_any": ["Gauss", "decay"],
        "must_cite_project_any": ["lhcb/Gauss"],
        "forbid_bad_projects": True,
    },
    # --- Detector / geometry ---
    {
        "id": "detector_dd4hep",
        "topic": "geometry",
        "question": (
            "In lhcb/Detector, find a DD4hep or geometry-related source/config path and "
            "summarize what it configures. Cite the path."
        ),
        "reference": "lhcb/Detector DD4hep/geometry files.",
        "must_include_any": ["Detector"],
        "must_cite_project_any": ["lhcb/Detector"],
        "forbid_bad_projects": True,
    },
    # --- LHCb core ---
    {
        "id": "lhcb_event_model",
        "topic": "event_model",
        "question": (
            "In lhcb/LHCb, find a Particle or Track-related event-model header/source "
            "and cite it."
        ),
        "reference": "Event model under lhcb/LHCb.",
        "must_include_any": ["LHCb", "Particle"],
        "must_cite_project_any": ["lhcb/LHCb"],
        "forbid_bad_projects": True,
    },
    # --- Stack / tooling meta ---
    {
        "id": "stack_projects_list",
        "topic": "stack",
        "question": (
        "List the core LHCb stack GitLab projects this chatbot indexes by default "
        "(Moore, Allen, and related) with canonical lhcb/... paths."
        ),
        "reference": "Moore, Allen, Gaudi, Rec, LHCb, Gauss, DaVinci.",
        "must_include_any": ["lhcb/Moore", "lhcb/Rec", "lhcb/DaVinci"],
        "forbid_bad_projects": True,
    },
    {
        "id": "recoconf_not_a_project",
        "topic": "stack",
        "question": (
            "Is RecoConf a CERN GitLab project? Where should one search for RecoConf "
            "ParticleCombiner code?"
        ),
        "reference": "RecoConf lives under Rec/Moore packages, not as lhcb/RecoConf project.",
        "must_include_any": ["RecoConf", "lhcb/Rec"],
        "forbid_phrases_any": ["project named RecoConf"],
        "forbid_bad_projects": True,
    },
    # --- Cross-cutting / pitfalls ---
    {
        "id": "gitlab_search_hygiene",
        "topic": "tooling",
        "question": (
            "When searching CERN GitLab for ParticleAdder, which project path casing "
            "should be used, and should search_term use regex OR like a|b?"
        ),
        "reference": "Use lhcb/Rec (not lhcb/reco); no regex OR — separate searches.",
        "must_include_any": ["lhcb/Rec"],
        "forbid_phrases_any": ["lhcb/reco"],
        "forbid_bad_projects": True,
    },
    {
        "id": "hybrid_rag_mcp_role",
        "topic": "tooling",
        "question": (
            "In this chatbot architecture, what is retrieve_code (RAG) for versus "
            "get_file_content / search_code (MCP)? Answer briefly."
        ),
        "reference": "RAG: fast local recall; MCP: live verify/fresh content.",
        "must_include_any": ["RAG", "MCP"],
        "forbid_bad_projects": True,
    },
    # --- Releases / docs style ---
    {
        "id": "moore_readme",
        "topic": "docs",
        "question": "Fetch the README for lhcb/Moore and summarize what Moore is in one sentence with a citation.",
        "reference": "get_project_readme on lhcb/Moore.",
        "must_include_any": ["Moore"],
        "must_cite_project_any": ["lhcb/Moore"],
        "expect_tools_any": ["get_project_readme", "get_file_content", "search_code"],
        "forbid_bad_projects": True,
    },
    {
        "id": "davinci_branches_or_tags",
        "topic": "scm",
        "question": "List a few branches or tags available on lhcb/DaVinci (use list_branches or list_tags) and cite the project.",
        "reference": "list_branches/list_tags on lhcb/DaVinci.",
        "must_include_any": ["lhcb/DaVinci"],
        "expect_tools_any": ["list_branches", "list_tags", "get_project_info"],
        "forbid_bad_projects": True,
    },
    {
        "id": "rec_phys_layout",
        "topic": "reconstruction",
        "question": "Using list_project_files on lhcb/Rec, what top-level directories exist under Phys/ (or similar) related to vertexing?",
        "reference": "Phys/ under lhcb/Rec including VertexFit.",
        "must_include_any": ["Phys", "VertexFit"],
        "must_cite_project_any": ["lhcb/Rec"],
        "expect_tools_any": ["list_project_files", "search_code"],
        "forbid_bad_projects": True,
    },
]
