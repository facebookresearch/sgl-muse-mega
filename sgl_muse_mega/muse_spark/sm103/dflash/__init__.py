"""The DFlash2 draft on the stack: a 5-layer block-diffusion draft, the verify and the accept (greedy or sampled) as one
CUDA graph.

Modules: ``weights`` (the draft's config, its per-rank weights and the kernels' layouts), the launches' binds
(``layer``: the draft layer, its qkv and the attention's geometry; ``head``: the entry and the draft head;
``verify_accept``: the verify's lm_head and the accept) and ``iteration`` (one rank's draft + verify).
"""
