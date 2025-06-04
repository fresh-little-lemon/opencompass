from opencompass.models import HuggingFacewithChatTemplate
from opencompass.datasets import HFDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_evaluator import PaperCLSEvaluator
from opencompass.openicl.icl_retriever import ZeroRetriever

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr="internlm3",
        path='/root/share/new_models/internlm3/internlm3-8b-instruct',
        tokenizer_path='/root/share/new_models/internlm3/internlm3-8b-instruct',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        max_seq_len=32768,
        max_out_len=16384,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]

reader_cfg = dict(
    input_columns=["title", "abstract"],
    output_column="categories"
)

infer_cfg = dict(
    prompt_template=dict(
        type = PromptTemplate,
        template = dict(
            begin=[
                dict(role="SYSTEM", fallback_role='HUMAN', prompt='You are a highly intelligent and reliable academic paper classification assistant. You are known for your exceptional ability to accurately categorize scientific papers into the correct research fields based solely on their titles and abstracts. You approach every task with careful reasoning and a deep understanding of scientific disciplines. Always think step by step, and double-check your reasoning before you make a final decision.'),
            ],
            round=[
                dict(role="HUMAN", prompt="""**Paper Classification Guide**

**Task**

You are an experienced academic reviewer with a deep understanding of various research domains.  Based on the paper's title: `{title}` and abstract: `{abstract}`, your mission is to assign it to the most appropriate category from the list below. Your response must be formatted as `<category>[CATEGORY_NAME]</category>`.

**Categories and Descriptions**

### astro-ph : Astrophysics

  * **Cosmology**: Early universe phenomenology, cosmic microwave background, cosmological parameters, primordial element abundances.
  * **Large-scale structure**: Extragalactic distance scale, large-scale structure of the universe, groups, superclusters, voids, intergalactic medium.
  * **Particle astrophysics**: Dark energy, dark matter, baryogenesis, leptogenesis, inflationary models, reheating, monopoles, WIMPs, cosmic strings, primordial black holes, cosmological gravitational radiation.

    *Example*: Research on dark matter distribution in galaxy clusters.

### cond-mat.mes-hall : Mesoscale and Nanoscale Physics

  * **Nanostructures**: Semiconducting nanostructures (quantum dots, wires, wells).
  * **Electronic properties**: Single electronics, spintronics, 2D electron gases, quantum Hall effect.
  * **Carbon materials**: Nanotubes, graphene, plasmonic nanostructures.

    *Example*: Study of quantum dots for optoelectronic applications.

### cond-mat.mtrl-sci : Materials Science

  * **Material properties**: Techniques, synthesis, characterization, structural phase transitions, mechanical properties, phonons.
  * **Defects and interfaces**: Defects, adsorbates, interfaces.

    *Example*: Investigation of novel synthesis methods for high-temperature superconductors.

### cs.CL : Computation and Language

  * **Natural language processing**: Computational linguistics, speech processing, text retrieval.
  * **Scope**: Includes material related to natural language processing, computational linguistics, speech, and text retrieval.

    *Note*: Programming languages and formal systems without natural language focus are excluded.

    *Example*: Development of a machine translation algorithm.

### cs.CV : Computer Vision and Pattern Recognition

  * **Image analysis**: Image processing, computer vision, pattern recognition, scene understanding.

    *Example*: Object detection algorithm for autonomous vehicles.

### cs.LG : Machine Learning

  * **Learning paradigms**: Supervised learning, unsupervised learning, reinforcement learning, bandit problems.
  * **ML aspects**: Robustness, explanation, fairness, methodology.
  * **Applications**: Applications of machine learning methods.

    *Example*: Novel neural network architecture for image classification.

### gr-qc : General Relativity and Quantum Cosmology

  * **Gravitational physics**: Gravitational wave detection, experimental tests of gravitational theories, computational general relativity.
  * **Relativistic astrophysics**: Relativistic astrophysics, solutions to Einstein's equations, alternative gravity theories.
  * **Cosmology**: Classical and quantum cosmology, quantum gravity.

    *Example*: Numerical simulation of black hole mergers.

### hep-ph : High Energy Physics - Phenomenology

  * **Particle physics**: Theoretical particle physics, experimental interrelations, particle physics observables prediction.
  * **Models and techniques**: Models and effective field theories, calculation techniques, experimental results analysis.

    *Example*: Prediction of Higgs boson decay channels.

### hep-th : High Energy Physics - Theory

  * **Quantum field theory**: Formal aspects of quantum field theory.
  * **Unified theories**: String theory, supersymmetry, supergravity.

    *Example*: String theory model of particle interactions.

### quant-ph : Quantum Physics

  * **Quantum foundations**: Fundamental quantum mechanics.
  * **Quantum technologies**: Quantum information, quantum computing, quantum optics, quantum thermodynamics.

    *Example*: Quantum algorithm for factoring large numbers.

**Classification Instructions**

  1. **Read carefully**: Carefully read the paper's title and abstract, as if you were reviewing a high - impact research paper.
  2. **Identify keywords**: Identify key concepts, methodologies, and applications, leveraging your extensive knowledge of academic literature.
  3. **Match to category**: Match these elements to the most specific category, drawing upon your experience in categorizing diverse research works.
  4. **Resolve conflicts**: If multiple categories seem applicable, use your expertise to choose the one with the strongest relevance to the paper's primary contribution.
  5. **Format response**: Format your answer as `<category>[CATEGORY_NAME]</category>`, ensuring clarity and precision in your classification.

**Classification Process**

When assigning the category, follow these steps:

  1. Carefully extract the paper's **main topic**, **methods used**, and **target problem or application**.
  2. Compare these with the candidate categories.
  3. Take into account the observed category distribution in the training set:
     * quant-ph: 15.99%
     * astro-ph: 13.36%
     * hep-ph: 11.94%
     * gr-qc: 11.74%
     * cs.LG: 9.92%
     * cs.CV: 9.31%
     * cs.CL: 8.50%
     * hep-th: 8.50%
     * cond-mat.mtrl-sci: 5.87%
     * cond-mat.mes-hall: 4.86%

Use this distribution as a soft prior—favoring more common categories only when the content is ambiguous.

**Important Guidelines**

  * You **must choose exactly one** category from the list.
  * ❌ Do NOT invent new categories.
  * ❌ Do NOT return multiple labels.

**Thinking Process**

🔴 Think step by step. First, explain your reasoning in 2-3 sentences. Then, output the final result in the format below:

`<category>your_selected_category</category>`

Let's begin."""),
                dict(role="BOT", prompt="{categories}")
            ],
        )
    ),
    inferencer = dict(type=ChatInferencer),
    retriever=dict(type=ZeroRetriever)
)

eval_cfg = dict(evaluator=dict(type=PaperCLSEvaluator))

datasets = [
    dict(
        type=HFDataset,
        path='json',
        reader_cfg=reader_cfg,
        data_files="/root/finetune/dataprep/arxiv-metadata-oai-snapshot-balanced-sampled-30/train.jsonl",
        split="all",
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]