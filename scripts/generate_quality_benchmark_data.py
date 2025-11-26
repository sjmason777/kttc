"""Generate high-quality benchmark data using LLM (100+ diverse samples per pair)."""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kttc.utils.config import get_settings

DIVERSE_TOPICS = [
    "Artificial intelligence is revolutionizing modern industries.",
    "Cloud computing enables scalable data storage solutions.",
    "Quantum computers promise exponential computational power.",
    "Blockchain technology ensures transparent transaction records.",
    "Neural networks learn complex patterns from data.",
    "Cybersecurity protects sensitive information from threats.",
    "Internet of Things connects billions of devices.",
    "Machine learning algorithms improve through experience.",
    "5G networks provide ultra-fast wireless connectivity.",
    "Virtual reality creates immersive digital experiences.",
    "Autonomous vehicles navigate without human intervention.",
    "Big data analytics reveals valuable business insights.",
    "Robotics automates repetitive manufacturing tasks.",
    "Augmented reality overlays digital information on reality.",
    "Edge computing processes data near its source.",
    "Natural language processing enables human-computer communication.",
    "Computer vision allows machines to interpret images.",
    "Distributed systems coordinate across multiple machines.",
    "API integration connects different software applications.",
    "DevOps practices streamline software development cycles.",
    "Climate change affects global weather patterns significantly.",
    "Renewable energy sources reduce carbon emissions effectively.",
    "Genetic engineering modifies organism DNA sequences.",
    "Space exploration expands human knowledge of universe.",
    "Nanotechnology manipulates matter at atomic scale.",
    "Ocean acidification threatens marine ecosystem balance.",
    "Biodiversity loss endangers planetary ecological stability.",
    "Particle physics studies fundamental universe building blocks.",
    "Stem cell research offers regenerative medicine potential.",
    "Fusion energy could provide clean unlimited power.",
    "CRISPR technology enables precise genome editing.",
    "Microplastics contaminate water sources worldwide.",
    "Gravitational waves confirm Einstein's relativity predictions.",
    "Dark matter comprises majority of universe mass.",
    "Antibiotic resistance threatens public health systems.",
    "Solar panels convert sunlight into electricity.",
    "Carbon capture technology removes greenhouse gases.",
    "Quantum entanglement enables secure communications.",
    "Exoplanets orbit stars beyond our solar system.",
    "Synthetic biology creates new biological systems.",
    "Market research identifies consumer preferences and trends.",
    "Supply chain optimization reduces operational costs.",
    "Digital transformation modernizes traditional business models.",
    "Customer relationship management improves client satisfaction.",
    "E-commerce platforms facilitate online shopping transactions.",
    "Financial technology innovations disrupt banking services.",
    "Venture capital funds support startup companies.",
    "Business intelligence tools analyze organizational data.",
    "Brand awareness campaigns increase product visibility.",
    "Agile methodology enables flexible project management.",
    "Corporate social responsibility demonstrates ethical commitment.",
    "Competitive analysis evaluates market positioning strategies.",
    "Revenue streams diversify company income sources.",
    "Risk management frameworks protect business assets.",
    "Intellectual property rights safeguard innovations.",
    "Merger acquisitions consolidate market share.",
    "Quarterly earnings reports inform investor decisions.",
    "Business process automation increases efficiency.",
    "Customer retention strategies build long-term loyalty.",
    "Market segmentation targets specific consumer groups.",
    # Daily Life (20)
    "Morning exercise improves overall health significantly.",
    "Balanced diet provides essential nutrients daily.",
    "Quality sleep enhances cognitive function performance.",
    "Regular meditation reduces stress levels effectively.",
    "Social connections strengthen mental wellbeing.",
    "Time management skills increase productivity.",
    "Lifelong learning expands knowledge and skills.",
    "Financial planning ensures future security.",
    "Sustainable lifestyle choices protect environment.",
    "Creative hobbies foster personal expression.",
    "Public transportation reduces traffic congestion.",
    "Recycling programs minimize waste production.",
    "Volunteering activities benefit local communities.",
    "Cultural experiences broaden worldview perspectives.",
    "Pet ownership provides companionship and joy.",
    "Home organization creates peaceful living space.",
    "Cooking healthy meals saves money regularly.",
    "Reading books stimulates imagination and intellect.",
    "Outdoor activities connect people with nature.",
    "Family traditions strengthen generational bonds.",
    "Online learning platforms democratize education access.",
    "Critical thinking skills evaluate information objectively.",
    "Collaborative projects develop teamwork abilities.",
    "Educational technology enhances classroom instruction.",
    "Curriculum design addresses diverse learning needs.",
    "Assessment methods measure student achievement.",
    "Teacher training programs improve instruction quality.",
    "Special education supports students with disabilities.",
    "Literacy programs empower disadvantaged communities.",
    "Higher education prepares workforce for careers.",
    "Preventive care reduces chronic disease occurrence.",
    "Telemedicine expands healthcare access remotely.",
    "Vaccination programs protect against infectious diseases.",
    "Medical imaging diagnostics detect health conditions early.",
    "Personalized medicine tailors treatments to individuals.",
    "Mental health services address psychological wellbeing.",
    "Emergency response systems save critical lives.",
    "Clinical trials evaluate new treatment efficacy.",
    "Health insurance provides financial protection.",
    "Pharmaceutical research develops life-saving medications.",
]


async def generate_benchmark_data():
    """Generate 100+ high-quality translations per language pair."""
    print("\n🎯 Generating High-Quality Benchmark Data (100+ samples/pair)\n")
    print("=" * 80)

    settings = get_settings()
    llm = None

    # Try to get LLM
    try:
        api_key = settings.get_llm_provider_key("anthropic")
        from kttc.llm import AnthropicProvider

        llm = AnthropicProvider(api_key=api_key, model="claude-3-5-haiku-20241022")
        print("✅ Using Anthropic Claude 3.5 Haiku for translations\n")
    except Exception:
        try:
            api_key = settings.get_llm_provider_key("openai")
            from kttc.llm import OpenAIProvider

            llm = OpenAIProvider(api_key=api_key, model="gpt-4o-mini")
            print("✅ Using OpenAI GPT-4o-mini for translations\n")
        except Exception:
            # Silently ignore OpenAI setup errors and try Anthropic instead
            pass

    if not llm:
        print("❌ No LLM provider available. Need API key.")
        return

    # Language pairs
    pairs = [
        ("en", "ru", "Russian"),
        ("en", "zh", "Chinese"),
        ("ru", "en", "English"),
        ("zh", "en", "English"),
        ("ru", "zh", "Chinese"),
        ("zh", "ru", "Russian"),
    ]

    data_dir = Path("tests/benchmarks/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    for src_code, tgt_code, tgt_name in pairs:
        print(f"\n[{src_code}→{tgt_code}] Generating {len(DIVERSE_TOPICS)} translations...")
        print("-" * 80)

        samples = []

        # Get source texts
        if src_code == "en":
            sources = DIVERSE_TOPICS
        else:
            # Translate English topics to source language first
            print(f"  Translating topics to {src_code}...")
            sources = []
            for i, en_text in enumerate(DIVERSE_TOPICS):
                if (i + 1) % 20 == 0:
                    print(f"    Progress: {i + 1}/{len(DIVERSE_TOPICS)}")

                prompt = f"Translate to {src_code}: {en_text}\nTranslation:"
                translation = await llm.complete(prompt, temperature=0.3, max_tokens=200)
                sources.append(translation.strip())

        # Translate to target
        print(f"  Translating {len(sources)} texts to {tgt_name}...")
        for i, src_text in enumerate(sources):
            if (i + 1) % 20 == 0:
                print(f"    Progress: {i + 1}/{len(sources)}")

            prompt = f"Translate to {tgt_name}: {src_text}\nTranslation:"
            translation = await llm.complete(prompt, temperature=0.3, max_tokens=200)

            samples.append(
                {
                    "id": f"quality_{src_code}-{tgt_code}_{i}",
                    "source": src_text,
                    "translation": translation.strip(),
                    "source_lang": src_code,
                    "target_lang": tgt_code,
                    "domain": "diverse",
                    "dataset": "quality_llm",
                }
            )

        # Save
        output_file = data_dir / f"quality_{src_code}_{tgt_code}.json"
        output_file.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  ✅ Saved {len(samples)} samples to {output_file.name}")

    print("\n" + "=" * 80)
    print("✅ High-quality benchmark data generated!")
    print("=" * 80)
    print(
        f"\nTotal: {len(DIVERSE_TOPICS)} samples per pair × 6 pairs = {len(DIVERSE_TOPICS) * 6} translations"
    )


if __name__ == "__main__":
    asyncio.run(generate_benchmark_data())
