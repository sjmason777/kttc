"""Streamlit web demo for KTTC Translation QA.

This app provides an interactive web interface to test KTTC's translation quality
evaluation capabilities.

To run locally:
    streamlit run examples/streamlit_app.py

To deploy on Streamlit Cloud:
    1. Push to GitHub
    2. Go to https://share.streamlit.io/
    3. Connect your repository
    4. Set main file: examples/streamlit_app.py
"""

import asyncio
from typing import Any

import streamlit as st

# Page config
st.set_page_config(
    page_title="KTTC - Translation QA Demo",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_llm_provider(use_demo: bool, provider_type: str, api_key: str) -> Any:
    """Get LLM provider based on user selection.

    Args:
        use_demo: Whether to use demo mode
        provider_type: Type of provider (openai/anthropic)
        api_key: API key for the provider

    Returns:
        LLM provider instance
    """
    if use_demo:
        from kttc.cli.demo import DemoLLMProvider

        return DemoLLMProvider()

    if provider_type == "openai":
        from kttc.llm import OpenAIProvider

        return OpenAIProvider(api_key=api_key)
    from kttc.llm import AnthropicProvider

    return AnthropicProvider(api_key=api_key)


async def evaluate_translation(
    source_text: str,
    translation: str,
    source_lang: str,
    target_lang: str,
    llm_provider: Any,
) -> Any:
    """Evaluate translation quality.

    Args:
        source_text: Source text
        translation: Translation to evaluate
        source_lang: Source language code
        target_lang: Target language code
        llm_provider: LLM provider instance

    Returns:
        Evaluation report
    """
    from kttc.agents import AgentOrchestrator
    from kttc.core import TranslationTask

    orchestrator = AgentOrchestrator(llm_provider)

    task = TranslationTask(
        source_text=source_text,
        translation=translation,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    report = await orchestrator.evaluate(task)
    return report


# Header
st.title("🌐 KTTC Translation QA")
st.markdown(
    """
**AI-powered translation quality assurance** with multi-agent evaluation system.

[![GitHub](https://img.shields.io/badge/GitHub-kttc--ai%2Fkttc-blue?logo=github)](https://github.com/kttc-ai/kttc)
[![PyPI](https://img.shields.io/pypi/v/kttc)](https://pypi.org/project/kttc/)
"""
)

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# Mode selection
use_demo_mode = st.sidebar.checkbox(
    "🎭 Demo Mode (no API key needed)", value=True, help="Use simulated responses for testing"
)

provider_type = "openai"
api_key = ""

if not use_demo_mode:
    st.sidebar.info("💡 Get your API key:")
    st.sidebar.markdown("- [OpenAI](https://platform.openai.com/api-keys)")
    st.sidebar.markdown("- [Anthropic](https://console.anthropic.com/)")

    provider_type = st.sidebar.selectbox(
        "LLM Provider", ["openai", "anthropic"], help="Choose your LLM provider"
    )

    api_key = st.sidebar.text_input(
        "API Key",
        type="password",
        help="Your OpenAI or Anthropic API key",
        placeholder="sk-..." if provider_type == "openai" else "sk-ant-...",
    )

    if not api_key:
        st.sidebar.warning("⚠️ Please enter your API key to use full mode")

# Language selection
st.sidebar.header("🌍 Languages")

language_options = {
    "English": "en",
    "Russian (Русский)": "ru",
    "Chinese (中文)": "zh",
    "Spanish (Español)": "es",
    "Hindi (हिन्दी)": "hi",
    "Persian (فارسی)": "fa",
    "French (Français)": "fr",
    "German (Deutsch)": "de",
}

source_lang_name = st.sidebar.selectbox("Source Language", list(language_options.keys()), index=0)
target_lang_name = st.sidebar.selectbox("Target Language", list(language_options.keys()), index=1)

source_lang = language_options[source_lang_name]
target_lang = language_options[target_lang_name]

# Main area - Input
st.header("📝 Input")

# Preset examples
example_presets = {
    "Custom (enter your own)": {"source": "", "translation": ""},
    "English → Russian (Good)": {
        "source": "Artificial intelligence is transforming the way we work and live. "
        "This technology opens up new possibilities for innovation.",
        "translation": "Искусственный интеллект трансформирует то, как мы работаем и живём. "
        "Эта технология открывает новые возможности для инноваций.",
    },
    "English → Russian (Bad)": {
        "source": "Artificial intelligence is transforming the way we work and live. "
        "This technology opens up new possibilities for innovation.",
        "translation": "Искусственный интеллект трансформирует как мы работать и жить. "
        "Это технология открывает возможности для инновация.",
    },
    "English → Chinese": {
        "source": "Machine learning algorithms can process vast amounts of data in seconds.",
        "translation": "机器学习算法可以在几秒钟内处理大量数据。",
    },
}

preset = st.selectbox("💡 Examples", list(example_presets.keys()))

col1, col2 = st.columns(2)

with col1:
    source_text = st.text_area(
        f"Source Text ({source_lang_name})",
        value=example_presets[preset]["source"],
        height=200,
        placeholder="Enter your source text here...",
    )

with col2:
    translation_text = st.text_area(
        f"Translation ({target_lang_name})",
        value=example_presets[preset]["translation"],
        height=200,
        placeholder="Enter the translation to evaluate...",
    )

# Evaluate button
if st.button("🔍 Evaluate Translation", type="primary", use_container_width=True):
    if not source_text or not translation_text:
        st.error("⚠️ Please enter both source text and translation")
    elif not use_demo_mode and not api_key:
        st.error("⚠️ Please enter your API key or enable demo mode")
    else:
        with st.spinner("🤖 AI agents are analyzing translation quality..."):
            try:
                # Get LLM provider
                llm_provider = get_llm_provider(use_demo_mode, provider_type, api_key)

                # Run evaluation
                report = asyncio.run(
                    evaluate_translation(
                        source_text, translation_text, source_lang, target_lang, llm_provider
                    )
                )

                # Display results
                st.header("📊 Results")

                # MQM Score
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("MQM Score", f"{report.mqm_score:.1f}/100")

                with col2:
                    status_color = "🟢" if report.status == "pass" else "🔴"
                    st.metric("Status", f"{status_color} {report.status.upper()}")

                with col3:
                    st.metric("Errors Found", len(report.errors))

                # Overall assessment
                if report.mqm_score >= 95:
                    st.success(
                        "✅ Excellent quality! This translation meets professional standards."
                    )
                elif report.mqm_score >= 85:
                    st.info("ℹ️ Good quality with minor issues. May need light editing.")
                elif report.mqm_score >= 70:
                    st.warning("⚠️ Acceptable quality with several issues. Editing recommended.")
                else:
                    st.error("❌ Poor quality. Significant revision needed.")

                # Errors breakdown
                if report.errors:
                    st.subheader("🔍 Issues Found")

                    for i, error in enumerate(report.errors, 1):
                        severity_color = {
                            "minor": "🟡",
                            "major": "🟠",
                            "critical": "🔴",
                        }.get(error.severity.name.lower(), "⚪")

                        with st.expander(
                            f"{severity_color} {i}. [{error.severity.name.upper()}] {error.category}"
                        ):
                            st.markdown(f"**Description:** {error.description}")

                            if error.suggestion:
                                st.markdown(f"**Suggestion:** {error.suggestion}")

                            if error.location:
                                st.markdown(
                                    f"**Location:** Characters {error.location[0]}-{error.location[1]}"
                                )
                else:
                    st.success("✨ No errors detected! Perfect translation.")

                # Mode indicator
                if use_demo_mode:
                    st.info(
                        "🎭 **Demo Mode:** Results are simulated. "
                        "Disable demo mode and add your API key for real evaluation."
                    )

            except Exception as e:
                st.error(f"❌ Error during evaluation: {str(e)}")
                st.exception(e)

# Footer
st.divider()
st.markdown(
    """
### 📚 Resources

- 📖 [Documentation](https://github.com/kttc-ai/kttc/tree/main/docs)
- 💻 [GitHub Repository](https://github.com/kttc-ai/kttc)
- 🐛 [Report Issues](https://github.com/kttc-ai/kttc/issues)
- 💬 [Discussions](https://github.com/kttc-ai/kttc/discussions)

### Install Locally

```bash
pip install kttc
export KTTC_OPENAI_API_KEY="your-key"
kttc check source.txt translation.txt
```

---

**Enjoying KTTC?** ⭐ [Star us on GitHub](https://github.com/kttc-ai/kttc)!
"""
)
